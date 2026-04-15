import asyncio
import cv2
import json
import langcodes
import os
import regex as re
import time
import torch
import logging
import sys
import traceback
import numpy as np
from PIL import Image
from typing import Optional, Any, List, Union
import py3langid as langid

from .config import Config, Colorizer, Detector, Translator, Renderer, Inpainter
from .utils import (
    BASE_PATH,
    LANGUAGE_ORIENTATION_PRESETS,
    ModelWrapper,
    Context,
    load_image,
    dump_image,
    visualize_textblocks,
    is_valuable_text,
    sort_regions,
)

from .detection import dispatch as dispatch_detection, prepare as prepare_detection, unload as unload_detection
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling, unload as unload_upscaling
from .ocr import dispatch as dispatch_ocr, prepare as prepare_ocr, unload as unload_ocr
from .textline_merge import dispatch as dispatch_textline_merge
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting, unload as unload_inpainting
from .translators import (
    dispatch as dispatch_translation,
    prepare as prepare_translation,
    unload as unload_translation,
)
from .translators.common import ISO_639_1_TO_VALID_LANGUAGES
from .colorization import dispatch as dispatch_colorization, prepare as prepare_colorization, unload as unload_colorization
from .rendering import dispatch as dispatch_rendering, dispatch_eng_render, dispatch_eng_render_pillow

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')

# 全域 console 實例，用於日誌重新導向
_global_console = None
_log_console = None

def set_main_logger(l):
    global logger
    logger = l

class TranslationInterrupt(Exception):
    """
    Can be raised from within a progress hook to prematurely terminate
    the translation.
    """
    pass

def load_dictionary(file_path):
    dictionary = []
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                # Ignore empty lines and lines starting with '#' or '//'
                if not line.strip() or line.strip().startswith('#') or line.strip().startswith('//'):
                    continue
                # Remove comment parts
                line = line.split('#')[0].strip()
                line = line.split('//')[0].strip()
                parts = line.split()
                if len(parts) == 1:
                    # If there is only the left part, the right part defaults to an empty string, meaning delete the left part
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, '', line_number))
                elif len(parts) == 2:
                    # If both left and right parts are present, perform the replacement
                    pattern = re.compile(parts[0])
                    dictionary.append((pattern, parts[1], line_number))
                else:
                    logger.error(f'Invalid dictionary entry at line {line_number}: {line.strip()}')
    return dictionary

def apply_dictionary(text, dictionary):
    for pattern, value, line_number in dictionary:
        original_text = text  
        text = pattern.sub(value, text)
        if text != original_text:  
            logger.info(f'Line {line_number}: Replaced "{original_text}" with "{text}" using pattern "{pattern.pattern}" and value "{value}"')
    return text

class MangaTranslator:
    verbose: bool
    ignore_errors: bool
    _gpu_limited_memory: bool
    device: Optional[str]
    kernel_size: Optional[int]
    models_ttl: int
    _progress_hooks: list[Any]
    result_sub_folder: str
    batch_size: int

    def __init__(self, params: dict = None):
        params = params or {}
        self.pre_dict = params.get('pre_dict', None)
        self.post_dict = params.get('post_dict', None)
        self.font_path = None
        self.use_mtpe = False
        self.kernel_size = None
        self.device = None
        self._gpu_limited_memory = False
        self.ignore_errors = False
        self.verbose = False
        self.models_ttl = 0
        self.batch_size = 1  # 預設不批次處理

        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        
        self._batch_contexts = []  # 儲存批次處理的上下文
        self._batch_configs = []   # 儲存批次處理的配置
        self.disable_memory_optimization = params.get('disable_memory_optimization', False)
        # batch_concurrent 會在 parse_init_params 中驗證並設置
        self.batch_concurrent = params.get('batch_concurrent', False)
        self.batch_all = params.get('batch_all', False)
        
        self.parse_init_params(params)
        self.result_sub_folder = ''

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        self._model_usage_timestamps = {}
        self._detector_cleanup_task = None
        self.prep_manual = params.get('prep_manual', None)
        self.context_size = params.get('context_size', 0)
        self.all_page_translations = []
        self._original_page_texts = []  # 儲存原文頁面資料，用於並發模式下的上下文

        # 除錯圖片管理相關屬性
        self._current_image_context = None  # 儲存當前處理圖片的上下文資訊
        self._saved_image_contexts = {}     # 儲存批次處理中每張圖片的上下文資訊
        
        # 設置日誌檔案
        self._setup_log_file()

    def _setup_log_file(self):
        """設置日誌檔案，在 result 資料夾下建立帶時間戳記的 log 檔案"""
        try:
            # 建立 result 目錄
            result_dir = os.path.join(BASE_PATH, 'result')
            os.makedirs(result_dir, exist_ok=True)
            
            # 生成带时间戳的日志文件名
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            log_filename = f"log_{timestamp}.txt"
            log_path = os.path.join(result_dir, log_filename)
            
            # 配置文件日志处理器
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            # 使用自定义格式器，保持与控制台输出一致
            from .utils.log import Formatter
            formatter = Formatter()
            file_handler.setFormatter(formatter)
            
            # 添加到manga-translator根logger以捕获所有输出
            mt_logger = logging.getLogger('manga-translator')
            mt_logger.addHandler(file_handler)
            if not mt_logger.level or mt_logger.level > logging.DEBUG:
                mt_logger.setLevel(logging.DEBUG)
            
            # 保存日志文件路径供后续使用
            self._log_file_path = log_path
            
            # 简单的print重定向
            import builtins
            original_print = builtins.print
            
            def log_print(*args, **kwargs):
                # 正常打印到控制台
                original_print(*args, **kwargs)
                # 同时写入日志文件
                try:
                    import io
                    buffer = io.StringIO()
                    original_print(*args, file=buffer, **kwargs)
                    output = buffer.getvalue()
                    if output.strip():
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(output)
                except Exception:
                    pass
            
            builtins.print = log_print
            
            # Rich Console输出重定向
            try:
                from rich.console import Console
                import sys
                
                # 创建一个自定义的文件对象，同时写入控制台和日志文件
                class TeeFile:
                    def __init__(self, log_file_path, original_file):
                        self.log_file_path = log_file_path
                        self.original_file = original_file
                    
                    def write(self, text):
                        # 写入原始输出
                        self.original_file.write(text)
                        # 写入日志文件
                        try:
                            if text.strip():
                                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                                    f.write(text)
                        except Exception:
                            pass
                        return len(text)
                    
                    def flush(self):
                        self.original_file.flush()
                    
                    def __getattr__(self, name):
                        return getattr(self.original_file, name)
                
                # 建立一個僅用於日誌紀錄的 Console（無顏色、無樣式）
                class LogOnlyFile:
                    def __init__(self, log_file_path):
                        self.log_file_path = log_file_path
                    
                    def write(self, text):
                        try:
                            if text.strip():
                                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                                    f.write(text)
                        except Exception:
                            pass
                        return len(text)
                    
                    def flush(self):
                        pass
                    
                    def isatty(self):
                        return False
                
                # 為日誌建立純文字 console
                log_file_only = LogOnlyFile(log_path)
                log_console = Console(file=log_file_only, force_terminal=False, no_color=True, width=80)
                
                # 建立帶顏色的主控台 console
                display_console = Console(force_terminal=True)
                
                # 全域設定 console 實例，供 translator 使用
                global _global_console, _log_console
                _global_console = display_console  # 主控台顯示用
                _log_console = log_console         # 日誌紀錄用
                
            except Exception as e:
                logger.debug(f"Failed to setup rich console logging: {e}")
            
            logger.info(f"Log file created: {log_path}")
        except Exception as e:
            print(f"Failed to setup log file: {e}")

    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        self.use_mtpe = params.get('use_mtpe', False)
        self.font_path = params.get('font_path', None)
        self.models_ttl = params.get('models_ttl', 0)
        self.batch_size = params.get('batch_size', 1)  # 新增批次大小參數
        
        # 批量處理全體 (batch_all) 模式下，batch_size 會被設置為總圖片數量
        self.batch_all = params.get('batch_all', False)
        
        # 驗證 batch_concurrent 參數
        if self.batch_concurrent and self.batch_size < 2 and not self.batch_all:
            logger.warning('--batch-concurrent requires --batch-size to be at least 2. When batch_size is 1, concurrent mode has no effect.')
            logger.info('Suggestion: Use --batch-size 2 (or higher) with --batch-concurrent, or remove --batch-concurrent flag.')
            # 自動停用並發模式
            self.batch_concurrent = False
            
        self.ignore_errors = params.get('ignore_errors', False)
        # check mps for apple silicon or cuda for nvidia
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        self.device = device if params.get('use_gpu', False) else 'cpu'
        self._gpu_limited_memory = params.get('use_gpu_limited', False)
        if self._gpu_limited_memory and not self.using_gpu:
            self.device = device
        if self.using_gpu and ( not torch.cuda.is_available() and not torch.backends.mps.is_available()):
            raise Exception(
                'CUDA or Metal compatible device could not be found in torch whilst --use-gpu args was set.\n'
                'Is the correct pytorch version installed? (See https://pytorch.org/)')
        if params.get('model_dir'):
            ModelWrapper._MODEL_DIR = params.get('model_dir')
        #todo: fix why is kernel size loaded in the constructor
        self.kernel_size = int(params.get('kernel_size')) if params.get('kernel_size') else None
        # Set input files
        self.input_files = params.get('input', [])
        # Set save_text
        self.save_text = params.get('save_text', False)
        # Set load_text
        self.load_text = params.get('load_text', False)
        
        # batch_concurrent 已在初始化時設置並驗證
        

        
    def _set_image_context(self, config: Config, image=None, image_name: str = None):
        """設置當前處理圖片的上下文資訊，用於產生除錯圖片子資料夾"""
        from .utils.generic import get_image_md5

        # 使用毫秒級時間戳記確保唯一性
        timestamp = str(int(time.time() * 1000))
        detection_size = str(getattr(config.detector, 'detection_size', 1024))
        target_lang = getattr(config.translator, 'target_lang', 'unknown')
        translator = getattr(config.translator, 'translator', 'unknown')

        # 如果提供 image_name，提取原始檔名 (不含擴展名)
        if image_name:
            # 取得檔名主體
            base_filename = os.path.splitext(os.path.basename(image_name))[0]
            # 產生子資料夾名：{base_filename}-{target_lang}-{translator}
            subfolder_name = f"{base_filename}-{target_lang}-{translator}"
        else:
            # 計算圖片 MD5 哈希值作為後備
            if image is not None:
                file_md5 = get_image_md5(image)
            else:
                file_md5 = "unknown"
            # 產生子資料夾名：{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}
            subfolder_name = f"{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}"

        self._current_image_context = {
            'subfolder': subfolder_name,
            'file_md5': get_image_md5(image) if image is not None else "unknown",
            'config': config
        }
        
    def _get_image_subfolder(self) -> str:
        """獲取當前圖片的除錯子資料夾名"""
        if self._current_image_context:
            return self._current_image_context['subfolder']
        return ''
    
    def _save_current_image_context(self, image_md5: str):
        """儲存當前圖片上下文，用於批次處理中保持一致性"""
        if self._current_image_context:
            self._saved_image_contexts[image_md5] = self._current_image_context.copy()

    def _restore_image_context(self, image_md5: str):
        """還原儲存的圖片上下文"""
        if image_md5 in self._saved_image_contexts:
            self._current_image_context = self._saved_image_contexts[image_md5].copy()
            return True
        return False

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    async def translate(self, image: Image.Image, config: Config, image_name: str = None, skip_context_save: bool = False, 
                        stop_at_ocr: bool = False, translation_list: List[str] = None) -> Context:
        """
        Translates a single image.

        :param image: Input image.
        :param config: Translation config.
        :param image_name: Deprecated parameter, kept for compatibility.
        :param stop_at_ocr: If True, stop after OCR and return context with text regions.
        :param translation_list: If provided, use these translations instead of calling a translator.
        :return: Translation context.
        """
        await self._report_progress('running_pre_translation_hooks')
        for hook in self._progress_hooks:
            try:
                hook('running_pre_translation_hooks', False)
            except Exception as e:
                logger.error(f"Error in progress hook: {e}")

        ctx = Context()
        ctx.input = image
        ctx.result = None
        ctx.verbose = self.verbose
        ctx.image_name = image_name

        # 設置圖片上下文以產生除錯圖片子資料夾
        self._set_image_context(config, image, image_name)
        
        # 儲存 debug 資料夾資訊到 Context 中（用於 Web 模式的快取存取）
        # 在 web 模式下總是儲存，不僅僅是 verbose 模式
        ctx.debug_folder = self._get_image_subfolder()
        
        # 儲存原始輸入圖片用於除錯
        if self.verbose:
            try:
                input_img = np.array(image)

                if len(input_img.shape) == 3:
                    channels = input_img.shape[2]    
                    if channels == 3:
                        # 標準 RGB
                        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                    elif channels == 4:
                        # 帶透明度的 RGBA
                        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGBA2BGR)
                    elif channels == 2:
                        # 灰階+透明
                        input_img = cv2.cvtColor(input_img[:, :, 0], cv2.COLOR_GRAY2BGR)
                elif len(input_img.shape) == 2:
                    # 純灰階圖 (只有高、寬兩個維度)
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

                result_path = self._result_path('input.png')
                success = cv2.imwrite(result_path, input_img)
                if not success:
                    logger.warning(f"Failed to save debug image: {result_path}")
            except Exception as e:
                logger.error(f"Error saving input.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # preload and download models (not strictly necessary, remove to lazy load)
        if ( self.models_ttl == 0 ):
            logger.info('Loading models')
            if config.upscale.upscale_ratio:
                await prepare_upscaling(config.upscale.upscaler)
            await prepare_detection(config.detector.detector)
            await prepare_ocr(config.ocr.ocr, self.device)
            await prepare_inpainting(config.inpainter.inpainter, self.device)
            await prepare_translation(config.translator.translator_gen)
            if config.colorizer.colorizer != Colorizer.none:
                await prepare_colorization(config.colorizer.colorizer)

        # translate
        ctx = await self._translate(config, ctx, stop_at_ocr=stop_at_ocr, translation_list=translation_list)

        # 在翻譯流程的最後儲存翻譯結果, 確保儲存的是最終結果 (including retry results)
        # Save translation results at the end of translation process to ensure final results are saved
        if not skip_context_save and ctx.text_regions:
            # 彙總本頁翻譯，供下一頁做上文
            page_translations = {r.text_raw if hasattr(r, "text_raw") else r.text: r.translation
                                 for r in ctx.text_regions}
            self.all_page_translations.append(page_translations)

            # 同時儲存原文用於並發模式的上下文
            page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                  for i, r in enumerate(ctx.text_regions)}
            self._original_page_texts.append(page_original_texts)

        return ctx

    async def _translate(self, config: Config, ctx: Context, stop_at_ocr: bool = False, translation_list: List[str] = None) -> Context:
        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())
        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as e:  
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.img_colorized = ctx.input  # Fallback to input image if colorization fails

        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as e:  
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.upscaled = ctx.img_colorized # Fallback to colorized (or input) image if upscaling fails
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as e:  
            logger.error(f"Error during detection:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] 
            ctx.mask_raw = None
            ctx.mask = None

        if self.verbose and ctx.mask_raw is not None:
            cv2.imwrite(self._result_path('mask_raw.png'), ctx.mask_raw)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            cv2.imwrite(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        # -- OCR
        await self._report_progress('ocr')
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
            
            # Web模式優化：將 OCR 辨識到的原文傳送給前端，並在背景預先載入模型
            if hasattr(self, '_is_streaming_mode') and self._is_streaming_mode:
                # 在等待翻譯結果回傳時，可以先進行下一張圖片的辨識準備（預加載模型）
                if (self.models_ttl > 0):
                    asyncio.create_task(prepare_translation(config.translator.translator_gen))
                
                if ctx.textlines:
                    # 彙整所有 textline 的文字
                    ocr_texts = [line.text for line in ctx.textlines]
                    import json
                    # 使用 status 碼 1 (狀態更新) 發送特殊前綴的消息
                    await self._report_progress(f"ocr_data:{json.dumps(ocr_texts, ensure_ascii=False)}")
                else:
                    await self._report_progress("ocr_data:[]")
            
            if stop_at_ocr:
                return ctx
        except Exception as e:  
            logger.error(f"Error during ocr:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] # Fallback to empty textlines if OCR fails

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as e:  
            logger.error(f"Error during textline_merge:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = [] # Fallback to empty text_regions if textline merge fails

        if self.verbose and ctx.text_regions:
            show_panels = not config.force_simple_sort  # 當不使用簡單排序時顯示 panel
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions, 
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        # 在文本行合併後套用翻譯前字典 (pre-dictionary)
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            original = region.text  
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("Pre-translation replacements:")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("No pre-translation replacements made.")
            
        # -- 翻譯
        await self._report_progress('translating')
        try:
            ctx.text_regions = await self._run_text_translation(config, ctx, translation_list=translation_list)
        except Exception as e:  
            logger.error(f"Error during translating:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = [] # 如果翻譯失敗則回退到空列表 (text_regions)

        await self._report_progress('after-translating')

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- 遮罩優化 (Mask refinement)
        # (延遲執行，以便利用 OCR 和翻譯後完成的區域篩選結果)
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as e:  
                logger.error(f"Error during mask-generation:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(ctx.img_rgb, dtype=np.uint8)[:,:,0] # 回退到原始遮罩或空遮罩

        if self.verbose and ctx.mask is not None:
            inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                          self.device, self.verbose)
            cv2.imwrite(self._result_path('inpaint_input.png'), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('mask_final.png'), ctx.mask)

        # -- 修補 (Inpainting)
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)
        except Exception as e:  
            logger.error(f"Error during inpainting:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise
            else:
                ctx.img_inpainted = ctx.img_rgb
        ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png')
                success = cv2.imwrite(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))
                if not success:
                    logger.warning(f"Failed to save debug image: {inpainted_path}")
            except Exception as e:
                logger.error(f"Error saving inpainted.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
        # -- 渲染 (Rendering)
        await self._report_progress('rendering')

        # 在 rendering 狀態後立即傳送資料夾資訊，用於前端精確檢查 final.png
        if hasattr(self, '_progress_hooks') and self._current_image_context:
            folder_name = self._current_image_context['subfolder']
            # 傳送特殊格式的訊息，前端可以解析
            await self._report_progress(f'rendering_folder:{folder_name}')

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.img_rendered = ctx.img_inpainted # 如果渲染失敗則回退到修補後（或原始 RGB）圖片

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)

        return await self._revert_upscale(config, ctx)
    
    # If `revert_upscaling` is True, revert to input size
    # Else leave `ctx` as-is
    async def _revert_upscale(self, config: Config, ctx: Context):
        if config.upscale.revert_upscaling:
            await self._report_progress('downscaling')
            ctx.result = ctx.result.resize(ctx.input.size)

        # 在 verbose 模式下儲存 final.png 到除錯資料夾
        if ctx.result and self.verbose:
            try:
                final_img = np.array(ctx.result)
                if len(final_img.shape) == 3:  # 彩色圖片，轉換 BGR 順序
                    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
                final_path = self._result_path('final.png')
                success = cv2.imwrite(final_path, final_img)
                if not success:
                    logger.warning(f"Failed to save debug image: {final_path}")
            except Exception as e:
                logger.error(f"Error saving final.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # Web 串流模式優化：儲存 final.png 並使用佔位符
        if ctx.result and not self.result_sub_folder and (getattr(self, '_is_streaming_mode', False) or (config and getattr(config, '_web_frontend_optimized', False))):
            # 儲存 final.png 檔案
            final_img = np.array(ctx.result)
            if len(final_img.shape) == 3:  # 彩色圖片，轉換 BGR 順序
                final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(self._result_path('final.png'), final_img)

            # 儲存原始檔名
            if hasattr(ctx, 'image_name') and ctx.image_name:
                try:
                    with open(self._result_path('original_name.txt'), 'w', encoding='utf-8') as f:
                        f.write(ctx.image_name)
                except Exception as e:
                    logger.error(f"Error saving original_name.txt: {e}")

            # 通知前端檔案已就緒
            if hasattr(self, '_progress_hooks') and self._current_image_context:
                folder_name = self._current_image_context['subfolder']
                await self._report_progress(f'final_ready:{folder_name}')

            # 建立佔位符結果並立即返回
            from PIL import Image
            placeholder = Image.new('RGB', (1, 1), color='white')
            ctx.result = placeholder
            ctx.use_placeholder = True
            return ctx

        return ctx

    async def _run_colorizer(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("colorizer", config.colorizer.colorizer)] = current_time
        #todo: im pretty sure the ctx is never used. does it need to be passed in?
        return await dispatch_colorization(
            config.colorizer.colorizer,
            colorization_size=config.colorizer.colorization_size,
            denoise_sigma=config.colorizer.denoise_sigma,
            device=self.device,
            image=ctx.input,
            **ctx
        )

    async def _run_upscaling(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("upscaling", config.upscale.upscaler)] = current_time
        return (await dispatch_upscaling(config.upscale.upscaler, [ctx.img_colorized], config.upscale.upscale_ratio, self.device))[0]

    async def _run_detection(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("detection", config.detector.detector)] = current_time
        result = await dispatch_detection(config.detector.detector, ctx.img_rgb, config.detector.detection_size, config.detector.text_threshold,
                                        config.detector.box_threshold,
                                        config.detector.unclip_ratio, config.detector.det_invert, config.detector.det_gamma_correct, config.detector.det_rotate,
                                        config.detector.det_auto_rotate,
                                        self.device, self.verbose)        
        return result

    async def _unload_model(self, tool: str, model: str):
        logger.info(f"Unloading {tool} model: {model}")
        match tool:
            case 'colorization':
                await unload_colorization(model)
            case 'detection':
                await unload_detection(model)
            case 'inpainting':
                await unload_inpainting(model)
            case 'ocr':
                await unload_ocr(model)
            case 'upscaling':
                await unload_upscaling(model)
            case 'translation':
                await unload_translation(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # empty CUDA cache

    # Background models cleanup job.
    async def _detector_cleanup_job(self):
        while True:
            if self.models_ttl == 0:
                await asyncio.sleep(1)
                continue
            now = time.time()
            for (tool, model), last_used in list(self._model_usage_timestamps.items()):
                if now - last_used > self.models_ttl:
                    await self._unload_model(tool, model)
                    del self._model_usage_timestamps[(tool, model)]
            await asyncio.sleep(1)

    async def _run_ocr(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("ocr", config.ocr.ocr)] = current_time
        
        # 為 OCR 建立子資料夾（只在 verbose 模式下）
        if self.verbose:
            image_subfolder = self._get_image_subfolder()
            if image_subfolder:
                if self.result_sub_folder:
                    ocr_result_dir = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, 'ocrs')
                else:
                    ocr_result_dir = os.path.join(BASE_PATH, 'result', image_subfolder, 'ocrs')
                os.makedirs(ocr_result_dir, exist_ok=True)
            else:
                ocr_result_dir = os.path.join(BASE_PATH, 'result', self.result_sub_folder, 'ocrs')
                os.makedirs(ocr_result_dir, exist_ok=True)
        else:
            # 非 verbose 模式下使用暫存目錄或不建立 OCR 結果目錄
            ocr_result_dir = None
        
        # 暫時設定環境變數供 OCR 模組使用
        old_ocr_dir = os.environ.get('MANGA_OCR_RESULT_DIR', None)
        if ocr_result_dir:
            os.environ['MANGA_OCR_RESULT_DIR'] = ocr_result_dir
        
        try:
            textlines = await dispatch_ocr(config.ocr.ocr, ctx.img_rgb, ctx.textlines, config.ocr, self.device, self.verbose)
        finally:
            # 還原環境變數
            if old_ocr_dir is not None:
                os.environ['MANGA_OCR_RESULT_DIR'] = old_ocr_dir
            elif 'MANGA_OCR_RESULT_DIR' in os.environ:
                del os.environ['MANGA_OCR_RESULT_DIR']

        new_textlines = []
        for textline in textlines:
            if textline.text.strip():
                if config.render.font_color_fg:
                    textline.fg_r, textline.fg_g, textline.fg_b = config.render.font_color_fg
                if config.render.font_color_bg:
                    textline.bg_r, textline.bg_g, textline.bg_b = config.render.font_color_bg
                new_textlines.append(textline)
        return new_textlines

    async def _run_textline_merge(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("textline_merge", "textline_merge")] = current_time
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],
                                                     verbose=self.verbose)
        for region in text_regions:
            if not hasattr(region, "text_raw"):
                region.text_raw = region.text      # <- 儲存原始 OCR 結果以擴展渲染偵測框。同時，避免影響禁用翻譯功能。
        # 濾除要跳過的語言
        if config.translator.skip_lang is not None:  
            skip_langs = [lang.strip().upper() for lang in config.translator.skip_lang.split(',')]  
            filtered_textlines = []  
            for txtln in ctx.textlines:  
                try:  
                    detected_lang, confidence = langid.classify(txtln.text)
                    source_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
                    if source_language != 'UNKNOWN':
                        source_language = source_language.upper()
                except Exception:  
                    source_language = 'UNKNOWN'  
    
                # 輸出偵測到的 source_language 以及是否在 skip_langs 中
                # logger.info(f'偵測到的來源語言：{source_language}，是否在 skip_langs 中：{source_language in skip_langs}，文字："{txtln.text}"')  
    
                if source_language in skip_langs:  
                    logger.info(f'已過濾：{txtln.text}')  
                    logger.info(f'原因：偵測到的語言 {source_language} 在 skip_langs 列表中')  
                    continue  # 跳過此區域  
                filtered_textlines.append(txtln)  
            ctx.textlines = filtered_textlines  
    
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],  
                                                     verbose=self.verbose)  

        new_text_regions = []
        for region in text_regions:
            # Remove leading spaces after pre-translation dictionary replacement                
            original_text = region.text  
            stripped_text = original_text.strip()  
            
            # Record removed leading characters  
            removed_start_chars = original_text[:len(original_text) - len(stripped_text)]  
            if removed_start_chars:  
                logger.info(f'Removed leading characters: "{removed_start_chars}" from "{original_text}"')  
            
            # Modified filtering condition: handle incomplete parentheses  
            bracket_pairs = {  
                '(': ')', '（': '）', '[': ']', '【': '】', '{': '}', '〔': '〕', '〈': '〉', '「': '」',  
                '"': '"', '＂': '＂', "'": "'", "“": "”", '《': '》', '『': '』', '"': '"', '〝': '〞', '﹁': '﹂', '﹃': '﹄',  
                '⸂': '⸃', '⸄': '⸅', '⸉': '⸊', '⸌': '⸍', '⸜': '⸝', '⸠': '⸡', '‹': '›', '«': '»', '＜': '＞', '<': '>'  
            }   
            left_symbols = set(bracket_pairs.keys())  
            right_symbols = set(bracket_pairs.values())  
            
            has_brackets = any(s in stripped_text for s in left_symbols) or any(s in stripped_text for s in right_symbols)  
            
            if has_brackets:  
                result_chars = []  
                stack = []  
                to_skip = []    
                
                # 第一次遍歷：標記匹配的括號  
                # First traversal: mark matching brackets
                for i, char in enumerate(stripped_text):  
                    if char in left_symbols:  
                        stack.append((i, char))  
                    elif char in right_symbols:  
                        if stack:  
                            # 有對應的左括號，出棧  
                            # There is a corresponding left bracket, pop the stack
                            stack.pop()  
                        else:  
                            # 沒有對應的左括號，標記為刪除  
                            # No corresponding left parenthesis, marked for deletion
                            to_skip.append(i)  
                
                # 標記未匹配的左括號為刪除
                # Mark unmatched left brackets as delete  
                for pos, _ in stack:  
                    to_skip.append(pos)  
                
                has_removed_symbols = len(to_skip) > 0  
                
                # 第二次遍歷：處理匹配但不對應的括號
                # Second pass: Process matching but mismatched brackets
                stack = []  
                for i, char in enumerate(stripped_text):  
                    if i in to_skip:  
                        # 跳過孤立的括號
                        # Skip isolated parentheses
                        continue  
                        
                    if char in left_symbols:  
                        stack.append(char)  
                        result_chars.append(char)  
                    elif char in right_symbols:  
                        if stack:  
                            left_bracket = stack.pop()  
                            expected_right = bracket_pairs.get(left_bracket)  
                            
                            if char != expected_right:  
                                # 替換不匹配的右括號為對應左括號的正確右括號
                                # Replace mismatched right brackets with the correct right brackets corresponding to the left brackets
                                result_chars.append(expected_right)  
                                logger.info(f'修正不匹配括號：將 "{char}" 替換為 "{expected_right}"')  
                            else:  
                                result_chars.append(char)  
                    else:  
                        result_chars.append(char)  
                
                new_stripped_text = ''.join(result_chars)  
                
                if has_removed_symbols:  
                    logger.info(f'從 "{stripped_text}" 中移除未配對的括號')  
                
                if new_stripped_text != stripped_text and not has_removed_symbols:  
                    logger.info(f'修正括號："{stripped_text}" → "{new_stripped_text}"')  
                
                stripped_text = new_stripped_text  
              
            region.text = stripped_text.strip()     
            
            if len(region.text) < config.ocr.min_text_length \
                    or not is_valuable_text(region.text) \
                    or (not config.translator.no_text_lang_skip and langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0):
                if region.text.strip():
                    logger.info(f'已過濾：{region.text}')
                    if len(region.text) < config.ocr.min_text_length:
                        logger.info('原因：文字長度小於要求的最小長度。')
                    elif not is_valuable_text(region.text):
                        logger.info('原因：該文字被認為不具備有效價值。')
                    elif langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0:
                        logger.info('原因：文字語言與目標語言相匹配，且 no_text_lang_skip 為 False。')
            else:
                if config.render.font_color_fg or config.render.font_color_bg:
                    if config.render.font_color_bg:
                        region.adjust_bg_color = False
                new_text_regions.append(region)
        text_regions = new_text_regions

        text_regions = sort_regions(
            text_regions,
            right_to_left=config.render.rtl,
            img=ctx.img_rgb,
            force_simple_sort=config.force_simple_sort
        )   
        
        return text_regions

    def _build_prev_context(self, use_original_text=False, current_page_index=None, batch_index=None, batch_original_texts=None):
        """
        跳过句子数为0的页面，取最近 context_size 个非空页面，拼成：
        <|1|>句子
        <|2|>句子
        ...
        的格式；如果没有任何非空页面，返回空串。

        Args:
            use_original_text: 是否使用原文而不是譯文作為上下文 (context)
            current_page_index: 當前頁面索引，用於確定上下文範圍
            batch_index: 當前頁面在批次中的索引
            batch_original_texts: 當前批次的原文資料
        """
        if self.context_size <= 0:
            return ""

        # 在並行模式下，需要特殊處理上下文範圍
        if batch_index is not None and batch_original_texts is not None:
            # 並行模式：使用已完成的頁面 + 當前批次中已處理的頁面
            available_pages = self.all_page_translations.copy()

            # 添加當前批次中在當前頁面之前的頁面
            for i in range(batch_index):
                if i < len(batch_original_texts) and batch_original_texts[i]:
                    # 在並行模式下，我們使用原文作為"已完成"的頁面
                    if use_original_text:
                        available_pages.append(batch_original_texts[i])
                    else:
                        # 如果不使用原文，則跳過當前批次的頁面（因為它們還沒有翻譯完成）
                        pass
        elif current_page_index is not None:
            # 使用指定頁面索引之前的頁面作為上下文
            available_pages = self.all_page_translations[:current_page_index] if self.all_page_translations else []
        else:
            # 使用所有已完成的頁面
            available_pages = self.all_page_translations or []

        if not available_pages:
            return ""

        # 篩選出有句子的頁面
        non_empty_pages = [
            page for page in available_pages
            if any(sent.strip() for sent in page.values())
        ]
        # 實際要用的頁數
        pages_used = min(self.context_size, len(non_empty_pages))
        if pages_used == 0:
            return ""
        tail = non_empty_pages[-pages_used:]

        # 拼接 - 根據參數決定使用原文還是譯文
        lines = []
        for page in tail:
            for sent in page.values():
                if sent.strip():
                    lines.append(sent.strip())

        # 如果使用原文，需要從原始資料中獲取
        if use_original_text and hasattr(self, '_original_page_texts'):
            # 嘗試獲取對應的原文
            original_lines = []
            for i, page in enumerate(tail):
                page_idx = available_pages.index(page)
                if page_idx < len(self._original_page_texts):
                    original_page = self._original_page_texts[page_idx]
                    for sent in original_page.values():
                        if sent.strip():
                            original_lines.append(sent.strip())
            if original_lines:
                lines = original_lines

        numbered = [f"<|{i+1}|>{s}" for i, s in enumerate(lines)]
        context_type = "original text" if use_original_text else "translation results"
        return f"Here are the previous {context_type} for reference:\n" + "\n".join(numbered)

    async def _dispatch_with_context(self, config: Config, texts: list[str], ctx: Context):
        # 計算實際要使用的上下文頁數和跳過的空頁數
        # Calculate the actual number of context pages to use and empty pages to skip
        done_pages = self.all_page_translations
        if self.context_size > 0 and done_pages:
            pages_expected = min(self.context_size, len(done_pages))
            non_empty_pages = [
                page for page in done_pages
                if any(sent.strip() for sent in page.values())
            ]
            pages_used = min(self.context_size, len(non_empty_pages))
            skipped = pages_expected - pages_used
        else:
            pages_used = skipped = 0

        if self.context_size > 0:
            logger.info(f"Context-aware translation enabled with {self.context_size} pages of history")

        # 建立上下文字串
        # Build the context string
        prev_ctx = self._build_prev_context()

        # 如果是 ChatGPT 或 ChatGPT2Stage 翻譯器，則專門處理上下文注入
        # Special handling for ChatGPT and ChatGPT2Stage translators: inject context
        if config.translator.translator in [Translator.chatgpt, Translator.chatgpt_2stage]:
            if config.translator.translator == Translator.chatgpt:
                from .translators.chatgpt import OpenAITranslator
                translator = OpenAITranslator()
            else:  # chatgpt_2stage
                from .translators.chatgpt_2stage import ChatGPT2StageTranslator
                translator = ChatGPT2StageTranslator()
                
            translator.parse_args(config.translator)
            translator.set_prev_context(prev_ctx)

            if pages_used > 0:
                context_count = prev_ctx.count("<|")
                logger.info(f"Carrying {pages_used} pages of context, {context_count} sentences as translation reference")
            if skipped > 0:
                logger.warning(f"Skipped {skipped} pages with no sentences")
                

            
            # ChatGPT2Stage 需要传递 ctx 参数，普通 ChatGPT 不需要
            if config.translator.translator == Translator.chatgpt_2stage:
                # 添加result_path_callback到Context，让translator可以保存bboxes_fixed.png
                ctx.result_path_callback = self._result_path
                return await translator._translate(ctx.from_lang, config.translator.target_lang, texts, ctx)
            else:
                return await translator._translate(ctx.from_lang, config.translator.target_lang, texts)


        return await dispatch_translation(
            config.translator.translator_gen,
            texts,
            config.translator,
            self.use_mtpe,
            ctx,
            'cpu' if self._gpu_limited_memory else self.device
        )

    async def _run_text_translation(self, config: Config, ctx: Context, translation_list: List[str] = None):
        # 檢查 text_regions 是否為 None 或空
        if not ctx.text_regions:
            return []

        if translation_list is not None:
            logger.info(f"Using provided translation list ({len(translation_list)} items)")
            # 將傳入的譯文分配給 text_regions
            for i, region in enumerate(ctx.text_regions):
                if i < len(translation_list):
                    region.translation = translation_list[i]
                else:
                    region.translation = ""
            return ctx.text_regions
            
        # 如果設置了 prep_manual 則將 translator 設置為 none，防止 token 浪費
        # Set translator to none to provent token waste if prep_manual is True  
        if self.prep_manual:  
            config.translator.translator = Translator.none
    
        current_time = time.time()
        self._model_usage_timestamps[("translation", config.translator.translator)] = current_time

        # 為 none 翻譯器添加特殊處理  
        # Add special handling for none translator  
        if config.translator.translator == Translator.none:  
            # 使用 none 翻譯器時，為所有文本區域設置必要的屬性  
            # When using none translator, set necessary properties for all text regions  
            for region in ctx.text_regions:  
                region.translation = ""  # 空翻譯將建立空白區域 / Empty translation will create blank areas  
                region.target_lang = config.translator.target_lang  
                region._alignment = config.render.alignment  
                region._direction = config.render.direction    
            return ctx.text_regions  

        # 以下翻譯處理僅在非 none 翻譯器或有 none 翻譯器但沒有 prep_manual 時執行  
        # Translation processing below only happens for non-none translator or none translator without prep_manual  
        if self.load_text:  
            input_filename = os.path.splitext(os.path.basename(self.input_files[0]))[0]  
            with open(self._result_path(f"{input_filename}_translations.txt"), "r") as f:  
                    translated_sentences = json.load(f)  
        else:  
            # 如果是 none 翻譯器，不需要調用翻譯服務，文本已經設置為空  
            # If using none translator, no need to call translation service, text is already set to empty  
            if config.translator.translator != Translator.none:  
                # 自動給 ChatGPT 加上下文，其他翻譯器不改變
                # Automatically add context to ChatGPT, no change for other translators
                texts = [region.text for region in ctx.text_regions]
                translated_sentences = \
                    await self._dispatch_with_context(config, texts, ctx)
            else:  
                # 對於 none 翻譯器，建立一個空翻譯列表  
                # For none translator, create an empty translation list  
                translated_sentences = ["" for _ in ctx.text_regions]  

            # Save translation if args.save_text is set and quit  
            if self.save_text:  
                input_filename = os.path.splitext(os.path.basename(self.input_files[0]))[0]  
                with open(self._result_path(f"{input_filename}_translations.txt"), "w") as f:  
                    json.dump(translated_sentences, f, indent=4, ensure_ascii=False)  
                print("Don't continue if --save-text is used")  
                exit(-1)  

        # 如果不是 none 翻譯器或者是 none 翻譯器但沒有 prep_manual  
        # If not none translator or none translator without prep_manual  
        if config.translator.translator != Translator.none or not self.prep_manual:  
            for region, translation in zip(ctx.text_regions, translated_sentences):  
                if config.render.uppercase:  
                    translation = translation.upper()  
                elif config.render.lowercase:  
                    translation = translation.lower()  # 修正：應該是 lower 而不是 upper  
                region.translation = translation  
                region.target_lang = config.translator.target_lang  
                region._alignment = config.render.alignment  
                region._direction = config.render.direction  

        # Punctuation correction logic. for translators often incorrectly change quotation marks from the source language to those commonly used in the target language.
        check_items = [
            # 圆括号处理
            ["(", "（", "「", "【"],
            ["（", "(", "「", "【"],
            [")", "）", "」", "】"],
            ["）", ")", "」", "】"],
            
            # 方括号处理
            ["[", "［", "【", "「"],
            ["［", "[", "【", "「"],
            ["]", "］", "】", "」"],
            ["］", "]", "】", "」"],
            
            # 引号处理
            ["「", "“", "‘", "『", "【"],
            ["」", "”", "’", "』", "】"],
            ["『", "“", "‘", "「", "【"],
            ["』", "”", "’", "」", "】"],
            
            # 新增【】处理
            ["【", "(", "（", "「", "『", "["],
            ["】", ")", "）", "」", "』", "]"],
        ]

        replace_items = [
            ["「", "“"],
            ["「", "‘"],
            ["」", "”"],
            ["」", "’"],
            ["【", "["],  
            ["】", "]"],  
        ]

        for region in ctx.text_regions:
            if region.text and region.translation:
                if '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                elif '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '【' in region.text and '】' in region.text: 
                    quote_type = '【】'
                else:
                    quote_type = None
                
                if quote_type:
                    src_quote_count = region.text.count(quote_type[0])
                    dst_dquote_count = region.translation.count('"')
                    dst_fwquote_count = region.translation.count('＂')
                    
                    if (src_quote_count > 0 and
                        (src_quote_count == dst_dquote_count or src_quote_count == dst_fwquote_count) and
                        not region.translation.isascii()):
                        
                        if quote_type == '「」':
                            region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                        elif quote_type == '『』':
                            region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
                        elif quote_type == '【】':  
                            region.translation = re.sub(r'"([^"]*)"', r'【\1】', region.translation)

                # === 優化後的數量判斷邏輯 ===
                for v in check_items:
                    num_src_std = region.text.count(v[0])
                    num_src_var = sum(region.text.count(t) for t in v[1:])
                    num_dst_std = region.translation.count(v[0])
                    num_dst_var = sum(region.translation.count(t) for t in v[1:])
                    
                    if (num_src_std > 0 and
                        num_src_std != num_src_var and
                        num_src_std == num_dst_std + num_dst_var):
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])

                # 強制替換規則
                # Forced replacement rules
                for v in replace_items:
                    region.translation = region.translation.replace(v[1], v[0])

        # 注意：翻譯結果的儲存移動到了翻譯流程的最後，確保儲存的是最終結果而不是重試前的結果

        # Apply post dictionary after translating
        post_dict = load_dictionary(self.post_dict)
        post_replacements = []  
        for region in ctx.text_regions:  
            original = region.translation  
            region.translation = apply_dictionary(region.translation, post_dict)
            if original != region.translation:  
                post_replacements.append(f"{original} => {region.translation}")  

        if post_replacements:  
            logger.info("Post-translation replacements:")  
            for replacement in post_replacements:  
                logger.info(replacement)  
        else:  
            logger.info("No post-translation replacements made.")

        # 譯後檢查和重試邏輯 - 第一階段：單個 region 幻覺檢測
        failed_regions = []
        if config.translator.enable_post_translation_check:
            logger.info("Starting post-translation check...")
            
            # 單個 region 級別的幻覺檢測（在篩選前進行）
            for region in ctx.text_regions:
                if region.translation and region.translation.strip():
                    # 只檢查重複內容幻覺，不進行頁面級目標語言檢查
                    if await self._check_repetition_hallucination(
                        region.translation, 
                        config.translator.post_check_repetition_threshold,
                        silent=False
                    ):
                        failed_regions.append(region)
            
            # 對失敗的區域進行重試
            if failed_regions:
                logger.warning(f"Found {len(failed_regions)} regions that failed repetition check, starting retry...")
                for region in failed_regions:
                    await self._retry_translation_with_validation(region, config, ctx)
                logger.info("Repetition check retry finished.")

        # 譯後檢查和重試邏輯 - 第二階段：頁面級目標語言檢查（使用篩選後的區域）
        if config.translator.enable_post_translation_check:
            # 修正：不要進行頁面級目標語言檢查，避免將已翻譯的中文誤判為日文導致重複翻譯
            # logger.info("Skipping page-level target language check as requested to avoid incorrect Japanese/Chinese detection.")
            # page_lang_check_result = True
            # if ctx.text_regions and len(ctx.text_regions) > 10:
            #     logger.info(f"Starting page-level target language check with {len(ctx.text_regions)} regions...")
            #     page_lang_check_result = await self._check_target_language_ratio(
            #         ctx.text_regions,
            #         config.translator.target_lang,
            #         min_ratio=0.3
            #     )
                
            #     if not page_lang_check_result:
            #         logger.warning("Page-level target language ratio check failed")
                    
            #         # 第二阶段：整个批次重新翻译逻辑
            #         max_batch_retry = config.translator.post_check_max_retry_attempts
            #         batch_retry_count = 0
                    
            #         while batch_retry_count < max_batch_retry and not page_lang_check_result:
            #             batch_retry_count += 1
            #             logger.warning(f"Starting batch retry {batch_retry_count}/{max_batch_retry} for page-level target language check...")
                        
            #             # 重新翻译所有区域
            #             original_texts = []
            #             for region in ctx.text_regions:
            #                 if hasattr(region, 'text') and region.text:
            #                     original_texts.append(region.text)
            #                 else:
            #                     original_texts.append("")
                        
            #             if original_texts:
            #                 try:
            #                     # 重新批次翻譯
            #                     logger.info(f"Retrying translation for {len(original_texts)} regions...")
            #                     new_translations = await self._batch_translate_texts(original_texts, config, ctx)
                                
            #                     # 更新翻译结果到regions
            #                     for i, region in enumerate(ctx.text_regions):
            #                         if i < len(new_translations) and new_translations[i]:
            #                             old_translation = region.translation
            #                             region.translation = new_translations[i]
            #                             logger.debug(f"Region {i+1} translation updated: '{old_translation}' -> '{new_translations[i]}'")
                                    
            #                     # 重新检查目标语言比例
            #                     logger.info(f"Re-checking page-level target language ratio after batch retry {batch_retry_count}...")
            #                     page_lang_check_result = await self._check_target_language_ratio(
            #                         ctx.text_regions,
            #                         config.translator.target_lang,
            #                         min_ratio=0.5
            #                     )
                                
            #                     if page_lang_check_result:
            #                         logger.info(f"Page-level target language check passed")
            #                         break
            #                     else:
            #                         logger.warning(f"Page-level target language check still failed")
                                    
            #                 except Exception as e:
            #                     logger.error(f"Error during batch retry {batch_retry_count}: {e}")
            #                     break
            #             else:
            #                 logger.warning("No text found for batch retry")
            #                 break
                    
            #         if not page_lang_check_result:
            #             logger.error(f"Page-level target language check failed after all {max_batch_retry} batch retries")
            #     else:
            #         logger.info("Page-level target language ratio check passed")
            # else:
            #     logger.info(f"Skipping page-level target language check: only {len(ctx.text_regions)} regions (threshold: 5)")
            
            # 统一的成功信息
            logger.info("Skipping page-level target language check as requested to avoid incorrect Japanese/Chinese detection.")
            page_lang_check_result = True
            
            # 使用统一的成功信息
            if page_lang_check_result:
                logger.info("All translation regions passed post-translation check.")
            else:
                logger.warning("Some translation regions failed post-translation check.")

        # 过滤逻辑（简化版本，保留主要过滤条件）
        new_text_regions = []
        for region in ctx.text_regions:
            should_filter = False
            filter_reason = ""

            if not region.translation.strip():
                should_filter = True
                filter_reason = "Translation contain blank areas"
            elif config.translator.translator != Translator.none:
                if region.translation.isnumeric():
                    should_filter = True
                    filter_reason = "Numeric translation"
                elif config.filter_text and re.search(config.re_filter_text, region.translation):
                    should_filter = True
                    filter_reason = f"Matched filter text: {config.filter_text}"
                elif not config.translator.translator == Translator.original:
                    text_equal = region.text.lower().strip() == region.translation.lower().strip()
                    if text_equal:
                        should_filter = True
                        filter_reason = "Translation identical to original"

            if should_filter:
                if region.translation.strip():
                    logger.info(f'Filtered out: {region.translation}')
                    logger.info(f'Reason: {filter_reason}')
            else:
                new_text_regions.append(region)

        return new_text_regions

    async def _run_mask_refinement(self, config: Config, ctx: Context):
        """
        執行遮罩優化 (Mask Refinement)。
        """
        return await dispatch_mask_refinement(ctx.text_regions, ctx.img_rgb, ctx.mask_raw, 'fit_text',
                                              config.mask_dilation_offset, config.ocr.ignore_bubble, self.verbose,self.kernel_size)

    async def _run_inpainting(self, config: Config, ctx: Context):
        """
        執行圖片修補 (Inpainting)。
        """
        current_time = time.time()
        self._model_usage_timestamps[("inpainting", config.inpainter.inpainter)] = current_time
        return await dispatch_inpainting(config.inpainter.inpainter, ctx.img_rgb, ctx.mask, config.inpainter, config.inpainter.inpainting_size, self.device,
                                         self.verbose)

    async def _run_text_rendering(self, config: Config, ctx: Context):
        """
        執行文字渲染 (Rendering)。
        """
        current_time = time.time()
        self._model_usage_timestamps[("rendering", config.render.renderer)] = current_time
        if config.render.renderer == Renderer.none:
            output = ctx.img_inpainted
        # manga2eng 目前僅支援水平從左到右的渲染
        # manga2eng currently only supports horizontal left to right rendering
        elif (config.render.renderer == Renderer.manga2Eng or config.render.renderer == Renderer.manga2EngPillow) and ctx.text_regions and LANGUAGE_ORIENTATION_PRESETS.get(ctx.text_regions[0].target_lang) == 'h':
            if config.render.renderer == Renderer.manga2EngPillow:
                output = await dispatch_eng_render_pillow(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, config.render.line_spacing)
            else:
                output = await dispatch_eng_render(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, self.font_path, config.render.line_spacing)
        else:
            output = await dispatch_rendering(ctx.img_inpainted, ctx.text_regions, self.font_path, config.render.font_size,
                                              config.render.font_size_offset,
                                              config.render.font_size_minimum, not config.render.no_hyphenation, ctx.render_mask, config.render.line_spacing)
        return output

    def _result_path(self, path: str) -> str:
        """
        返回結果資料夾的路徑，用於儲存 verbose 模式下的中間圖片
        或 web 模式下快取的輸入/結果圖片。
        """
        # 僅在 verbose 模式下才使用圖片級子資料夾
        if self.verbose:
            image_subfolder = self._get_image_subfolder()
            if image_subfolder:
                if self.result_sub_folder:
                    result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, path)
                else:
                    result_path = os.path.join(BASE_PATH, 'result', image_subfolder, path)
                # 確保目錄存在
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                return result_path
        
        # 在 server/web 模式下（result_sub_folder 為空）且為非 verbose 模式時
        # 需要建立一個子資料夾來儲存 final.png
        if not self.result_sub_folder:
            if self._current_image_context:
                # 直接使用已產生的子資料夾名
                sub_folder = self._current_image_context['subfolder']
            else:
                # 沒有上下文資訊時使用預設值
                timestamp = str(int(time.time() * 1000))
                sub_folder = f"{timestamp}-unknown-1024-unknown-unknown"

            result_path = os.path.join(BASE_PATH, 'result', sub_folder, path)
        else:
            result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)
        
        # 確保目錄存在
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        return result_path

    def add_progress_hook(self, ph):
        """
        新增進度鉤子 (Progress Hook)。
        """
        self._progress_hooks.append(ph)

    async def _report_progress(self, state: str, finished: bool = False):
        """
        上報翻譯進度。
        """
        for ph in self._progress_hooks:
            await ph(state, finished)

    def _add_logger_hook(self):
        """
        新增日誌鉤子 (Logger Hook)。
        """
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling': 'Running upscaling',
            'detection': 'Running text detection',
            'ocr': 'Running ocr',
            'mask-generation': 'Running mask refinement',
            'translating': 'Running text translation',
            'rendering': 'Running rendering',
            'colorizing': 'Running colorization',
            'downscaling': 'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions': 'No text regions! - Skipping',
            'skip-no-text': 'No text regions with text! - Skipping',
            'error-translating': 'Text translator returned empty queries',
            'cancelled': 'Image translation cancelled',
        }
        LOG_MESSAGES_ERROR = {
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    async def translate_batch(self, items: List[Union[tuple, dict]], batch_size: int = None, image_names: List[str] = None) -> List[Context]:
        """
        批次翻譯多張圖片，在翻譯階段進行批次處理以提高效率。
        
        Args:
            items: (圖片, 配置) 元組的列表，或是包含 image, config, image_name 的字典列表。
            batch_size: 批次大小，如果為 None 則使用實例的 batch_size。
            image_names: 已棄用的參數，保留用於相容性。
            
        Returns:
            包含翻譯結果的 Context 物件列表。
        """
        # 標準化輸入
        images_with_configs = []
        for item in items:
            if isinstance(item, dict):
                images_with_configs.append((item['image'], item['config'], item.get('image_name')))
            else:
                images_with_configs.append((item[0], item[1], None))

        batch_size = batch_size or self.batch_size
        if batch_size <= 1:
            # 不使用批次處理時，回到原有的逐個處理方式
            logger.debug('Batch size <= 1, switching to individual processing mode')
            results = []
            for i, (image, config, name) in enumerate(images_with_configs):
                ctx = await self.translate(image, config, image_name=name)  # 單頁翻譯時正常儲存上下文
                results.append(ctx)
            return results
        
        logger.debug(f'Starting batch translation: {len(images_with_configs)} images, batch size: {batch_size}')
        
        # 簡化的記憶體檢查
        memory_optimization_enabled = not self.disable_memory_optimization
        if not memory_optimization_enabled:
            logger.debug('Memory optimization disabled for batch translation')
        
        results = []
        
        # 處理所有圖片到翻譯前的步驟
        logger.debug('Starting pre-processing phase...')
        pre_translation_contexts = []
        
        for i, (image, config, name) in enumerate(images_with_configs):
            logger.debug(f'Pre-processing image {i+1}/{len(images_with_configs)}')
            
            # 如果啟用了 batch_all，強制在此步驟將 translator 設為 none，避免單頁翻譯
            original_translator = config.translator.translator
            if self.batch_all:
                config.translator.translator = Translator.none

            # 簡化的記憶體檢查
            if memory_optimization_enabled:
                try:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 85:
                        logger.warning(f'High memory usage during pre-processing: {memory_percent:.1f}%')
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except ImportError:
                    pass  # psutil 不可用時忽略
                except Exception as e:
                    logger.debug(f'Memory check failed: {e}')
                
            try:
                # 為批次處理中的每張圖片設定上下文
                self._set_image_context(config, image, name)
                # 儲存圖片上下文，確保後處理階段使用相同的資料夾
                if self._current_image_context:
                    image_md5 = self._current_image_context['file_md5']
                    self._save_current_image_context(image_md5)
                ctx = await self._translate_until_translation(image, config, image_name=name)

                # 如果啟用了 batch_all，還原翻譯器設定供後續批次階段使用
                if self.batch_all:
                    config.translator.translator = original_translator

                # 儲存圖片上下文到 Context 物件中，用於後續批次處理
                if self._current_image_context:
                    ctx.image_context = self._current_image_context.copy()
                # 儲存 verbose 標誌到 Context 物件中
                ctx.verbose = self.verbose
                pre_translation_contexts.append((ctx, config))
                logger.debug(f'Image {i+1} pre-processing successful')
            except MemoryError as e:
                logger.error(f'Memory error in pre-processing image {i+1}: {e}')
                if not memory_optimization_enabled:
                    logger.error('Consider enabling memory optimization')
                    raise
                    
                # 嘗試降級處理
                try:
                    logger.warning(f'Image {i+1} attempting fallback processing...')
                    import copy
                    recovery_config = copy.deepcopy(config)
                    
                    # 強制清理
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 重新設定圖片上下文
                    self._set_image_context(recovery_config, image)
                    # 如果啟用了 batch_all，強制在此步驟將 translator 設為 none
                    if self.batch_all:
                        recovery_config.translator.translator = Translator.none

                    # 儲存 fallback 圖片上下文
                    if self._current_image_context:
                        image_md5 = self._current_image_context['file_md5']
                        self._save_current_image_context(image_md5)
                    ctx = await self._translate_until_translation(image, recovery_config)

                    # 還原翻譯器設定
                    if self.batch_all:
                        recovery_config.translator.translator = original_translator

                    # 儲存圖片上下文到 Context 物件中
                    if self._current_image_context:
                        ctx.image_context = self._current_image_context.copy()
                    # 儲存 verbose 標誌到 Context 物件中
                    ctx.verbose = self.verbose
                    pre_translation_contexts.append((ctx, recovery_config))
                    logger.info(f'Image {i+1} fallback processing successful')
                except Exception as retry_error:
                    logger.error(f'Image {i+1} fallback processing also failed: {retry_error}')
                    # 建立空 context 作為佔位符
                    ctx = Context()
                    ctx.input = image
                    ctx.text_regions = []  # 確保 text_regions 被初始化為空列表
                    pre_translation_contexts.append((ctx, config))
            except Exception as e:
                logger.error(f'Image {i+1} pre-processing error: {e}')
                # 建立空 context 作為佔位符
                ctx = Context()
                ctx.input = image
                ctx.text_regions = []  # 確保 text_regions 被初始化為空列表
                pre_translation_contexts.append((ctx, config))
        
        if not pre_translation_contexts:
            logger.warning('No images pre-processed successfully')
            return results
            
        logger.debug(f'Pre-processing completed: {len(pre_translation_contexts)} images')
            
        # 批次翻譯處理
        logger.debug('Starting batch translation phase...')
        try:
            if self.batch_concurrent:
                logger.info(f'Using concurrent mode for batch translation')
                translated_contexts = await self._concurrent_translate_contexts(pre_translation_contexts)
            else:
                logger.debug(f'Using standard batch mode for translation')
                translated_contexts = await self._batch_translate_contexts(pre_translation_contexts, batch_size)
        except MemoryError as e:
            logger.error(f'Memory error in batch translation: {e}')
            if not memory_optimization_enabled:
                logger.error('Consider enabling memory optimization')
                raise
                
            logger.warning('Batch translation failed, switching to individual page translation mode...')
            # 降級到每頁逐個翻譯
            translated_contexts = []
            for ctx, config in pre_translation_contexts:
                try:
                    if ctx.text_regions:  # 檢查 text_regions 是否不為 None 且不為空
                        # 對整頁進行翻譯處理
                        translated_texts = await self._batch_translate_texts([region.text for region in ctx.text_regions], config, ctx)
                        
                        # 將翻譯結果套用到各個 region
                        for region, translation in zip(ctx.text_regions, translated_texts):
                            region.translation = translation
                            region.target_lang = config.translator.target_lang
                            region._alignment = config.render.alignment
                            region._direction = config.render.direction
                    translated_contexts.append((ctx, config))
                    
                    # 每頁翻譯後都清理記憶體
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as individual_error:
                    logger.error(f'Individual page translation failed: {individual_error}')
                    translated_contexts.append((ctx, config))
        
        # 完成翻譯後的處理
        logger.debug('Starting post-processing phase...')
        for i, (ctx, config) in enumerate(translated_contexts):
            try:
                if ctx.text_regions:
                    # 還原預處理階段儲存的圖片上下文，確保使用相同的資料夾
                    # 透過圖片計算 MD5 來還原上下文
                    from .utils.generic import get_image_md5
                    image = ctx.input  # 從 context 中獲取原始圖片
                    image_md5 = get_image_md5(image)
                    if not self._restore_image_context(image_md5):
                        # 如果還原失敗，作為 fallback 重新設定（理論上不應該發生）
                        logger.warning(f"Failed to restore image context for MD5 {image_md5}, creating new context")
                        self._set_image_context(config, image)
                    ctx = await self._complete_translation_pipeline(ctx, config)
                results.append(ctx)
                logger.debug(f'Image {i+1} post-processing completed')
            except Exception as e:
                logger.error(f'Image {i+1} post-processing error: {e}')
                results.append(ctx)
        
        logger.info(f'Batch translation completed: processed {len(results)} images')

        # 批次處理完成後，儲存所有頁面的最終翻譯結果
        for ctx in results:
            if ctx.text_regions:
                # 彙總本頁翻譯，供下一頁做上文
                page_translations = {r.text_raw if hasattr(r, "text_raw") else r.text: r.translation
                                     for r in ctx.text_regions}
                self.all_page_translations.append(page_translations)

                # 同時儲存原文用於並發模式的上下文
                page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                      for i, r in enumerate(ctx.text_regions)}
                self._original_page_texts.append(page_original_texts)

        # 清理批次處理的圖片上下文快取
        self._saved_image_contexts.clear()
        
        return results

    async def _translate_until_translation(self, image: Image.Image, config: Config, image_name: str = None) -> Context:
        """
        执行翻译之前的所有步骤（彩色化、上采样、检测、OCR、文本行合并）
        """
        ctx = Context()
        ctx.input = image
        ctx.result = None
        ctx.image_name = image_name
        
        # 儲存原始輸入圖片用於除錯
        if self.verbose:
            try:
                input_img = np.array(image)
                if len(input_img.shape) == 3:  # 彩色圖片，轉換 BGR 順序
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                result_path = self._result_path('input.png')
                success = cv2.imwrite(result_path, input_img)
                if not success:
                    logger.warning(f"無法儲存除錯圖片：{result_path}")
            except Exception as e:
                logger.error(f"儲存 input.png 除錯圖片時出錯：{e}")
                logger.debug(f"異常細節：\n{traceback.format_exc()}")

        # preload and download models (not strictly necessary, remove to lazy load)
        if ( self.models_ttl == 0 ):
            logger.info('Loading models')
            if config.upscale.upscale_ratio:
                await prepare_upscaling(config.upscale.upscaler)
            await prepare_detection(config.detector.detector)
            await prepare_ocr(config.ocr.ocr, self.device)
            await prepare_inpainting(config.inpainter.inpainter, self.device)
            await prepare_translation(config.translator.translator_gen)
            if config.colorizer.colorizer != Colorizer.none:
                await prepare_colorization(config.colorizer.colorizer)

        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())

        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as e:  
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.img_colorized = ctx.input
        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as e:  
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.upscaled = ctx.img_colorized
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        # -- 文字偵測 (Detection)
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as e:  
            logger.error(f"文字偵測出錯：\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] 
            ctx.mask_raw = None
            ctx.mask = None

        if self.verbose and ctx.mask_raw is not None:
            cv2.imwrite(self._result_path('mask_raw.png'), ctx.mask_raw)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            cv2.imwrite(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        # -- 光學字元辨識 (OCR)
        await self._report_progress('ocr')
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
        except Exception as e:  
            logger.error(f"OCR 出錯：\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = []

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- 文字行合併 (Textline merge)
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as e:  
            logger.error(f"文字行合併出錯：\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = []

        if self.verbose and ctx.text_regions:
            show_panels = not config.force_simple_sort  # 當不使用簡單排序時顯示區塊 (panel)
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions, 
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        # 文字行合併後套用前置字典 (Pre-dictionary)
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            original = region.text  
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("翻譯前替換內容：")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("未進行任何翻譯前替換。")

        # 儲存目前圖片上下文到 ctx 中，用於並行翻譯時的路徑管理
        if self._current_image_context:
            ctx.image_context = self._current_image_context.copy()

        return ctx

    async def _batch_translate_contexts(self, contexts_with_configs: List[tuple], batch_size: int) -> List[tuple]:
        """
        批次處理翻譯步驟，防止記憶體溢出。
        """
        results = []
        total_contexts = len(contexts_with_configs)
        
        # 按批次處理，防止記憶體溢出
        for i in range(0, total_contexts, batch_size):
            batch = contexts_with_configs[i:i + batch_size]
            logger.info(f'正在處理翻譯批次 {i//batch_size + 1}/{(total_contexts + batch_size - 1)//batch_size}')
            
            # 收集目前批次的所有文字
            all_texts = []
            batch_text_mapping = []  # 記錄每個文字屬於哪個 context 和 region
            
            for ctx_idx, (ctx, config) in enumerate(batch):
                if not ctx.text_regions:
                    continue
                    
                region_start_idx = len(all_texts)
                for region_idx, region in enumerate(ctx.text_regions):
                    all_texts.append(region.text)
                    batch_text_mapping.append((ctx_idx, region_idx))
                
            if not all_texts:
                # 目前批次沒有需要翻譯的文字
                results.extend(batch)
                continue
                
            # 批次翻譯
            try:
                await self._report_progress('translating')
                # 使用第一個配置進行翻譯（假設批次內配置相同）
                sample_config = batch[0][1] if batch else None
                if sample_config:
                    # 支援批次翻譯 - 傳遞所有批次上下文
                    batch_contexts = [ctx for ctx, config in batch]
                    translated_texts = await self._batch_translate_texts(all_texts, sample_config, batch[0][0], batch_contexts)
                else:
                    translated_texts = all_texts  # 無法翻譯時保持原文
                    
                # 將翻譯結果分配回各個 context
                text_idx = 0
                for ctx_idx, (ctx, config) in enumerate(batch):
                    if not ctx.text_regions:  # 檢查 text_regions 是否為 None 或空
                        continue
                    for region_idx, region in enumerate(ctx.text_regions):
                        if text_idx < len(translated_texts):
                            region.translation = translated_texts[text_idx]
                            region.target_lang = config.translator.target_lang
                            region._alignment = config.render.alignment
                            region._direction = config.render.direction
                            text_idx += 1
                        
                # 套用後處理邏輯（括號修正、過濾等）
                for ctx, config in batch:
                    if ctx.text_regions:
                        ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                        
                # 批次層級的目標語言檢查
                if batch and batch[0][1].translator.enable_post_translation_check:
                    # 收集批次內所有頁面的過濾後區域 (filtered regions)
                    all_batch_regions = []
                    for ctx, config in batch:
                        if ctx.text_regions:
                            all_batch_regions.extend(ctx.text_regions)
                    
                    # 進行批次層級的目標語言檢查
                    batch_lang_check_result = True
                    if all_batch_regions and len(all_batch_regions) > 10:
                        sample_config = batch[0][1]
                        logger.info(f"正在對 {len(all_batch_regions)} 個區域進行批次層級的目標語言檢查...")
                        batch_lang_check_result = await self._check_target_language_ratio(
                            all_batch_regions,
                            sample_config.translator.target_lang,
                            min_ratio=0.5
                        )
                        
                        if not batch_lang_check_result:
                            logger.warning("批次層級目標語言比例檢查失敗")
                            
                            # 批次重新翻譯邏輯
                            max_batch_retry = sample_config.translator.post_check_max_retry_attempts
                            batch_retry_count = 0
                            
                            while batch_retry_count < max_batch_retry and not batch_lang_check_result:
                                batch_retry_count += 1
                                logger.warning(f"正在開始批次重試 {batch_retry_count}/{max_batch_retry}")
                                
                                # 重新翻譯批次內所有區域
                                all_original_texts = []
                                region_mapping = []  # 記錄每個文字屬於哪個 ctx
                                
                                for ctx_idx, (ctx, config) in enumerate(batch):
                                    if ctx.text_regions:
                                        for region in ctx.text_regions:
                                            if hasattr(region, 'text') and region.text:
                                                all_original_texts.append(region.text)
                                                region_mapping.append((ctx_idx, region))
                                
                                if all_original_texts:
                                    try:
                                        # 重新批次翻譯
                                        logger.info(f"正在為 {len(all_original_texts)} 個區域重試翻譯...")
                                        new_translations = await self._batch_translate_texts(all_original_texts, sample_config, batch[0][0])
                                        
                                        # 更新翻譯結果到各個 region
                                        for i, (ctx_idx, region) in enumerate(region_mapping):
                                            if i < len(new_translations) and new_translations[i]:
                                                old_translation = region.translation
                                                region.translation = new_translations[i]
                                                logger.debug(f"區域 {i+1} 翻譯已更新：'{old_translation}' -> '{new_translations[i]}'")
                                        
                                        # 重新收集所有 regions 並檢查目標語言比例
                                        all_batch_regions = []
                                        for ctx, config in batch:
                                            if ctx.text_regions:
                                                all_batch_regions.extend(ctx.text_regions)
                                        
                                        logger.info(f"批次重試 {batch_retry_count} 後，重新檢查批次層級目標語言比例...")
                                        batch_lang_check_result = await self._check_target_language_ratio(
                                            all_batch_regions,
                                            sample_config.translator.target_lang,
                                            min_ratio=0.5
                                        )
                                        
                                        if batch_lang_check_result:
                                            logger.info(f"批次層級目標語言檢查通過")
                                            break
                                        else:
                                            logger.warning(f"批次層級目標語言檢查仍失敗")
                                            
                                    except Exception as e:
                                        logger.error(f"批次重試 {batch_retry_count} 時出錯：{e}")
                                        break
                                else:
                                    logger.warning("批次重試未找到文字")
                                    break
                            
                            if not batch_lang_check_result:
                                logger.error(f"在所有 {max_batch_retry} 次批次重試後，批次層級目標語言檢查仍失敗")
                    else:
                        logger.info(f"跳過批次層級目標語言檢查：僅有 {len(all_batch_regions)} 個區域 (門檻值：10)")
                    
                    # 統一的成功資訊
                    if batch_lang_check_result:
                        logger.info("所有翻譯區域均通過翻譯後檢查。")
                    else:
                        logger.warning("部分翻譯區域未通過翻譯後檢查。")
                        
                # 過濾邏輯（簡化版本，保留主要過濾條件）
                for ctx, config in batch:
                    if ctx.text_regions:
                        new_text_regions = []
                        for region in ctx.text_regions:
                            should_filter = False
                            filter_reason = ""

                            if not region.translation.strip():
                                should_filter = True
                                filter_reason = "翻譯包含空白區域"
                            elif config.translator.translator != Translator.none:
                                if region.translation.isnumeric():
                                    should_filter = True
                                    filter_reason = "數值翻譯"
                                elif config.filter_text and re.search(config.re_filter_text, region.translation):
                                    should_filter = True
                                    filter_reason = f"符合過濾文字：{config.filter_text}"
                                elif not config.translator.translator == Translator.original:
                                    text_equal = region.text.lower().strip() == region.translation.lower().strip()
                                    if text_equal:
                                        should_filter = True
                                        filter_reason = "翻譯與原文相同"

                            if should_filter:
                                if region.translation.strip():
                                    logger.info(f'已過濾：{region.translation}')
                                    logger.info(f'原因：{filter_reason}')
                            else:
                                new_text_regions.append(region)
                        ctx.text_regions = new_text_regions
                        
                results.extend(batch)
                
            except Exception as e:
                logger.error(f"批次翻譯出錯：{e}")
                if not self.ignore_errors:
                    raise
                # 出錯時保持原文
                for ctx, config in batch:
                    if not ctx.text_regions:  # 檢查 text_regions 是否為 None 或空
                        continue
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                results.extend(batch)
                
            # 強制垃圾回收以釋放記憶體
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results

    async def _concurrent_translate_contexts(self, contexts_with_configs: List[tuple]) -> List[tuple]:
        """
        並行處理翻譯步驟，為每張圖片單獨發送翻譯請求，避免合併大批次。
        """

        # 在並行模式下，先儲存所有頁面的原文用於上下文
        batch_original_texts = []  # 儲存目前批次的原文
        if self.context_size > 0:
            for i, (ctx, config) in enumerate(contexts_with_configs):
                if ctx.text_regions:
                    # 儲存目前頁面的原文
                    page_texts = {}
                    for j, region in enumerate(ctx.text_regions):
                        page_texts[j] = region.text
                    batch_original_texts.append(page_texts)

                    # 確保 _original_page_texts 有足夠的長度
                    while len(self._original_page_texts) <= len(self.all_page_translations) + i:
                        self._original_page_texts.append({})

                    self._original_page_texts[len(self.all_page_translations) + i] = page_texts
                else:
                    batch_original_texts.append({})

        async def translate_single_context(ctx_config_pair_with_index):
            """翻譯單個 context 的非同步函數"""
            ctx, config, page_index, batch_index = ctx_config_pair_with_index
            try:
                if not ctx.text_regions:
                    return ctx, config

                # 收集該 context 的所有文字
                texts = [region.text for region in ctx.text_regions]

                if not texts:
                    return ctx, config

                logger.debug(f'正在並行模式下為單張圖片翻譯 {len(texts)} 個區域 (頁面 {page_index}, 批次 {batch_index})')

                # 單獨翻譯這張圖片的文字，傳遞頁面索引和批次索引以獲取正確的上下文
                translated_texts = await self._batch_translate_texts(
                    texts, config, ctx,
                    page_index=page_index,
                    batch_index=batch_index,
                    batch_original_texts=batch_original_texts
                )

                # 將翻譯結果分配回各個 region
                for i, region in enumerate(ctx.text_regions):
                    if i < len(translated_texts):
                        region.translation = translated_texts[i]
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                
                # 套用後處理邏輯（括號修正、過濾等）
                if ctx.text_regions:
                    ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                
                # 單頁目標語言檢查（如果啟用）
                if config.translator.enable_post_translation_check and ctx.text_regions:
                    page_lang_check_result = await self._check_target_language_ratio(
                        ctx.text_regions,
                        config.translator.target_lang,
                        min_ratio=0.3  # 對單頁使用較寬鬆的門檻值
                    )
                    
                    if not page_lang_check_result:
                        logger.warning(f"單張圖片的頁面層級目標語言檢查失敗")
                        
                        # 單頁重試邏輯
                        max_retry = config.translator.post_check_max_retry_attempts
                        retry_count = 0
                        
                        while retry_count < max_retry and not page_lang_check_result:
                            retry_count += 1
                            logger.info(f"正在重試單張圖片翻譯 {retry_count}/{max_retry}")
                            
                            # 重新翻譯
                            original_texts = [region.text for region in ctx.text_regions if hasattr(region, 'text') and region.text]
                            if original_texts:
                                try:
                                    new_translations = await self._batch_translate_texts(original_texts, config, ctx)
                                    
                                    # 更新翻譯結果
                                    text_idx = 0
                                    for region in ctx.text_regions:
                                        if hasattr(region, 'text') and region.text and text_idx < len(new_translations):
                                            old_translation = region.translation
                                            region.translation = new_translations[text_idx]
                                            logger.debug(f"區域翻譯已更新：'{old_translation}' -> '{new_translations[text_idx]}'")
                                            text_idx += 1
                                    
                                    # 重新檢查
                                    page_lang_check_result = await self._check_target_language_ratio(
                                        ctx.text_regions,
                                        config.translator.target_lang,
                                        min_ratio=0.3
                                    )
                                    
                                    if page_lang_check_result:
                                        logger.info(f"重試 {retry_count} 次後，單張圖片目標語言檢查通過")
                                        break
                                        
                                except Exception as e:
                                    logger.error(f"單張圖片重試 {retry_count} 次時出錯：{e}")
                                    break
                            else:
                                break
                        
                        if not page_lang_check_result:
                            logger.warning(f"在所有 {max_retry} 次重試後，單張圖片目標語言檢查仍失敗")
                
                # 過濾邏輯
                if ctx.text_regions:
                    new_text_regions = []
                    for region in ctx.text_regions:
                        should_filter = False
                        filter_reason = ""

                        if not region.translation.strip():
                            should_filter = True
                            filter_reason = "翻譯包含空白區域"
                        elif config.translator.translator != Translator.none:
                            if region.translation.isnumeric():
                                should_filter = True
                                filter_reason = "數值翻譯"
                            elif config.filter_text and re.search(config.re_filter_text, region.translation):
                                should_filter = True
                                filter_reason = f"符合過濾文字：{config.filter_text}"
                            elif not config.translator.translator == Translator.original:
                                text_equal = region.text.lower().strip() == region.translation.lower().strip()
                                if text_equal:
                                    should_filter = True
                                    filter_reason = "翻譯與原文相同"

                        if should_filter:
                            if region.translation.strip():
                                logger.info(f'已過濾：{region.translation}')
                                logger.info(f'原因：{filter_reason}')
                        else:
                            new_text_regions.append(region)
                    ctx.text_regions = new_text_regions
                
                return ctx, config
                
            except Exception as e:
                logger.error(f"單張圖片並行翻譯時出錯：{e}")
                if not self.ignore_errors:
                    raise
                # 出錯時保持原文
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                return ctx, config
        
        # 建立並行任務，為每個任務加入頁面索引和批次索引
        tasks = []
        for i, ctx_config_pair in enumerate(contexts_with_configs):
            # 計算目前頁面在整個翻譯序列中的索引
            page_index = len(self.all_page_translations) + i
            batch_index = i  # 在目前批次中的索引
            ctx_config_pair_with_index = (*ctx_config_pair, page_index, batch_index)
            task = asyncio.create_task(translate_single_context(ctx_config_pair_with_index))
            tasks.append(task)
        
        logger.info(f'開始並行翻譯 {len(tasks)} 張圖片...')
        
        # 等待所有任務完成
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"並行翻譯 gather 時出錯：{e}")
            raise
        
        # 處理結果，檢查是否有異常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"第 {i+1} 張圖片並行翻譯失敗：{result}")
                if not self.ignore_errors:
                    raise result
                # 建立失敗的佔位符
                ctx, config = contexts_with_configs[i]
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.target_lang  # 這裡原代碼是 config.translator.target_lang，修正一下
                        if hasattr(config, 'translator'):
                            region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                final_results.append((ctx, config))
            else:
                final_results.append(result)
        
        logger.info(f'並行翻譯完成：已處理 {len(final_results)} 張圖片')
        return final_results

    async def _batch_translate_texts(self, texts: List[str], config: Config, ctx: Context, batch_contexts: List[Context] = None, page_index: int = None, batch_index: int = None, batch_original_texts: List[dict] = None) -> List[str]:
        """
        批次翻譯文字列表，使用現有的翻譯器接口。

        Args:
            texts: 要翻譯的文字列表。
            config: 配置物件。
            ctx: 上下文物件。
            batch_contexts: 批次處理上下文列表。
            page_index: 目前頁面索引，用於並行模式下的上下文計算。
            batch_index: 目前頁面在批次中的索引。
            batch_original_texts: 目前批次的原文資料。
        """
        if config.translator.translator == Translator.none:
            return ["" for _ in texts]



        # 如果是 ChatGPT 翻譯器（包括 chatgpt 和 chatgpt_2stage），需要處理上下文
        if config.translator.translator in [Translator.chatgpt, Translator.chatgpt_2stage]:
            if config.translator.translator == Translator.chatgpt:
                from .translators.chatgpt import OpenAITranslator
                translator = OpenAITranslator()
            else:  # chatgpt_2stage
                from .translators.chatgpt_2stage import ChatGPT2StageTranslator
                translator = ChatGPT2StageTranslator()

            # 確定是否使用並行模式和原文上下文
            use_original_text = self.batch_concurrent and self.batch_size > 1

            done_pages = self.all_page_translations
            if self.context_size > 0 and done_pages:
                pages_expected = min(self.context_size, len(done_pages))
                non_empty_pages = [
                    page for page in done_pages
                    if any(sent.strip() for sent in page.values())
                ]
                pages_used = min(self.context_size, len(non_empty_pages))
                skipped = pages_expected - pages_used
            else:
                pages_used = skipped = 0

            if self.context_size > 0:
                context_type = "原文" if use_original_text else "翻譯結果"
                logger.info(f"上下文感知翻譯已啟用，使用 {self.context_size} 頁歷史記錄（{context_type}）")

            translator.parse_args(config.translator)

            # 構建上下文 - 在並行模式下使用原文和頁面索引
            prev_ctx = self._build_prev_context(
                use_original_text=use_original_text,
                current_page_index=page_index,
                batch_index=batch_index,
                batch_original_texts=batch_original_texts
            )
            translator.set_prev_context(prev_ctx)

            if pages_used > 0:
                context_count = prev_ctx.count("<|")
                logger.info(f"攜帶 {pages_used} 頁上下文，{context_count} 個句子作為翻譯參考")
            if skipped > 0:
                logger.warning(f"跳過 {skipped} 頁空白頁面")

            # ChatGPT2Stage 需要特殊處理
            if config.translator.translator == Translator.chatgpt_2stage:
                # 為目前圖片建立專用的 result_path_callback，避免並行時路徑錯亂
                current_image_context = getattr(ctx, 'image_context', None) or self._current_image_context

                def result_path_callback(path: str) -> str:
                    """為特定圖片建立結果路徑，使用儲存的圖片上下文"""
                    original_context = self._current_image_context
                    self._current_image_context = current_image_context
                    try:
                        return self._result_path(path)
                    finally:
                        self._current_image_context = original_context

                ctx.result_path_callback = result_path_callback

                # 檢查是否啟用批次處理並提供了 batch_contexts
                if batch_contexts and len(batch_contexts) > 1 and not self.batch_concurrent:
                    # 為 chatgpt_2stage 啟用批次處理
                    ctx.batch_contexts = batch_contexts
                    logger.info(f"正在為 chatgpt_2stage 啟用 {len(batch_contexts)} 張圖片的批次處理")

                    # 為批次中的每個 context 設定 result_path_callback
                    for batch_ctx in batch_contexts:
                        if hasattr(batch_ctx, 'image_context'):
                            batch_image_context = batch_ctx.image_context
                        else:
                            batch_image_context = self._current_image_context

                        def create_result_path_callback(image_context):
                            def result_path_callback(path: str) -> str:
                                """為特定圖片建立結果路徑，使用儲存的圖片上下文"""
                                original_context = self._current_image_context
                                self._current_image_context = image_context
                                try:
                                    return self._result_path(path)
                                finally:
                                    self._current_image_context = original_context
                            return result_path_callback

                        batch_ctx.result_path_callback = create_result_path_callback(batch_image_context)

                # ChatGPT2Stage 需要傳遞 ctx 參數
                return await translator._translate(
                    ctx.from_lang,
                    config.translator.target_lang,
                    texts,
                    ctx
                )
            else:
                # 普通 ChatGPT 不需要 ctx 參數
                return await translator._translate(
                    ctx.from_lang,
                    config.translator.target_lang,
                    texts
                )

        else:
            # 使用通用翻譯排程器 (scheduler)
            return await dispatch_translation(
                config.translator.translator_gen,
                texts,
                config.translator,
                self.use_mtpe,
                ctx,
                'cpu' if self._gpu_limited_memory else self.device
            )
            
    async def _apply_post_translation_processing(self, ctx: Context, config: Config) -> List:
        """
        套用翻譯後處理邏輯（括號修正、過濾等）。
        """
        # 檢查 text_regions 是否為 None 或空
        if not ctx.text_regions:
            return []
            
        check_items = [
            # 圓括號處理
            ["(", "（", "「", "【"],
            ["（", "(", "「", "【"],
            [")", "）", "」", "】"],
            ["）", ")", "」", "】"],
            
            # 方括號處理
            ["[", "［", "【", "「"],
            ["［", "[", "【", "「"],
            ["]", "］", "】", "」"],
            ["］", "]", "】", "」"],
            
            # 引號處理
            ["「", "“", "‘", "『", "【"],
            ["」", "”", "’", "』", "】"],
            ["『", "“", "‘", "「", "【"],
            ["』", "”", "’", "」", "】"],
            
            # 新增【】處理
            ["【", "(", "（", "「", "『", "["],
            ["】", ")", "）", "」", "』", "]"],
        ]

        replace_items = [
            ["「", "“"],
            ["「", "‘"],
            ["」", "”"],
            ["」", "’"],
            ["【", "["],  
            ["】", "]"],  
        ]

        for region in ctx.text_regions:
            if region.text and region.translation:
                # 引號處理邏輯
                if '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                elif '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '【' in region.text and '】' in region.text: 
                    quote_type = '【】'
                else:
                    quote_type = None
                
                if quote_type:
                    src_quote_count = region.text.count(quote_type[0])
                    dst_dquote_count = region.translation.count('"')
                    dst_fwquote_count = region.translation.count('＂')
                    
                    if (src_quote_count > 0 and
                        (src_quote_count == dst_dquote_count or src_quote_count == dst_fwquote_count) and
                        not region.translation.isascii()):
                        
                        if quote_type == '「」':
                            region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                        elif quote_type == '『』':
                            region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
                        elif quote_type == '【】':  
                            region.translation = re.sub(r'"([^"]*)"', r'【\1】', region.translation)

                # 括號修正邏輯
                for v in check_items:
                    num_src_std = region.text.count(v[0])
                    num_src_var = sum(region.text.count(t) for t in v[1:])
                    num_dst_std = region.translation.count(v[0])
                    num_dst_var = sum(region.translation.count(t) for t in v[1:])
                    
                    if (num_src_std > 0 and
                        num_src_std != num_src_var and
                        num_src_std == num_dst_std + num_dst_var):
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])

                # 強制替換規則
                for v in replace_items:
                    region.translation = region.translation.replace(v[1], v[0])

        # 注意：翻譯結果的儲存已移動到 translate 方法的最後，以確保儲存的是最終結果

        # 套用後置字典 (Post-dictionary)
        post_dict = load_dictionary(self.post_dict)
        post_replacements = []  
        for region in ctx.text_regions:  
            original = region.translation  
            region.translation = apply_dictionary(region.translation, post_dict)
            if original != region.translation:  
                post_replacements.append(f"{original} => {region.translation}")  

        if post_replacements:  
            logger.info("翻譯後替換內容：")  
            for replacement in post_replacements:  
                logger.info(replacement)  
        else:  
            logger.info("未進行任何翻譯後替換。")

        # 單個區域 (region) 幻覺檢測
        failed_regions = []
        if config.translator.enable_post_translation_check:
            logger.info("正在開始翻譯後檢查...")
            
            # 單個區域層級的幻覺檢測
            for region in ctx.text_regions:
                if region.translation and region.translation.strip():
                    # 只檢查重複內容幻覺
                    if await self._check_repetition_hallucination(
                        region.translation, 
                        config.translator.post_check_repetition_threshold,
                        silent=False
                    ):
                        failed_regions.append(region)
            
            # 對失敗的區域進行重試
            if failed_regions:
                logger.warning(f"發現 {len(failed_regions)} 個區域未通過重複檢查，正在開始重試...")
                for region in failed_regions:
                    try:
                        logger.info(f"正在重試翻譯區域，原文為：'{region.text}'")
                        new_translation = await self._retry_translation_with_validation(region, config, ctx)
                        if new_translation:
                            old_translation = region.translation
                            region.translation = new_translation
                            logger.info(f"區域重試成功：'{old_translation}' -> '{new_translation}'")
                        else:
                            logger.warning(f"區域重試失敗，保持原樣：'{region.translation}'")
                    except Exception as e:
                        logger.error(f"區域重試時出錯：{e}")

        return ctx.text_regions

    async def _complete_translation_pipeline(self, ctx: Context, config: Config) -> Context:
        """
        完成翻譯後的處理步驟（掩碼細化、修補、渲染）。
        """
        await self._report_progress('after-translating')

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- 掩碼細化 (Mask refinement)
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as e:  
                logger.error(f"掩碼生成出錯：\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(ctx.img_rgb, dtype=np.uint8)[:,:,0]

        if self.verbose and ctx.mask is not None:
            try:
                inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter, config.inpainter.inpainting_size,
                                                              self.device, self.verbose)
                
                # 儲存 inpaint_input.png
                inpaint_input_path = self._result_path('inpaint_input.png')
                success1 = cv2.imwrite(inpaint_input_path, cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
                if not success1:
                    logger.warning(f"無法儲存除錯圖片：{inpaint_input_path}")
                
                # 儲存 mask_final.png
                mask_final_path = self._result_path('mask_final.png')
                success2 = cv2.imwrite(mask_final_path, ctx.mask)
                if not success2:
                    logger.warning(f"無法儲存除錯圖片：{mask_final_path}")
            except Exception as e:
                logger.error(f"儲存除錯圖片 (inpaint_input.png, mask_final.png) 時出錯：{e}")
                logger.debug(f"異常細節：\n{traceback.format_exc()}")

        # -- 圖像修補 (Inpainting)
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)

        except Exception as e:  
            logger.error(f"圖像修補出錯：\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise
            else:
                ctx.img_inpainted = ctx.img_rgb
        ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png')
                success = cv2.imwrite(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))
                if not success:
                    logger.warning(f"無法儲存除錯圖片：{inpainted_path}")
            except Exception as e:
                logger.error(f"儲存 inpainted.png 除錯圖片時出錯：{e}")
                logger.debug(f"異常細節：\n{traceback.format_exc()}")

        # -- 渲染 (Rendering)
        await self._report_progress('rendering')

        # 在 rendering 狀態後立即發送資料夾資訊，用於前端精確檢查 final.png
        if hasattr(self, '_progress_hooks') and self._current_image_context:
            folder_name = self._current_image_context['subfolder']
            # 發送特殊格式的訊息，前端可以解析
            await self._report_progress(f'rendering_folder:{folder_name}')

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as e:
            logger.error(f"渲染出錯：\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.img_rendered = ctx.img_inpainted

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)
        
        # 儲存除錯資料夾資訊到 Context 中（用於 Web 模式的快取存取）
        if self.verbose:
            ctx.debug_folder = self._get_image_subfolder()

        return await self._revert_upscale(config, ctx)
    
    async def _check_repetition_hallucination(self, text: str, threshold: int = 5, silent: bool = False) -> bool:
        """
        檢查文字是否包含重複內容（模型幻覺）。
        
        Args:
            text: 要檢查的文字。
            threshold: 重複次數門檻值。
            silent: 是否靜默模式（不輸出日誌）。
            
        Returns:
            bool: True 表示檢測到幻覺，False 表示正常。
        """
        if not text or len(text.strip()) < threshold:
            return False
            
        # 檢查字元級重複
        consecutive_count = 1
        prev_char = None
        
        for char in text:
            if char == prev_char:
                consecutive_count += 1
                if consecutive_count >= threshold:
                    if not silent:
                        logger.warning(f'檢測到字元重複幻覺："{text}" - 重複字元："{char}"，連續次數：{consecutive_count}')
                    return True
            else:
                consecutive_count = 1
            prev_char = char
        
        # 檢查詞語級重複（依字元分割中文，依空格分割其他語言）
        segments = re.findall(r'[\u4e00-\u9fff]|\S+', text)
        
        if len(segments) >= threshold:
            consecutive_segments = 1
            prev_segment = None
            
            for segment in segments:
                if segment == prev_segment:
                    consecutive_segments += 1
                    if consecutive_segments >= threshold:
                        if not silent:
                            logger.warning(f'檢測到詞語重複幻覺："{text}" - 重複片段："{segment}"，連續次數：{consecutive_segments}')
                        return True
                else:
                    consecutive_segments = 1
                prev_segment = segment
        
        # 檢查短語級重複
        words = text.split()
        if len(words) >= threshold * 2:
            for i in range(len(words) - threshold + 1):
                phrase = ' '.join(words[i:i + threshold//2])
                remaining_text = ' '.join(words[i + threshold//2:])
                if phrase in remaining_text:
                    phrase_count = text.count(phrase)
                    if phrase_count >= 3:  # 降低短語重複檢測門檻值
                        if not silent:
                            logger.warning(f'檢測到短語重複幻覺："{text}" - 重複短語："{phrase}"，出現次數：{phrase_count}')
                        return True
                        
        return False

    async def _check_target_language_ratio(self, text_regions: List, target_lang: str, min_ratio: float = 0.5) -> bool:
        """
        檢查翻譯結果中目標語言的佔比是否達到要求。
        使用 py3langid 進行語言檢測。
        
        Args:
            text_regions: 文字區域列表。
            target_lang: 目標語言代碼。
            min_ratio: 最小目標語言佔比（此參數在新邏輯中不使用，保留以供相容）。
            
        Returns:
            bool: True 表示通過檢查，False 表示未通過。
        """
        if not text_regions or len(text_regions) <= 10:
            # 如果區域數量不超過 10 個，跳過此檢查
            return True
            
        # 合併所有翻譯文字
        all_translations = []
        for region in text_regions:
            translation = getattr(region, 'translation', '')
            if translation and translation.strip():
                all_translations.append(translation.strip())
        
        if not all_translations:
            logger.debug('目標語言佔比檢查沒有有效的翻譯文字')
            return True
            
        # 將所有翻譯合併為一個文字進行檢測
        merged_text = ''.join(all_translations)

        try:
            detected_lang, confidence = langid.classify(merged_text)
            detected_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN').upper()
        except Exception as e:
            detected_language = 'UNKNOWN'

        # --- 關鍵修正：繁簡中文模糊匹配 ---
        target_lang_upper = target_lang.upper()
        
        # 定義「中文語系家族」
        chinese_variants = ['CHT', 'CHS', 'ZH', 'CHINESE']
        
        # 如果 目標語言 是中文，且 偵測結果 也是中文（不論繁簡）
        if target_lang_upper in chinese_variants and detected_language in ['CHS', 'CHT', 'CHINESE', 'ZH']:
            logger.info(f"檢測到中文變體匹配 (目標: {target_lang_upper}, 檢測: {detected_language})，判定為通過。")
            return True
        logger.debug(f'語言檢測結果：{detected_language}，置信度：{confidence}，是否包含假名：{has_kana}')

        # 使用 py3langid 進行語言檢測
        try:
            detected_lang, confidence = langid.classify(merged_text)
            detected_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
            if detected_language != 'UNKNOWN':
                detected_language = detected_language.upper()
        except Exception as e:
            logger.debug(f'對合併文字進行 py3langid 檢測失敗：{e}')
            detected_language = 'UNKNOWN'
            confidence = -9999
        
        # 檢查檢測出的語言是否為目標語言
        is_target_lang = (detected_language == target_lang.upper())
        
        return is_target_lang

    async def _validate_translation(self, original_text: str, translation: str, target_lang: str, config, ctx: Context = None, silent: bool = False, page_lang_check_result: bool = None) -> bool:
        """
        驗證翻譯品質（包含目標語言比例檢查和幻覺檢測）。
        
        Args:
            original_text: 原文。
            translation: 翻譯。
            target_lang: 目標語言。
            config: 配置物件。
            ctx: 上下文物件。
            silent: 是否靜默模式。
            page_lang_check_result: 頁面級目標語言檢查結果，如果為 None 則進行檢查，如果已有結果則直接使用。
        """
        if not config.translator.enable_post_translation_check:
            return True
            
        if not translation or not translation.strip():
            return True
        
        # 1. 目標語言比例檢查（頁面層級）
        if page_lang_check_result is None and ctx and ctx.text_regions and len(ctx.text_regions) > 10:
            # 進行頁面級目標語言檢查
            page_lang_check_result = await self._check_target_language_ratio(
                ctx.text_regions,
                target_lang,
                min_ratio=0.5
            )
            
        # 如果頁面級檢查失敗，直接返回失敗
        if page_lang_check_result is False:
            if not silent:
                logger.debug("該區域的目標語言比例檢查失敗")
            return False
        
        # 2. 檢查重複內容幻覺（區域層級）
        if await self._check_repetition_hallucination(
            translation, 
            config.translator.post_check_repetition_threshold,
            silent
        ):
            return False
                
        return True

    async def _retry_translation_with_validation(self, region, config: Config, ctx: Context) -> str:
        """
        帶驗證的重試翻譯。
        """
        original_translation = region.translation
        max_attempts = config.translator.post_check_max_retry_attempts
        
        for attempt in range(max_attempts):
            # 驗證目前翻譯 - 在重試過程中只檢查單個區域（幻覺檢測），不進行頁面級檢查
            is_valid = await self._validate_translation(
                region.text, 
                region.translation, 
                config.translator.target_lang,
                config,
                ctx=None,  # 不傳 ctx 以避免頁面級檢查
                silent=True,  # 重試過程中禁用日誌輸出
                page_lang_check_result=True  # 傳入 True 跳過頁面級檢查，只做區域級檢查
            )
            
            if is_valid:
                if attempt > 0:
                    logger.info(f'翻譯後檢查通過（第 {attempt + 1}/{max_attempts} 次嘗試）："{region.translation}"')
                return region.translation
            
            # 如果不是最後一次嘗試，進行重新翻譯
            if attempt < max_attempts - 1:
                logger.warning(f'翻譯後檢查失敗（第 {attempt + 1}/{max_attempts} 次嘗試），正在重新翻譯："{region.text}"')
                
                try:
                    # 單獨重新翻譯這個文字區域
                    if config.translator.translator != Translator.none:
                        from .translators import dispatch
                        retranslated = await dispatch(
                            config.translator.translator_gen,
                            [region.text],
                            config.translator,
                            self.use_mtpe,
                            ctx,
                            'cpu' if self._gpu_limited_memory else self.device
                        )
                        if retranslated:
                            region.translation = retranslated[0]
                            
                            # 套用格式化處理
                            if config.render.uppercase:
                                region.translation = region.translation.upper()
                            elif config.render.lowercase:
                                region.translation = region.translation.lower()
                                
                            logger.info(f'重新翻譯完成："{region.text}" -> "{region.translation}"')
                        else:
                            logger.warning(f'重新翻譯失敗，保持原翻譯："{original_translation}"')
                            region.translation = original_translation
                            break
                    else:
                        logger.warning('翻譯器為 None，無法重新翻譯。')
                        break
                        
                except Exception as e:
                    logger.error(f'重新翻譯時出錯：{e}')
                    region.translation = original_translation
                    break
            else:
                logger.warning(f'翻譯後檢查失敗，已達最大重試次數 ({max_attempts})，保持原翻譯："{original_translation}"')
                region.translation = original_translation
        
        return region.translation