from asyncio import Event, Lock
from typing import List

from PIL import Image
from pydantic import BaseModel

from manga_translator import Config
from server.sent_data_internal import fetch_data_stream, NotifyType, fetch_data

class ExecutorInstance(BaseModel):
    ip: str
    port: int
    busy: bool = False

    def free_executor(self):
        self.busy = False

    async def sent(self, image: Image, config: Config, image_name: str = None):
        return await fetch_data("http://"+self.ip+":"+str(self.port)+"/simple_execute/translate", image, config, image_name=image_name)

    async def sent_stream(self, image: Image, config: Config, sender: NotifyType, image_name: str = None):
        await fetch_data_stream("http://"+self.ip+":"+str(self.port)+"/execute/translate", image, config, sender, image_name=image_name)

    async def sent_batch(self, images: List[Image.Image], config: Config, batch_size: int, image_names: List[str] = None):
        """发送批量翻译请求"""
        # Ensure images_with_configs format matches MangaTranslator.translate_batch
        images_with_configs = []
        for i, img in enumerate(images):
            name = image_names[i] if image_names and i < len(image_names) else None
            images_with_configs.append({"image": img, "config": config, "image_name": name})
            
        attributes = {"images_with_configs_or_dicts": images_with_configs, "batch_size": batch_size}
        return await fetch_data("http://"+self.ip+":"+str(self.port)+"/simple_execute/translate_batch", None, config, custom_data=attributes)

    async def sent_batch_stream(self, images: List[Image.Image], config: Config, batch_size: int, sender: NotifyType, image_names: List[str] = None):
        """发送批量翻译流式请求"""
        # Ensure images_with_configs format matches MangaTranslator.translate_batch
        images_with_configs = []
        for i, img in enumerate(images):
            name = image_names[i] if image_names and i < len(image_names) else None
            images_with_configs.append({"image": img, "config": config, "image_name": name})

        attributes = {"images_with_configs_or_dicts": images_with_configs, "batch_size": batch_size}
        await fetch_data_stream("http://"+self.ip+":"+str(self.port)+"/execute/translate_batch", None, config, sender, custom_data=attributes)

class Executors:
    def __init__(self):
        self.list: List[ExecutorInstance] = []
        self.lock: Lock = Lock()
        self.event = Event()

    def register(self, instance: ExecutorInstance):
        self.list.append(instance)

    def free_executors(self) -> int:
        return len([item for item in self.list if not item.busy])

    async def _find_instance(self):
        while True:
            instance = next((x for x in self.list if x.busy == False), None)
            if instance is not None:
                return instance
            #todo: cricial error: warn should never happen
            await self.event.wait()

    async def find_executor(self) -> ExecutorInstance:
        async with self.lock:  # Using async with for lock management
            instance = await self._find_instance()
            instance.busy = True
            return instance

    async def free_executor(self, instance: ExecutorInstance):
        from server.myqueue import task_queue
        instance.free_executor()
        self.event.set()
        self.event.clear()
        await task_queue.update_event()

executor_instances: Executors = Executors()
