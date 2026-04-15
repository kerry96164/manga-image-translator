import asyncio
import os
from typing import List, Optional

from PIL import Image
from fastapi import HTTPException
from fastapi.requests import Request

from manga_translator import Config
from server.instance import executor_instances
from server.sent_data_internal import NotifyType

class QueueElement:
    req: Request
    image: Image.Image | str
    config: Config
    image_name: Optional[str]

    def __init__(self, req: Request, image: Image.Image, config: Config, length, image_name: str = None):
        self.req = req
        if length > 10:
            #todo: store image in "upload-cache" folder
            self.image = image
        else:
            self.image = image
        self.config = config
        self.image_name = image_name

    def get_image(self)-> Image:
        if isinstance(self.image, str):
            return Image.open(self.image)
        else:
            return self.image

    def __del__(self):
        if isinstance(self.image, str):
            os.remove(self.image)

    async def is_client_disconnected(self) -> bool:
        if await self.req.is_disconnected():
            return True
        return False


class BatchQueueElement:
    """Batch translation queue element"""
    req: Request
    images: List[Image.Image]
    config: Config
    batch_size: int
    image_names: List[str]

    def __init__(self, req: Request, images: List[Image.Image], config: Config, batch_size: int, image_names: List[str] = None):
        self.req = req
        self.images = images
        self.config = config
        self.batch_size = batch_size
        self.image_names = image_names or []

    async def is_client_disconnected(self) -> bool:
        if await self.req.is_disconnected():
            return True
        return False


class TaskQueue:
    def __init__(self):
        self.queue: List[QueueElement | BatchQueueElement] = []
        self.queue_event: asyncio.Event = asyncio.Event()
        self.lock: asyncio.Lock = asyncio.Lock()

    def add_task(self, task: QueueElement | BatchQueueElement):
        self.queue.append(task)

    def get_pos(self, task: QueueElement | BatchQueueElement) -> Optional[int]:
        try:
            return self.queue.index(task)
        except ValueError:
            return None

    def get_pos_locked(self, task: QueueElement | BatchQueueElement) -> Optional[int]:
        try:
            return self.queue.index(task)
        except ValueError:
            return None

    async def update_event(self):
        async with self.lock:
            self.queue = [task for task in self.queue if not await task.is_client_disconnected()]
            self.queue_event.set()
            self.queue_event.clear()

    async def update_event_locked(self):
        self.queue = [task for task in self.queue if not await task.is_client_disconnected()]
        self.queue_event.set()
        self.queue_event.clear()

    async def remove(self, task: QueueElement | BatchQueueElement):
        async with self.lock:
            if task in self.queue:
                self.queue.remove(task)
            await self.update_event_locked()

    async def wait_for_event(self):
        await self.queue_event.wait()

task_queue = TaskQueue()

async def wait_in_queue(task: QueueElement | BatchQueueElement, notify: NotifyType):
    """Will get task position report it. If its in the range of translators then it will try to aquire an instance(blockig) and sent a task to it. when done the item will be removed from the queue and result will be returned"""
    while True:
        item_to_remove = None
        async with task_queue.lock:
            queue_pos = task_queue.get_pos_locked(task)
            if queue_pos is None:
                if notify:
                    return
                else:
                    raise HTTPException(500, detail="User is no longer connected")  # just for the logs
            if notify:
                notify(3, str(queue_pos).encode('utf-8'))
            
            if queue_pos < executor_instances.free_executors():
                if await task.is_client_disconnected():
                    await task_queue.update_event_locked()
                    if notify:
                        return
                    else:
                        raise HTTPException(500, detail="User is no longer connected") #just for the logs

                item_to_remove = task

        if item_to_remove:
            instance = await executor_instances.find_executor()
            await task_queue.remove(item_to_remove)
            if notify:
                notify(4, b"")

            try:
                # Process batch translation task
                if isinstance(task, BatchQueueElement):
                    if notify:
                        await instance.sent_batch_stream(task.images, task.config, task.batch_size, notify, image_names=task.image_names)
                    else:
                        result = await instance.sent_batch(task.images, task.config, task.batch_size, image_names=task.image_names)
                else:
                    # Process single translation task
                    if notify:
                        # 啟動任務後不等待，先釋放實例鎖讓其他任務可以開始辨識（如果有多個 instance）
                        # 或是讓 instance 在內部管理並發
                        await instance.sent_stream(task.image, task.config, notify, image_name=task.image_name)
                    else:
                        result = await instance.sent(task.image, task.config, image_name=task.image_name)

                await executor_instances.free_executor(instance)

                if notify:
                    return
                else:
                    return result

            except Exception as e:
                # 确保实例被释放
                await executor_instances.free_executor(instance)

                # 如果是连接错误，发送友好的错误消息
                if "Cannot connect to host" in str(e) or "Connection refused" in str(e):
                    error_msg = "Translation service is starting up, please wait a moment and try again."
                else:
                    error_msg = f"Translation failed: {str(e)}"

                if notify:
                    notify(2, error_msg.encode('utf-8'))
                    return
                else:
                    raise HTTPException(500, detail=error_msg)
        else:
            await task_queue.wait_for_event()