from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Literal
from .model_manager import ModelManager
import os
import base64
from PIL import Image
import io
import logging
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import time
import re
from janus.utils.io import load_pil_images
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ImageNet归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 创建线程池
executor = ThreadPoolExecutor(max_workers=4)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(
            (input_size, input_size), 
            interpolation=InterpolationMode.BICUBIC
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

app = FastAPI(title="TF Model LLM Service")
model_manager = ModelManager(max_workers=4)  # 设置模型管理器的最大工作线程数

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    """服务启动时的事件处理"""
    logger.info("正在启动服务...")
    # 可以在这里进行一些初始化操作

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时的事件处理"""
    logger.info("正在关闭服务...")
    executor.shutdown(wait=True)
    # 清理所有已加载的模型
    for model_id in list(model_manager.loaded_models.keys()):
        model_manager.unload_model(model_id)
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

async def cleanup_resources():
    """清理资源的异步函数"""
    try:
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"清理资源时出错: {str(e)}")

class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: Dict[str, str]

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class ChatResponse(BaseModel):
    id: str = "chat-1234"
    object: str = "chat.completion"
    created: int = 1234567890
    model: str
    choices: List[Dict] = []
    usage: Dict = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "response_time": 0.0
    }

@app.get("/")
async def root():
    """返回Web UI"""
    return FileResponse(os.path.join(static_dir, "index.html"))

async def process_request(request: ChatRequest) -> ChatResponse:
    """异步处理请求的函数"""
    start_time = time.time()
    try:
        # 处理消息
        messages = request.messages
        last_message = messages[-1]
        model_id = request.model
        
        # 构建对话历史，同时保存历史图片
        conversation_history = []
        history_images = []
        
        # 使用线程池处理图片加载和转换
        loop = asyncio.get_event_loop()
        
        async def process_message(msg):
            if isinstance(msg.content, str):
                return {
                    "role": msg.role,
                    "content": msg.content
                }
            elif isinstance(msg.content, list):
                content = ""
                local_images = []
                
                for item in msg.content:
                    if item.type == "text":
                        content += item.text + "\n"
                    elif item.type == "image_url" and item.image_url:
                        image_url = item.image_url.get("url", "")
                        if image_url.startswith("data:image"):
                            try:
                                # 在线程池中处理图片
                                image_data = image_url.split(",")[1]
                                image_bytes = base64.b64decode(image_data)
                                image = await loop.run_in_executor(
                                    executor,
                                    lambda: Image.open(io.BytesIO(image_bytes))
                                )
                                
                                # 使用transform处理图片
                                transform = build_transform(input_size=448)
                                image_tensor = await loop.run_in_executor(
                                    executor,
                                    lambda: transform(image).unsqueeze(0)
                                )
                                local_images.append(image_tensor)
                                logger.info("成功处理图片")
                            except Exception as e:
                                logger.error(f"处理图片时出错: {str(e)}")
                                raise HTTPException(
                                    status_code=400,
                                    detail="图片处理失败"
                                )
                
                return {
                    "role": msg.role,
                    "content": content.strip(),
                    "images": local_images
                }
        
        # 并发处理所有消息
        processed_messages = await asyncio.gather(
            *[process_message(msg) for msg in messages[:-1]]
        )
        
        for msg in processed_messages:
            conversation_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
            if "images" in msg:
                history_images.extend(msg["images"])
        
        logger.info(f"对话历史消息数: {len(conversation_history)}")
        logger.info(f"历史图片数: {len(history_images)}")
        
        # 提取文本和图片
        text = ""
        images = []
        
        # 处理最后一条消息
        last_msg_processed = await process_message(last_message)
        if isinstance(last_msg_processed.get("content"), str):
            text = last_msg_processed["content"]
        if "images" in last_msg_processed:
            images.extend(last_msg_processed["images"])
        
        # 设置处理超时
        timeout = 30  # 30秒超时
        current_time = time.time()
        
        # 如果不是特殊格式，加载模型处理请求
        logger.info("准备加载模型处理请求")
        model = model_manager.get_model(model_id)
        if model is None:
            model = model_manager.load_model(model_id)
            
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
            
        logger.info("模型加载成功")
        
        # 在线程池中执行模型推理
        response_text = await loop.run_in_executor(
            executor,
            lambda: _model_inference(
                model, model_id, text, images,
                conversation_history, request
            )
        )
        
        # 检查是否超时
        if time.time() - current_time > timeout:
            logger.warning("请求处理超时")
            raise HTTPException(
                status_code=408,
                detail="请求处理超时"
            )
        
        # 计算响应时间
        end_time = time.time()
        response_time = end_time - start_time
        
        return ChatResponse(
            model=model_id,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "response_time": response_time
            }
        )
        
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def _model_inference(model, model_id, text, images, conversation_history, request):
    """在线程池中执行的模型推理函数"""
    try:
        response_text = ""  # 初始化响应文本变量
        
        if images:
            logger.info("开始处理图文请求")
            device = next(model.parameters()).device
            image_tensor = images[-1].to(device).to(torch.float16)
            
            if "Janus" in model_id:
                logger.info("使用 Janus 模型处理请求")
                transform = T.ToPILImage()
                pil_image = transform(image_tensor[0].cpu())
                
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{text}",
                        "images": [pil_image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                prepare_inputs = model.processor(
                    conversations=conversation,
                    images=[pil_image],
                    force_batchify=True
                ).to(model.device)
                
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=model.tokenizer.eos_token_id,
                    bos_token_id=model.tokenizer.bos_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    max_new_tokens=min(request.max_tokens or 2048, 512),
                    do_sample=request.temperature > 0,
                    temperature=request.temperature or 0.7,
                    use_cache=True,
                )
                
                response_text = model.tokenizer.decode(
                    outputs[0].cpu().tolist(),
                    skip_special_tokens=True
                )
            else:
                logger.info("使用其他模型处理图文请求")
                # 构建输入
                if "<image>" not in text:
                    text = "<image>\n" + text
                
                # 添加对话历史到文本中
                if conversation_history:
                    history_text = "\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in conversation_history
                    ])
                    text = f"{history_text}\n{text}"
                    logger.info("已添加对话历史到请求中")
                
                # 生成配置
                generation_config = {
                    "max_new_tokens": min(request.max_tokens or 2048, 512),
                    "do_sample": True,
                    "temperature": request.temperature or 0.7,
                    "top_p": 0.9
                }
                
                # 生成回复
                response_text = model.chat(
                    model.processor,
                    pixel_values=image_tensor,
                    question=text,
                    generation_config=generation_config
                )
        else:
            logger.info("处理纯文本请求")
            # 纯文本处理
            if "Janus" in model_id:
                # 准备对话格式
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": text,
                        "images": [],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                # 准备输入
                prepare_inputs = model.processor(
                    conversations=conversation,
                    images=[],
                    force_batchify=True
                ).to(model.device)
                
                # 生成图像嵌入
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                
                # 生成回答
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=model.tokenizer.eos_token_id,
                    bos_token_id=model.tokenizer.bos_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    max_new_tokens=min(request.max_tokens or 2048, 512),
                    do_sample=request.temperature > 0,
                    temperature=request.temperature or 0.7,
                    use_cache=True,
                )
                
                # 解码回答
                response_text = model.tokenizer.decode(
                    outputs[0].cpu().tolist(),
                    skip_special_tokens=True
                )
            else:
                logger.info("使用其他模型处理纯文本请求")
                generation_config = {
                    "max_new_tokens": min(request.max_tokens or 2048, 512),
                    "do_sample": True,
                    "temperature": request.temperature or 0.7,
                    "top_p": 0.9
                }
                
                # 添加对话历史到文本中
                if conversation_history:
                    history_text = "\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in conversation_history
                    ])
                    text = f"{history_text}\n{text}"
                    logger.info("已添加对话历史到请求中")
                
                # 对于纯文本对话，使用generate方法
                text_inputs = model.processor(
                    text,
                    return_tensors="pt",
                    padding=True
                ).to(next(model.parameters()).device)
                
                outputs = model.generate(
                    **text_inputs,
                    **generation_config
                )
                
                response_text = model.processor.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
        
        if not response_text:
            logger.warning("生成的响应文本为空")
            response_text = "抱歉，我现在无法生成有效的回复。"
            
        return response_text
        
    except Exception as e:
        logger.error(f"模型推理时出错: {str(e)}")
        raise

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, background_tasks: BackgroundTasks):
    """处理聊天请求的异步端点"""
    try:
        response = await process_request(request)
        # 在后台清理资源
        background_tasks.add_task(cleanup_resources)
        return response
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}")
        raise

@app.get("/v1/models")
async def list_models():
    """列出可用的模型"""
    try:
        models = model_manager.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": model["model_id"],
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "organization",
                    "permission": [],
                    "root": model["model_id"],
                    "parent": None,
                    "display_name": model["display_name"],
                    "loaded": model["loaded"]
                }
                for model in models
            ]
        }
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 