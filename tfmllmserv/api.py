from fastapi import FastAPI, HTTPException
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

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ImageNet归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
model_manager = ModelManager()

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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    start_time = time.time()
    try:
        # 处理消息
        messages = request.messages
        last_message = messages[-1]
        model_id = request.model  # 从请求中获取模型 ID
        
        # 构建对话历史，同时保存历史图片
        conversation_history = []
        history_images = []
        
        for msg in messages[:-1]:  # 除了最后一条消息
            if isinstance(msg.content, str):
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
            elif isinstance(msg.content, list):
                # 处理多模态消息
                content = ""
                for item in msg.content:
                    if item.type == "text":
                        content += item.text + "\n"
                    elif item.type == "image_url" and item.image_url:
                        image_url = item.image_url.get("url", "")
                        if image_url.startswith("data:image"):
                            try:
                                # 解析base64图片
                                image_data = image_url.split(",")[1]
                                image_bytes = base64.b64decode(image_data)
                                image = Image.open(io.BytesIO(image_bytes))
                                
                                # 使用transform处理图片
                                transform = build_transform(input_size=448)
                                image_tensor = transform(image).unsqueeze(0)
                                history_images.append(image_tensor)
                                logger.info("成功处理历史消息中的图片")
                            except Exception as e:
                                logger.error(f"处理历史图片时出错: {str(e)}")
                
                conversation_history.append({
                    "role": msg.role,
                    "content": content.strip()
                })
        
        logger.info(f"对话历史消息数: {len(conversation_history)}")
        logger.info(f"历史图片数: {len(history_images)}")
        
        # 提取文本和图片
        text = ""
        images = []
        
        if isinstance(last_message.content, str):
            logger.info(
                f"收到文本请求: {last_message.content[:100]}..."
            )
            
            # 首先检查是否是系统消息
            system_markers = [
                "### Task:", 
                "### Instructions:", 
                "### Guidelines:", 
                "### Output:",
                "### Examples:",
                "### Chat History:",
                "### Context:",
                "### System:",
                "<system>",
                "You are an",
                "You're an",
                "As an AI",
                "Act as"
            ]
            
            is_system_message = False
            detected_marker = None
            for marker in system_markers:
                if marker in last_message.content:
                    is_system_message = True
                    detected_marker = marker
                    break
            
            if is_system_message:
                logger.info(
                    f"检测到系统消息，标记为: {detected_marker}"
                )
                response = ChatResponse(
                    model="OpenGVLab/InternVL2-2B",
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '{ "text": "" }'
                        },
                        "finish_reason": "stop"
                    }],
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "response_time": time.time() - start_time
                    }
                )
                logger.info(
                    f"成功返回系统消息响应，处理时间: {time.time() - start_time:.2f}秒"
                )
                return response
            
            # 检查是否包含特殊的搜索查询格式
            has_search_format = (
                "<type>" in last_message.content and 
                "<text>" in last_message.content
            )
            if has_search_format:
                logger.info("检测到搜索查询格式")
                # 提取实际的查询文本
                text_match = re.search(
                    r'<text>(.*?)</text>', 
                    last_message.content
                )
                if text_match:
                    text = text_match.group(1).strip()
                    logger.info(f"从搜索查询中提取文本: {text}")
                
                # 检查是否是示例文本
                example_texts = [
                    "The sun was setting over the horizon, painting the sky",
                    "Top-rated restaurants in"
                ]
                
                if text in example_texts:
                    logger.info(f"检测到示例文本: {text}")
                    example_responses = {
                        "The sun was setting over the horizon, painting the sky": 
                        "with vibrant shades of orange and pink.",
                        "Top-rated restaurants in": 
                        "New York City for Italian cuisine."
                    }
                    response = ChatResponse(
                        model="OpenGVLab/InternVL2-2B",
                        choices=[{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": (
                                    f'{{ "text": "{example_responses[text]}" }}'
                                )
                            },
                            "finish_reason": "stop"
                        }],
                        usage={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "response_time": time.time() - start_time
                        }
                    )
                    logger.info(
                        f"成功返回示例文本响应，处理时间: {time.time() - start_time:.2f}秒"
                    )
                    return response
            else:
                text = last_message.content
                logger.info("使用原始文本内容进行处理")
                
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
        
        # 处理图片和其他内容
        if isinstance(last_message.content, list):
            logger.info("处理多模态内容")
            for item in last_message.content:
                if item.type == "text":
                    text = item.text
                    logger.info(f"提取文本内容: {text}")
                elif item.type == "image_url" and item.image_url:
                    image_url = item.image_url.get("url", "")
                    if image_url.startswith("data:image"):
                        try:
                            # 解析base64图片
                            image_data = image_url.split(",")[1]
                            image_bytes = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            
                            # 使用transform处理图片
                            transform = build_transform(input_size=448)
                            image_tensor = transform(image).unsqueeze(0)
                            images.append(image_tensor)
                            logger.info("成功处理图片")
                        except Exception as e:
                            logger.error(f"处理图片时出错: {str(e)}")
                            raise HTTPException(
                                status_code=400,
                                detail="图片处理失败"
                            )
        
        try:
            if images:
                logger.info("开始处理图文请求")
                # 确保图片在正确的设备上并转换为float16
                device = next(model.parameters()).device
                # 使用最后一张图片（可能是当前消息的图片或历史消息的最后一张图片）
                image_tensor = images[-1].to(device).to(torch.float16)
                
                # 根据模型类型选择不同的处理方式
                if "Janus" in model_id:
                    logger.info("使用 Janus 模型处理请求")
                    # 将 tensor 转换回 PIL 图像
                    transform = T.ToPILImage()
                    pil_image = transform(image_tensor[0].cpu())
                    
                    # 准备对话格式
                    conversation = [
                        {
                            "role": "<|User|>",
                            "content": f"<image_placeholder>\n{text}",
                            "images": [pil_image],
                        },
                        {"role": "<|Assistant|>", "content": ""},
                    ]
                    
                    # 准备输入
                    prepare_inputs = model.processor(
                        conversations=conversation,
                        images=[pil_image],  # 直接使用 PIL 图像
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
                logger.info("图文请求处理完成")
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
                    
                    # 如果是搜索查询格式，包装响应
                    if has_search_format:
                        response_text = f'{{ "text": "{response_text}" }}'
                logger.info("文本请求处理完成")
            
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
            logger.info(f"请求处理完成，响应时间: {response_time:.2f}秒")
                
            # 构建响应
            response = ChatResponse(
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
            
            return response
            
        except HTTPException as he:
            # 直接重新抛出HTTP异常
            raise he
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
            
    except Exception as e:
        # 计算错误响应时间
        end_time = time.time()
        response_time = end_time - start_time
        logger.error(f"处理请求时出错: {str(e)}, 响应时间: {response_time:.2f}秒", exc_info=True)
        
        # 如果是HTTP异常就直接重新抛出
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=str(e))

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