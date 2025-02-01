#!/usr/bin/env python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import io
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ChatRequest(BaseModel):
    question: str

app = FastAPI()

# 全局变量存储模型和处理器
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def startup_event():
    global model, processor
    model_path = "deepseek-ai/Janus-Pro-1B"
    
    logging.info("正在加载 Janus 模型...")
    try:
        # 加载处理器
        processor = VLChatProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = model.to(device).eval()
        
        logging.info("模型加载完成！")
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        raise e

@app.post("/chat")
async def chat(image: UploadFile = File(...), request: ChatRequest = None):
    """
    处理图像和文本的多模态对话
    
    Args:
        image: 上传的图像文件
        request: 包含问题的请求体
    
    Returns:
        dict: 包含模型回答的响应
    """
    try:
        # 读取并转换图像
        image_content = await image.read()
        pil_image = Image.open(io.BytesIO(image_content))
        
        # 准备对话格式
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{request.question}",
                "images": [pil_image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # 处理输入
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(device)
        
        # 生成图像嵌入
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        
        # 生成回答
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        
        # 解码回答
        answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return {
            "answer": answer,
            "format": prepare_inputs['sft_format'][0]
        }
        
    except Exception as e:
        logging.error(f"处理请求失败: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 