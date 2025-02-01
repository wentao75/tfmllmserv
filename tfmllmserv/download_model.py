#!/usr/bin/env python
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from tqdm import tqdm
import logging
from janus.models import MultiModalityCausalLM, VLChatProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_model(model_id: str = "OpenGVLab/InternVL2-1B", force: bool = False):
    """
    下载并缓存模型文件
    
    Args:
        model_id: HuggingFace模型ID
        force: 是否强制重新下载
    """
    logging.info(f"开始下载模型: {model_id}")
    
    try:
        if "Janus" in model_id:
            # 对于 Janus 模型使用特殊配置
            logging.info("检测到 Janus 模型，使用专用加载器...")
            
            # 下载处理器
            logging.info("下载 VLChatProcessor...")
            processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)
            
            # 下载模型
            logging.info("下载模型文件...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                force_download=force
            )
            
            # 下载分词器（通过处理器）
            tokenizer = processor.tokenizer
            
        else:
            # 下载模型
            logging.info("下载模型文件...")
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                force_download=force
            )
            
            # 下载分词器
            logging.info("下载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=False,
                force_download=force
            )
        
        # 获取缓存目录
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        logging.info(f"模型文件已缓存在: {cache_dir}")
        
        # 计算缓存大小
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        logging.info(f"缓存目录大小: {total_size / 1024 / 1024:.2f} MB")
        logging.info("模型下载完成!")
        
    except Exception as e:
        logging.error(f"下载失败: {str(e)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='下载并缓存HuggingFace模型')
    parser.add_argument('--model', type=str, default="OpenGVLab/InternVL2-1B",
                      help='要下载的模型ID (默认: OpenGVLab/InternVL2-1B)')
    parser.add_argument('--force', action='store_true',
                      help='强制重新下载模型文件')
    
    args = parser.parse_args()
    
    try:
        # 下载模型
        success = download_model(args.model, args.force)
        
        if success:
            logging.info("\n模型已准备就绪，可以启动服务了！")
        else:
            logging.error("\n模型下载失败，请检查网络连接或重试。")
    except KeyboardInterrupt:
        logging.info("\n下载已取消")
    except Exception as e:
        logging.error(f"\n发生未知错误: {str(e)}")

if __name__ == "__main__":
    main() 