#!/usr/bin/env python
import os
import sys
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from janus.models import MultiModalityCausalLM, VLChatProcessor
import torch
import importlib.util

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    if version < (3, 9) or version >= (3, 10):
        logger.error("当前 Python 版本不兼容：%s", sys.version)
        logger.error("本项目需要 Python 3.9.6 版本")
        logger.error("请使用正确版本的 Python")
        logger.error("\n创建新的虚拟环境：")
        logger.error("uv venv --python=python3.9.6")
        logger.error("source .venv/bin/activate  # Linux/MacOS")
        logger.error("# 或 .venv\\Scripts\\activate  # Windows")
        return False
    return True

def check_deepseek_vl():
    """检查是否安装了 deepseek-vl 包"""
    if importlib.util.find_spec("deepseek_vl") is None:
        logger.error("未找到 deepseek-vl 包。对于 DeepSeek-VL2 模型，请按以下步骤安装：")
        logger.error("\n1. 克隆仓库并安装：")
        logger.error("git clone https://github.com/deepseek-ai/DeepSeek-VL.git")
        logger.error("cd DeepSeek-VL")
        logger.error("uv pip install -e .")
        logger.error("\n或者直接从 GitHub 安装：")
        logger.error("uv pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git")
        logger.error("\n2. 安装可选依赖（推荐）：")
        logger.error("uv pip install flash-attn --no-deps")
        logger.error("uv pip install opencv-python-headless")
        return False
    return True

def download_model(model_id: str, path: str = None, device: str = None):
    """
    下载模型文件
    
    Args:
        model_id: 模型ID
        path: 保存路径（可选）
        device: 设备类型（可选）
    """
    try:
        logger.info(f"开始下载模型: {model_id}")
        
        # 检查是否是 DeepSeek-VL 模型
        if "deepseek-vl" in model_id.lower():
            if not check_python_version() or not check_deepseek_vl():
                return False
            # 动态导入 DeepSeek-VL 相关模块
            from deepseek_vl.models import VLMImageProcessor, MultiModalityCausalLM
        
        # 设置下载参数
        kwargs = {
            "trust_remote_code": True,
            "local_files_only": False
        }
        
        if path:
            kwargs["cache_dir"] = path
            
        if device:
            kwargs["device_map"] = device
            
        logger.info("下载模型文件...")
        
        # 根据模型类型选择不同的下载方式
        if "Janus" in model_id:
            # 下载处理器
            processor = VLChatProcessor.from_pretrained(model_id, **kwargs)
            # 下载模型
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            logger.info("下载 Janus 模型完成")
            
        elif "deepseek-vl" in model_id.lower():
            # 下载处理器
            processor = AutoProcessor.from_pretrained(model_id, **kwargs)
            # 下载模型
            model = MultiModalityCausalLM.from_pretrained(model_id, **kwargs)
            logger.info("下载 DeepSeek-VL 模型完成")
            
        elif "InternVL" in model_id:
            # 下载处理器
            processor = AutoProcessor.from_pretrained(model_id, **kwargs)
            # 下载模型
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            logger.info("下载 InternVL 模型完成")
            
        else:
            # 通用模型下载
            if "Qwen" in model_id:
                try:
                    logger.info("下载 Qwen tokenizer...")
                    # 下载 tokenizer，移除 trust_remote_code 参数
                    tokenizer_kwargs = kwargs.copy()
                    tokenizer_kwargs["use_fast"] = False  # 使用慢速 tokenizer 以提高兼容性
                    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
                    logger.info("下载 Qwen tokenizer 完成")
                except Exception as e:
                    logger.error(f"下载 Qwen tokenizer 失败: {str(e)}")
                    raise
            # 下载模型
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            logger.info("下载通用模型完成")
            
        logger.info(f"模型 {model_id} 下载完成")
        return True
        
    except ImportError as e:
        logger.error(f"导入错误: {str(e)}")
        logger.error("请检查是否已安装所需的依赖包")
        return False
    except Exception as e:
        logger.error(f"下载失败: {str(e)}")
        logger.error("\n模型下载失败，请检查网络连接或重试。")
        return False

def main():
    parser = argparse.ArgumentParser(description="模型下载工具")
    parser.add_argument("--model", type=str, required=True, help="模型ID")
    parser.add_argument("--path", type=str, help="保存路径（可选）")
    parser.add_argument("--device", type=str, help="设备类型（可选）")
    
    args = parser.parse_args()
    success = download_model(args.model, args.path, args.device)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main() 