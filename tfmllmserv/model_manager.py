from typing import Dict, Optional, List, Tuple
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from janus.models import MultiModalityCausalLM, VLChatProcessor
import json
import os
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_config_path: str = "models_config.json", max_workers: int = 4):
        self.models_config_path = models_config_path
        self.models: Dict[str, dict] = {}
        self.loaded_models: Dict[str, dict] = {}
        self.model_locks: Dict[str, Lock] = {}  # 每个模型一个锁
        self.config_lock = Lock()  # 配置文件的锁
        self.executor = ThreadPoolExecutor(max_workers=max_workers)  # 线程池
        self._load_config()
        
    def _load_config(self):
        """加载模型配置"""
        with self.config_lock:
            if os.path.exists(self.models_config_path):
                with open(self.models_config_path, 'r') as f:
                    self.models = json.load(f)
                    
    def _save_config(self):
        """保存模型配置"""
        with self.config_lock:
            with open(self.models_config_path, 'w') as f:
                json.dump(self.models, f, indent=2)
            
    def add_model(self, model_id: str, display_name: Optional[str] = None):
        """添加新模型"""
        with self.config_lock:
            if model_id in self.models:
                raise ValueError(f"模型 {model_id} 已存在")
                
            self.models[model_id] = {
                "model_id": model_id,
                "display_name": display_name or model_id
            }
            self.model_locks[model_id] = Lock()  # 为新模型创建锁
            self._save_config()
        
    def remove_model(self, model_id: str):
        """删除模型"""
        with self.config_lock:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
                
            if model_id in self.loaded_models:
                self.unload_model(model_id)
                
            del self.models[model_id]
            if model_id in self.model_locks:
                del self.model_locks[model_id]
            self._save_config()
        
    def rename_model(self, model_id: str, new_name: str):
        """重命名模型"""
        with self.config_lock:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
                
            self.models[model_id]["display_name"] = new_name
            self._save_config()
        
    def load_model(self, model_id: str):
        """加载模型到内存"""
        if model_id not in self.models:
            raise ValueError(f"模型 {model_id} 不存在")
            
        # 获取或创建模型锁
        if model_id not in self.model_locks:
            with self.config_lock:
                self.model_locks[model_id] = Lock()
                
        # 使用模型特定的锁
        with self.model_locks[model_id]:
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]
                
            logger.info(f"开始加载模型: {model_id}")
            
            # 检测设备
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            
            try:
                # 根据模型类型选择不同的加载方式
                if "Janus" in model_id:
                    logger.info("检测到 Janus 模型，使用专用加载器...")
                    # 加载处理器
                    processor = VLChatProcessor.from_pretrained(
                        self.models[model_id]["model_id"],
                        trust_remote_code=True
                    )
                    
                    # 加载模型
                    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                        self.models[model_id]["model_id"],
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    model = model.to(torch.bfloat16).to(device).eval()
                    model.processor = processor
                    model.tokenizer = processor.tokenizer
                    
                elif "InternVL" in model_id:
                    logger.info("检测到 InternVL 模型，加载处理器")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.models[model_id]["model_id"],
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=device,
                        local_files_only=True
                    )
                    
                    processor = AutoProcessor.from_pretrained(
                        self.models[model_id]["model_id"],
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    model.processor = processor
                    
                    # 设置图像上下文token
                    model.img_context_token_id = 32001  # InternVL2的默认值
                    logger.info(f"设置图像上下文token: {model.img_context_token_id}")
                    
                else:
                    # 普通模型加载
                    model = AutoModelForCausalLM.from_pretrained(
                        self.models[model_id]["model_id"],
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=device,
                        local_files_only=True
                    )
                
                self.loaded_models[model_id] = model
                logger.info(f"模型加载完成: {model_id}")
                return model
                
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                raise
        
    def get_model(self, model_id: str):
        """获取已加载的模型"""
        if model_id in self.model_locks:
            with self.model_locks[model_id]:
                return self.loaded_models.get(model_id)
        return None
        
    def unload_model(self, model_id: str):
        """从内存中卸载模型"""
        if model_id in self.model_locks:
            with self.model_locks[model_id]:
                if model_id in self.loaded_models:
                    del self.loaded_models[model_id]
                    torch.cuda.empty_cache()
            
    def get_model_info(self, model_id: str) -> dict:
        """获取模型信息"""
        with self.config_lock:
            if model_id not in self.models:
                raise ValueError(f"模型 {model_id} 不存在")
            return self.models[model_id]
        
    def list_models(self) -> List[dict]:
        """列出所有模型"""
        with self.config_lock:
            return [
                {
                    "model_id": model_id,
                    "display_name": info["display_name"],
                    "loaded": model_id in self.loaded_models
                }
                for model_id, info in self.models.items()
            ] 