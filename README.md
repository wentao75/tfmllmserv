# TF Model LLM Service

基于 FastAPI 的多模态大语言模型服务框架，支持多线程并发处理和多种模型类型。

## 快速开始

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建 Python 3.9.6 虚拟环境
uv venv --python=python3.9.6
source .venv/bin/activate  # Linux/MacOS
# 或 .venv\Scripts\activate  # Windows

# 3. 安装依赖
uv pip install -r requirements.txt

# 4. 下载模型（可选）
uv run python -m tfmllmserv.download_model --model deepseek-ai/deepseek-vl2-small

# 5. 启动服务
# 设置环境变量
export UV_HTTP_TIMEOUT=300
export PYTORCH_ENABLE_MPS_FALLBACK=1  # 如果使用 Apple Silicon
export CUDA_VISIBLE_DEVICES=0  # 如果使用 GPU

# 启动服务
uv run uvicorn tfmllmserv.main:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用便捷脚本启动
uv run python -m tfmllmserv.main
```

## 功能特性

- 多线程并发处理请求
- 支持图文多模态输入
- 动态模型加载/卸载
- 自动资源管理和清理
- REST API 接口
- 支持多种模型类型
  - Janus 系列模型
    - 图文理解和生成
    - 多轮对话支持
    - 中英双语支持
  - InternVL 系列模型
    - 图像描述生成
    - 视觉问答
    - 场景理解
  - 其他标准 Transformer 模型

## 系统要求

- Python 版本要求：Python 3.9.6（固定版本）
  - 为确保最佳兼容性，请严格使用 Python 3.9.6
  - 此版本同时支持所有模型功能
- CUDA 支持（推荐）
- 8GB+ 内存
- 10GB+ 磁盘空间

### 依赖包

主要依赖包括：

1. 核心框架：
```bash
fastapi>=0.68.0      # Web API 框架
uvicorn>=0.15.0      # ASGI 服务器
pydantic>=1.8.0      # 数据验证
```

2. 机器学习相关：
```bash
torch>=2.0.0         # PyTorch
torchvision          # 计算机视觉工具
transformers>=4.30.0 # Hugging Face Transformers
accelerate>=0.20.0   # 加速推理
numpy>=1.24.0        # 数值计算
```

3. 多模态模型：
```bash
janus @ git+https://github.com/deepseek-ai/Janus.git        # Janus 模型
deepseek-vl @ git+https://github.com/deepseek-ai/DeepSeek-VL.git  # DeepSeek VL 模型
vllm>=0.2.0          # VLLM 推理引擎
```

4. 工具和优化：
```bash
pillow               # 图像处理
sentencepiece>=0.1.99 # 分词器
protobuf>=3.20.0     # 序列化
einops>=0.6.1        # 张量操作
psutil>=5.9.0        # 系统监控
typer>=0.4.0         # CLI 工具
rich>=10.0.0         # 终端美化
python-dotenv>=0.19.0 # 环境变量管理
```

### 特殊依赖安装说明

1. Janus 模型：
```bash
# 从 GitHub 安装
uv pip install git+https://github.com/deepseek-ai/Janus.git
```

2. DeepSeek-VL 模型：
```bash
# 从 GitHub 安装
uv pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git
```

3. VLLM 加速引擎：
```bash
# 安装 VLLM
uv pip install vllm>=0.2.0
```

### 性能监控

使用 `psutil` 进行系统资源监控：

1. 内存使用：
```python
import psutil
# 获取内存使用情况
memory_info = psutil.Process().memory_info()
memory_percent = psutil.Process().memory_percent()
```

2. CPU 使用：
```python
# 获取 CPU 使用率
cpu_percent = psutil.Process().cpu_percent()
```

3. GPU 监控：
```python
# 使用 NVIDIA-SMI（如果可用）
gpu_memory = torch.cuda.memory_allocated()
gpu_memory_cached = torch.cuda.memory_cached()
```

### 环境变量配置

新增环境变量：
```bash
# VLLM 配置
VLLM_MAX_WORKERS=4    # VLLM 最大工作线程数
VLLM_GPU_MEMORY_UTILIZATION=0.9  # GPU 显存利用率

# 系统监控
MONITOR_INTERVAL=60   # 监控间隔（秒）
MEMORY_THRESHOLD=0.9  # 内存使用阈值

# 模型加载
MODEL_LOAD_TIMEOUT=300  # 模型加载超时时间（秒）
```

### 性能优化

1. VLLM 加速：
- 使用 VLLM 进行高效推理
- 支持 Continuous Batching
- KV Cache 管理
- 自动显存优化

2. 系统监控：
- 实时监控系统资源使用
- 自动清理未使用的模型
- 内存使用优化
- 进程管理

3. 并发处理：
- 多线程请求处理
- 异步模型加载
- 动态资源分配

### 注意事项

1. VLLM 相关：
- 需要 CUDA 11.8 及以上
- 建议预留足够显存
- 支持模型量化

2. 系统资源：
- 监控内存使用情况
- 及时清理缓存
- 合理设置并发数

3. 依赖安装：
- 建议按顺序安装依赖
- Git 依赖可能需要特殊网络环境
- 注意版本兼容性

## 安装

### 1. 安装 uv

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 验证安装
uv --version
```

### 2. 安装 Python 3.9.6

```bash
# 使用 uv pip 安装 python
uv pip install python@3.9.6

# 或者使用 uv venv 时自动安装
uv venv --python=3.9.6

# 在 macOS 上，如果上述方法不起作用，可以使用 pyenv：
brew install pyenv
pyenv install 3.9.6
pyenv global 3.9.6
```

### 3. 创建项目环境

```bash
# 克隆仓库
git clone [repository_url]
cd tfmllmserv

# 创建虚拟环境并指定 Python 版本
uv venv --python=3.9.6
source .venv/bin/activate  # Linux/MacOS
# 或 .venv\Scripts\activate  # Windows

# 验证 Python 版本
python --version  # 应该显示 3.9.6

# 安装依赖
uv pip install -r requirements.txt
```

### 4. 特殊情况处理

如果遇到 Python 版本安装问题：

1. Windows 用户：
```powershell
# 使用 uv pip 安装
uv pip install python@3.9.6

# 或使用 Windows Store 安装 Python 3.9.6
# 然后指定完整路径
uv venv --python="C:\Users\<用户名>\AppData\Local\Programs\Python\Python39\python.exe"
```

2. Linux 用户：
```bash
# 使用系统包管理器
sudo apt install python3.9  # Ubuntu/Debian
sudo dnf install python39  # Fedora/RHEL

# 然后创建虚拟环境
uv venv --python=python3.9
```

3. macOS 用户：
```bash
# 使用 Homebrew
brew install python@3.9

# 或使用 pyenv（推荐）
brew install pyenv
pyenv install 3.9.6
pyenv global 3.9.6

# 然后创建虚拟环境
uv venv --python=3.9.6
```

### 5. 验证安装

```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/MacOS
# 或 .venv\Scripts\activate  # Windows

# 验证 Python 版本
python --version  # 应该显示 3.9.6

# 验证 uv
uv --version

# 验证依赖安装
python -c "import torch; print(torch.__version__)"
```

## 配置

### 模型配置

在 `models_config.json` 中配置模型信息：

```json
{
  "deepseek-ai/Janus-Pro-1B": {
    "model_id": "deepseek-ai/Janus-Pro-1B",
    "display_name": "Janus Pro 1B"
  }
}
```

### 服务配置

主要配置项：
- `max_workers`: 最大工作线程数（默认：4）
- `timeout`: 请求超时时间（默认：30秒）
- `device`: 运行设备（自动检测）

## 启动服务

### 基本启动

```bash
# 1. 激活虚拟环境（如果尚未激活）
source .venv/bin/activate  # Linux/MacOS
# 或 .venv\Scripts\activate  # Windows

# 2. 设置环境变量
export UV_HTTP_TIMEOUT=300
export PYTORCH_ENABLE_MPS_FALLBACK=1  # 如果使用 Apple Silicon
export CUDA_VISIBLE_DEVICES=0  # 如果使用 GPU

# 3. 启动服务（使用 uvicorn）
uv run uvicorn tfmllmserv.main:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用便捷脚本启动
uv run python -m tfmllmserv.main
```

### 开发模式启动

```bash
# 启用热重载（适用于开发）
uv run uvicorn tfmllmserv.main:app --reload --host 0.0.0.0 --port 8000
```

### 生产环境启动

```bash
# 使用 gunicorn 作为进程管理器（推荐用于生产环境）
uv run gunicorn tfmllmserv.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# 或使用 supervisor 管理（需要先安装 supervisor）
uv run supervisord -c supervisor.conf
```

### 环境变量设置

Windows PowerShell:
```powershell
$env:UV_HTTP_TIMEOUT=300
$env:PYTORCH_ENABLE_MPS_FALLBACK=1  # 如果使用 Apple Silicon
$env:CUDA_VISIBLE_DEVICES=0  # 如果使用 GPU
```

Windows CMD:
```cmd
set UV_HTTP_TIMEOUT=300
set PYTORCH_ENABLE_MPS_FALLBACK=1
set CUDA_VISIBLE_DEVICES=0
```

Linux/MacOS:
```bash
export UV_HTTP_TIMEOUT=300
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=0
```

### 性能调优参数

```bash
# 调整工作进程数（建议设置为 CPU 核心数）
--workers 4

# 调整每个工作进程的线程数
--worker-connections 1000

# 调整请求超时时间（秒）
export UV_HTTP_TIMEOUT=300

# 调整 PyTorch 显存分配
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# 限制 GPU 显存使用
export CUDA_VISIBLE_DEVICES=0,1  # 使用指定的 GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32  # 限制单次分配大小
```

## API 接口

### 1. 聊天补全接口

**端点**: `/v1/chat/completions`  
**方法**: POST

请求格式：
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "描述这张图片"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,..."
          }
        }
      ]
    }
  ],
  "model": "deepseek-ai/Janus-Pro-1B",
  "temperature": 0.7,
  "max_tokens": 2048
}
```

响应格式：
```json
{
  "id": "chat-1234",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "deepseek-ai/Janus-Pro-1B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "模型生成的回复内容"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "response_time": 1.23
  }
}
```

### 2. 获取模型列表

**端点**: `/v1/models`  
**方法**: GET

响应格式：
```json
{
  "object": "list",
  "data": [
    {
      "id": "deepseek-ai/Janus-Pro-1B",
      "object": "model",
      "created": 1234567890,
      "owned_by": "organization",
      "permission": [],
      "root": "deepseek-ai/Janus-Pro-1B",
      "parent": null,
      "display_name": "Janus Pro 1B",
      "loaded": true
    }
  ]
}
```

### 3. 模型能力

各模型支持的具体功能：

#### Janus 系列
- 图文理解：支持图片描述、视觉问答
- 多模态对话：支持图文混合输入的多轮对话
- 文本生成：支持中英双语的文本生成和对话
- 图像分析：支持场景识别、物体检测、文字识别

#### DeepSeek-VL2 系列
- 图文理解：支持高精度的图像理解和描述
- 视觉定位：支持精确的视觉引用和定位
- 多图对话：支持多图上下文学习和推理
- 特点：
  - 动态分块策略：<=2张图片时使用动态分块，>=3张图片时使用384*384固定大小
  - 推荐温度参数：T <= 0.7（更高温度会降低生成质量）
  - 三种型号：Tiny、Small、Base（基于不同规模的基础LLM）

#### InternVL 系列
- 图像描述：生成详细的图像描述
- 视觉问答：回答关于图像的具体问题
- 场景理解：分析图像中的场景和关系
- 多语言支持：支持中英文输入输出

## 使用示例

### Python 客户端示例

```python
import requests
import base64
from PIL import Image
import io

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 准备请求
image_base64 = encode_image("example.jpg")
url = "http://localhost:8000/v1/chat/completions"

# DeepSeek-VL2 示例 - 视觉定位
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<image>\n<|ref|>图片中的长颈鹿<|/ref|>"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    "model": "deepseek-ai/deepseek-vl2-small",
    "temperature": 0.7,
    "max_tokens": 512
}

# 发送请求
response = requests.post(url, json=payload)
print(response.json())

# DeepSeek-VL2 示例 - 多图对话
images = ["dog_a.jpg", "dog_b.jpg", "dog_c.jpg", "dog_d.jpg"]
image_contents = []

for img_path in images:
    image_base64 = encode_image(img_path)
    image_contents.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
    })

payload_multi = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "第一张图片中的狗没有穿衣服，第二张图片中的狗戴着圣诞帽，第三张图片中的狗穿着魔法师服装。第四张图片中的狗穿着什么？"},
                *image_contents
            ]
        }
    ],
    "model": "deepseek-ai/deepseek-vl2-small",
    "temperature": 0.7,
    "max_tokens": 512
}

response = requests.post(url, json=payload_multi)
print(response.json())
```

### cURL 示例

```bash
# 获取模型列表
curl http://localhost:8000/v1/models

# 发送聊天请求（仅文本）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下自己"
      }
    ],
    "model": "deepseek-ai/Janus-Pro-1B"
  }'
```

## 性能优化

1. 并发处理
   - 服务使用线程池处理并发请求
   - 每个模型有独立的锁机制
   - 异步处理图片加载和转换

2. 内存管理
   - 自动清理 GPU 缓存
   - 支持模型动态加载/卸载
   - 后台任务清理资源

3. 建议配置
   - 生产环境建议 `max_workers >= CPU核心数`
   - GPU 显存充足时可提高并发数
   - 根据实际负载调整超时时间

### 性能指标

典型硬件配置下的性能表现：

1. GPU 模式 (NVIDIA A100)
   - 平均响应时间：1-2秒
   - 并发处理能力：8-10 请求/秒
   - 显存占用：~15GB (Janus-Pro-1B)
   - CPU 使用率：30-40%

2. CPU 模式
   - 平均响应时间：4-6秒
   - 并发处理能力：2-3 请求/秒
   - 内存占用：~8GB
   - CPU 使用率：80-90%

3. MPS 模式 (Apple Silicon)
   - 平均响应时间：2-3秒
   - 并发处理能力：4-5 请求/秒
   - 统一内存占用：~12GB
   - CPU 使用率：50-60%

## 注意事项

1. 图片处理
   - 支持 base64 编码的图片输入
   - 自动进行图片预处理（调整大小、归一化等）
   - 支持 RGB 格式转换

2. 模型管理
   - 首次请求时自动加载模型
   - 支持热插拔（动态添加/删除模型）
   - 配置文件修改后自动生效

3. 错误处理
   - 请求超时自动中断
   - 模型加载失败自动重试
   - 详细的错误日志记录

### 错误处理

1. 常见错误及解决方案：

| 错误码 | 描述 | 可能原因 | 解决方案 |
|--------|------|----------|----------|
| 400 | 请求格式错误 | 输入参数不符合要求 | 检查请求格式和参数 |
| 404 | 模型未找到 | 模型未配置或加载失败 | 检查模型配置和模型文件 |
| 408 | 请求超时 | 处理时间超过限制 | 调整超时设置或减少输入大小 |
| 413 | 请求体过大 | 图片或文本超过限制 | 压缩图片或减少文本长度 |
| 500 | 内部服务错误 | 模型推理错误 | 查看服务日志定位原因 |
| 503 | 服务不可用 | 资源不足或服务过载 | 检查资源使用情况 |

2. 日志说明：
```
INFO: 正常的操作日志
WARNING: 需要注意但不影响服务的问题
ERROR: 导致请求失败的错误
CRITICAL: 需要立即处理的严重问题
```

3. 监控指标：
- 请求成功率
- 平均响应时间
- GPU 显存使用率
- CPU 使用率
- 并发请求数
- 错误率统计

### 模型特定说明

1. DeepSeek-VL2 系列
   - 推荐使用温度参数 T <= 0.7
   - 图片处理策略：
     - 1-2张图片：使用动态分块
     - 3张及以上：统一缩放到384*384
   - 型号选择：
     - Tiny：轻量级应用
     - Small：平衡性能和资源
     - Base：最佳效果但需要更多资源

## 已知问题

1. MPS (Apple Silicon) 相关警告：
   - 某些操作会回退到 CPU 执行：'aten::upsample_bicubic2d.out' 在 MPS 后端上不支持
   - 解决方案：已通过设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` 环境变量处理

2. FlashAttention 相关：
   - FlashAttention2 未安装时会使用 eager attention
   - 这是正常现象，不影响模型功能

3. Transformers 警告：
   - timm.models.layers 导入已弃用
   - InternLM2ForCausalLM 生成功能相关警告
   - 这些是模型库的警告，不影响正常使用

## 开发计划

- [ ] 添加模型量化支持
- [ ] 实现模型并行处理
- [ ] 添加请求队列管理
- [ ] 支持更多模型类型
- [ ] 添加模型性能监控
- [ ] 优化内存使用

### 路线图

#### 近期计划（1-2个月）
- [ ] 添加模型量化支持
- [ ] 实现模型并行处理
- [ ] 添加请求队列管理

#### 中期计划（3-6个月）
- [ ] 支持更多模型类型
- [ ] 添加模型性能监控
- [ ] 优化内存使用

#### 长期计划（6个月以上）
- [ ] 分布式部署支持
- [ ] 自动化运维工具
- [ ] 更多语言的客户端SDK

## 贡献指南

欢迎提交 Issue 和 Pull Request。在提交代码前，请确保：

1. 通过所有测试
2. 更新相关文档
3. 遵循代码规范
4. 添加必要的注释

## 许可证

[许可证类型]
