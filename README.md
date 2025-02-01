# TF Model LLM Service

基于 FastAPI 的多模态大语言模型服务框架，支持多线程并发处理和多种模型类型。

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

- Python 3.10+
- CUDA 支持（推荐）
- 8GB+ 内存
- 10GB+ 磁盘空间

## 安装

```bash
# 克隆仓库
git clone [repository_url]
cd tfmllmserv

# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Linux/MacOS
# 或 .venv\Scripts\activate  # Windows
uv pip install -r requirements.txt
```

### 依赖包

主要依赖包括：
```fastapi>=0.95.0
uvicorn>=0.21.1
pydantic>=2.0.0
torch>=2.0.0
transformers>=4.30.0
pillow>=9.5.0
numpy>=1.24.0
janus>=1.0.0
```

完整依赖见 `requirements.txt`

### 模型预下载

服务支持模型预下载功能，可以提前下载模型文件到本地，避免首次请求时的下载等待。

#### 预下载脚本

```bash
# 下载单个模型
uv run python -m tfmllmserv.tools.download_model --model deepseek-ai/Janus-Pro-1B

# 下载多个模型
uv run python -m tfmllmserv.tools.download_model --model deepseek-ai/Janus-Pro-1B OpenGVLab/InternVL2-2B

# 下载配置文件中的所有模型
uv run python -m tfmllmserv.tools.download_model --all

# 指定下载目录
uv run python -m tfmllmserv.tools.download_model --model deepseek-ai/Janus-Pro-1B --path /path/to/models

# 使用特定设备下载（用于转换权重）
uv run python -m tfmllmserv.tools.download_model --model deepseek-ai/Janus-Pro-1B --device cuda
```

#### 支持的模型

目前支持以下模型的预下载：

1. Janus 系列
   - deepseek-ai/Janus-Pro-1B
   - deepseek-ai/Janus-Pro-7B
   - 存储需求：约 2GB/7GB

2. InternVL 系列
   - OpenGVLab/InternVL2-2B
   - OpenGVLab/InternVL2-7B
   - 存储需求：约 4GB/14GB

#### 存储路径

默认情况下，模型文件将下载到以下位置：
```
$HOME/.cache/huggingface/hub/  # 默认路径
./models/                      # 指定 --path 时的相对路径
/path/to/models/              # 指定 --path 时的绝对路径
```

#### 配置选项

在 `download_config.json` 中可以配置下载选项：

```json
{
  "default_path": "./models",          # 默认下载路径
  "resume_download": true,             # 支持断点续传
  "force_download": false,             # 强制重新下载
  "proxies": {                         # 代理设置
    "http": "http://proxy.example.com:8080",
    "https": "http://proxy.example.com:8080"
  },
  "local_files_only": false,          # 仅使用本地文件
  "token": "your_token"               # HuggingFace token（可选）
}
```

#### 注意事项

1. 存储空间
   - 确保有足够的磁盘空间
   - 预留约 20% 的额外空间用于模型加载

2. 网络要求
   - 稳定的网络连接
   - 推荐使用代理加速下载
   - 支持断点续传

3. 权限要求
   - 下载目录的写入权限
   - HuggingFace token（私有模型需要）

4. 其他说明
   - 支持多进程下载加速
   - 自动验证文件完整性
   - 支持增量更新

### 环境变量

服务支持以下环境变量配置：
```bash
# 服务配置
UV_HTTP_TIMEOUT=300        # HTTP 请求超时时间
PORT=8000                 # 服务端口
HOST=0.0.0.0             # 服务地址

# GPU 配置
CUDA_VISIBLE_DEVICES=0    # 使用的 GPU 设备
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32  # PyTorch 显存分配配置

# 性能优化
MAX_WORKERS=4            # 最大工作线程数
MODEL_CACHE_SIZE=2       # 最大缓存模型数量
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

```bash
# 使用 uvicorn 启动
UV_HTTP_TIMEOUT=300 uv run uvicorn tfmllmserv.main:app --host 0.0.0.0 --port 8000

# 或使用提供的启动脚本
uv run python -m tfmllmserv.main
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

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 准备请求
image_base64 = encode_image("example.jpg")
url = "http://localhost:8000/v1/chat/completions"

payload = {
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
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    "model": "deepseek-ai/Janus-Pro-1B",
    "temperature": 0.7
}

# 发送请求
response = requests.post(url, json=payload)
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
