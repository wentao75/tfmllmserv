# TF Model LLM Service

基于 FastAPI 的多模态大语言模型服务，支持图文理解和生成。

## 环境要求

- Python 3.10
- Conda 环境管理
- MacOS MPS (Apple Silicon) 或 NVIDIA GPU
- Docker（可选，用于Open WebUI）

## 快速开始

1. 克隆仓库：
```bash
git clone [repository_url]
cd tfmllmserv
```

2. 运行服务：
```bash
./run.sh
```
脚本会自动完成以下操作：
- 检查并创建 conda 环境
- 安装所需依赖
- 设置必要的环境变量
- 启动服务

服务将在 http://0.0.0.0:8000 上运行

## Open WebUI 部署

### 首次部署

1. 拉取最新镜像：
```bash
docker pull ghcr.io/open-webui/open-webui:main
```

2. 运行容器：
```bash
docker run -d \
  --name open-webui \
  -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e DEFAULT_MODELS='[{"name":"InternVL2-2B","type":"openai"}]' \
  -e OPENAI_API_BASE=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=sk-xxx \
  ghcr.io/open-webui/open-webui:main
```

### 更新部署

1. 停止并删除旧容器：
```bash
docker stop open-webui
docker rm open-webui
```

2. 拉取最新镜像：
```bash
docker pull ghcr.io/open-webui/open-webui:main
```

3. 重新运行容器（使用上述运行命令）

### 环境变量说明

- `DEFAULT_MODELS`: 预配置的模型列表
- `OPENAI_API_BASE`: 后端API地址
- `OPENAI_API_KEY`: API密钥（可以是任意值）

访问地址：http://localhost:8080

## 环境依赖

主要依赖包括：
```yaml
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - fastapi
  - uvicorn
  - pydantic
  - pillow
  - transformers
  - timm
```

完整的依赖配置见 `environment.yml`

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

## API 接口

### 1. 聊天补全接口

```
POST /v1/chat/completions
```

支持以下功能：
- 纯文本对话
- 图文理解
- 图像识别

请求格式：
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "识别图片中的文字"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "base64_encoded_image_data"
                    }
                }
            ]
        }
    ],
    "temperature": 0.7,
    "max_tokens": 2048
}
```

### 2. 模型列表接口

```
GET /v1/models
```

返回可用的模型列表及其状态。

## 开发说明

1. 添加新依赖：
   - 在 `environment.yml` 中添加依赖
   - 重新运行 `./run.sh` 更新环境

2. 修改配置：
   - 服务配置在 `tfmllmserv/api.py`
   - 模型管理在 `tfmllmserv/model_manager.py`

## 性能指标

- 响应时间：约 4-5 秒（基于测试数据）
- 支持并发请求
- 内存使用：取决于加载的模型大小

## 日志说明

服务会输出详细的日志信息，包括：
- 服务启动状态
- 模型加载进度
- 请求处理时间
- 错误信息

## 安全说明

1. CORS 已配置为允许所有源
2. 建议在生产环境中：
   - 限制允许的源
   - 添加适当的认证机制
   - 配置 HTTPS
