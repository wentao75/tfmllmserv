#!/bin/bash

# 获取conda的安装路径
CONDA_BASE=$(conda info --base)

# 检查conda环境是否存在
if ! $CONDA_BASE/bin/conda env list | grep -q "^tfmllmserv "; then
    echo "创建conda环境..."
    $CONDA_BASE/bin/conda env create -f environment.yml
fi

# 激活环境
echo "激活conda环境..."
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate tfmllmserv

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 启动服务
echo "启动服务..."
python3 -m tfmllmserv.main 