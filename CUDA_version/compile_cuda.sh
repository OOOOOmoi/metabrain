#!/bin/bash

# 检查是否提供了文件名作为参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <cuda_file.cu>"
    exit 1
fi

# 赋值参数到变量
CUDA_FILE=$1

# 去除文件扩展名，获取基础文件名
BASENAME=$(basename "$CUDA_FILE" .cu)

# 使用nvcc编译CUDA文件，生成同名的可执行文件
nvcc -o $BASENAME $CUDA_FILE

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "Compilation successful. Executing $BASENAME ..."
    ./$BASENAME
else
    echo "Compilation failed."
fi