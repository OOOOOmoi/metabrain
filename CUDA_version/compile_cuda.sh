#!/bin/bash

# 设置输出目录
output_dir="/home/yangjinhao/stucture/Cplus_version/exe"

# 检查是否提供了文件名
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename.cu>"
    exit 1
fi

# 获取输入的文件名
input_file="$1"

# 检查文件是否存在
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# 提取没有扩展名的文件名
filename=$(basename "$input_file")
executable="${filename%.*}"

# 确保输出目录存在
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    if [ $? -ne 0 ]; then
        echo "Failed to create output directory '$output_dir'."
        exit 1
    fi
fi

# 使用nvcc编译CUDA文件到指定的输出目录
nvcc -o "${output_dir}/${executable}" "$input_file"

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# 执行编译后的程序
echo "Running '${output_dir}/${executable}'..."
"${output_dir}/${executable}"
