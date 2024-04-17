#!/bin/bash

# 源文件名
source_file="$1"

# 编译文件名（去除扩展名）
compiled_file="${source_file%.*}"

# 指定生成的可执行文件路径
executable_path="bin/${compiled_file}"

# 编译源文件
g++ -o "${executable_path}" "${source_file}" -lpthread

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable file: ${executable_path}"
    # 运行可执行文件
    ./"${executable_path}"
    # 执行 Python 文件
    python plot.py
else
    echo "Compilation failed."
fi
