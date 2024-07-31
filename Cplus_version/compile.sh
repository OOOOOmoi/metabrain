#!/bin/bash

# 源文件名
source_file="$1"

# 输出文件名
input_file1="ESpikeTimesMT.txt"
input_file2="ISpikeTimesMT.txt"
output_file="EInetMT.png"

# 编译文件名（去除扩展名）
compiled_file="${source_file%.*}"

# 指定生成的可执行文件路径
executable_path="${compiled_file}"

# 编译源文件
g++ -o "${executable_path}" "${source_file}" -lpthread

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable file: ${executable_path}"
    # 运行可执行文件
    ./"${executable_path}" "${input_file1}" "${input_file2}"
    # 执行 Python 文件
    python plot.py --exc_file "${input_file1}" --inh_file "${input_file2}" --output_file "${output_file}"
else
    echo "Compilation failed."
fi
