#!/bin/bash

# 基础目录
base_dir="/Users/yangz/Documents/projects/llm4catalyst/results"
output_dir="${base_dir}/merged"

# 检查 output 目录是否存在，不存在则创建
if [[ ! -d "$output_dir" ]]; then
  echo "Output directory $output_dir not found, creating it..."
  mkdir -p "$output_dir"
fi

# 遍历 results 目录下所有 .json 文件
for json_file in "$base_dir"/*.json; do
  # 提取文件名，不包含路径和扩展名
  filename=$(basename "$json_file" .json)

  # 设置输出文件路径
  output_file="${output_dir}/${filename}_merged.json"

  # 运行 Python 脚本处理每个 .json 文件
  python3 merge.py "$json_file" "$output_file"
  
  # 打印日志
  echo "Processed $json_file and saved to $output_file"
done

echo "All JSON files have been processed and saved to $output_dir."
