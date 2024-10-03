#!/bin/bash

# 参数
chunk_size=2000
chunk_overlap=$((chunk_size / 10))
base_dir="/Users/yangz/Documents/projects/llm4catalyst/documents"
properties="/Users/yangz/Documents/projects/llm4catalyst/prompts/properties.txt"

# 计数器
count=0

# 遍历 maintext 目录下的所有文件
for maintext in "${base_dir}/maintext/"*_maintext.txt; do
  # 提取文件名去掉末尾的 _maintext
  paper=$(basename "$maintext" "_maintext.txt")
  abstract="${base_dir}/abstract/${paper}_abs.txt"

  # 检查文件是否存在
  if [[ ! -f "$abstract" ]]; then
    echo "Error: Abstract file $abstract not found for $paper!"
    continue
  fi

  if [[ ! -f "$maintext" ]]; then
    echo "Error: Main text file $maintext not found!"
    continue
  fi

  # 运行 python3 main.py 并传递参数
  python3 main.py --chunk_size="$chunk_size" --chunk_overlap="$chunk_overlap" --abstract="$abstract"  --maintext="$maintext" --property="$properties" --title="$paper"

  # 计数
  count=$((count + 1))
  
  # 如果 count > 10 则中断循环
  if [[ "$count" -gt 2 ]]; then
    echo "Processed to limits, stopping the loop."
    break
  fi
done
