#!/bin/bash

# 参数
chunk_size=1000
chunk_overlap=$((chunk_size / 10))
base_dir="/Users/yangz/Documents/projects/llm4catalyst/documents"
abstract_dir="${base_dir}/abstract"
intro_dir="${base_dir}/intro"
maintext_dir="${base_dir}/maintext"
properties="/Users/yangz/Documents/projects/llm4catalyst/prompts/properties.txt"  # 更新为 properties 文件的路径

# 检查用户是否传入了循环次数参数
if [[ -z "$1" ]]; then
  echo "Error: Please specify the number of loops as an argument."
  exit 1
fi

# 循环次数
loop_count=$1

# 检查文件夹是否存在
if [[ ! -d "$abstract_dir" ]]; then
  echo "Error: Abstract directory $abstract_dir not found!"
  exit 1
fi

if [[ ! -d "$intro_dir" ]]; then
  echo "Error: Introduction directory $intro_dir not found!"
  exit 1
fi

if [[ ! -d "$maintext_dir" ]]; then
  echo "Error: Main text directory $maintext_dir not found!"
  exit 1
fi

# 计数器初始化
count=0

# 遍历 maintext 文件夹下的所有 .txt 文件
for maintext_file in "$maintext_dir"/*.txt; do
  # 检查是否超过了指定的循环次数
  if [[ "$count" -ge "$loop_count" ]]; then
    echo "Processed to the specified limit, stopping the loop."
    break
  fi

  # 提取文件名作为 paper 的值 (去掉路径和扩展名)
  paper=$(basename "$maintext_file" _maintext.txt)

  # 检查对应的 abstract 和 intro 文件是否存在
  abstract="${abstract_dir}/${paper}_abs.txt"
  intro="${intro_dir}/${paper}_intro.txt"
  
  if [[ ! -f "$abstract" ]]; then
    echo "Warning: Abstract file $abstract not found, skipping this paper."
    continue
  fi
  
  if [[ ! -f "$intro" ]]; then
    echo "Warning: Introduction file $intro not found, skipping this paper."
    continue
  fi

  # 运行 python3 main.py 并传递参数
  python3 main.py --chunk_size="$chunk_size" --chunk_overlap="$chunk_overlap" --abstract="$abstract" --intro="$intro" --maintext="$maintext_file" --property="$properties" --title="$paper"
  
  # 计数
  count=$((count + 1))
  
  echo "Iteration $count completed, processed paper: $paper"
done

echo "Processed $count iterations."
