#!/bin/bash

# 参数
chunk_size=1000
chunk_overlap=$((chunk_size / 10))
paper="c60"
base_dir="/Users/yangz/Documents/projects/llm4catalyst/documents"
abstract="${base_dir}/abstract/${paper}_abs.txt"
intro="${base_dir}/intro/${paper}_intro.txt"
maintext="${base_dir}/maintext/${paper}_maintext.txt"

# 检查文件是否存在
if [[ ! -f "$abstract" ]]; then
  echo "Error: Abstract file $abstract not found!"
  exit 1
fi

if [[ ! -f "$intro" ]]; then
  echo "Error: Introduction file $intro not found!"
  exit 1
fi

if [[ ! -f "$maintext" ]]; then
  echo "Error: Main text file $maintext not found!"
  exit 1
fi

<<<<<<< Updated upstream
# 运行 python3 main.py 并传递参数
python3 main.py --chunk_size="$chunk_size" --chunk_overlap="$chunk_overlap" --abstract="$abstract" --intro="$intro" --maintext="$maintext"
=======
  if [[ ! -f "$maintext" ]]; then
    echo "Error: Main text file $maintext not found!"
    continue
  fi

  # 运行 python3 main.py 并传递参数
  python3 main.py --chunk_size="$chunk_size" --chunk_overlap="$chunk_overlap" --abstract="$abstract"  --maintext="$maintext" --property="$properties" --title="$paper"

  # 计数
  count=$((count + 1))
  
  # 如果 count > 10 则中断循环
  if [[ "$count" -gt 0 ]]; then
    echo "Processed to limits, stopping the loop."
    break
  fi
done
>>>>>>> Stashed changes
