import glob
import json
import pandas as pd
import os

import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        # 尝试加载为一个整体的 JSON 数组
        try:
            json_objects = json.loads(data)
        except json.JSONDecodeError:
            # 如果加载失败，尝试拆分为多个独立的 JSON 对象
            json_objects = [json.loads(obj) for obj in data.split('\n') if obj.strip()]
    return json_objects

def extract_properties_to_wide_format(data):
    rows = []
    
    for entry in data:
        materials = entry['materials']
        for material in materials:
            material_name = material['material_name']
            properties = material['properties']
            base_row = {'material_name': material_name}
            for prop in properties:
                col_name = f"{prop['type']}_{prop['conditions']}"
                base_row[col_name] = prop['value']
            rows.append(base_row)
    
    return pd.DataFrame(rows)

def load_all_json_files(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.json"))
    all_data = []
    
    for file_path in all_files:
        data = load_json(file_path)
        all_data.extend(data)
    
    return all_data


folder_path = '/Users/yangz/Documents/projects/llm4catalyst/results'
all_data = load_all_json_files(folder_path)

# 提取所有文件的数据并合并到一个 DataFrame 中
all_rows = []
for entry in all_data:
    df = extract_properties_to_wide_format(entry)
    all_rows.append(df)

final_df = pd.concat(all_rows, ignore_index=True)
print(final_df)