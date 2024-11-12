import json
import argparse

# 从文件中读取 JSON 数据
def read_json(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {input_file}: {e}")
            return None

# 将处理后的数据保存到指定文件
def save_json(output_file, data):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# 合并相同材料的函数
def merge_materials(data):
    merged_materials = {}
    
    # 遍历材料列表
    for material in data['materials']:
        name = material['material_name']
        
        # 如果材料名已经存在，合并其属性
        if name in merged_materials:
            for prop, value in material['properties'].items():
                # 仅当新的属性值不为 None 且不为 'NA' 或 '[]' 时，更新该属性
                if value['value'] not in [None, 'NA', '[]']:
                    merged_materials[name]['properties'][prop] = value
        else:
            # 检查 properties 是否为字典，如果是字典才处理
            if isinstance(material['properties'], dict):
                merged_materials[name] = {
                    'material_name': name,
                    'properties': {k: v for k, v in material['properties'].items() if v['value'] not in [None, 'NA', '[]']},
                    'benchmark': material['benchmark']
                }
            else:
                print(f"Warning: properties for material {name} is not a dictionary, skipping.")
    
    return merged_materials

# 读取、处理和保存 JSON 数据
def process_json(input_file, output_file):
    # 读取 JSON 数据
    data = read_json(input_file)
    
    if data is None:
        print(f"Skipping {input_file} due to JSON decoding errors.")
        return
    
    # 处理数据：合并相同材料，过滤无效值
    merged_materials = merge_materials(data)
    
    # 保留所有其他的 metadata 信息，并将合并后的材料替换进去
    merged_data = data.copy()
    merged_data['materials'] = list(merged_materials.values())
    
    # 保存处理后的数据
    save_json(output_file, merged_data)
    print(f"处理后的数据已保存到 {output_file}")

# 主函数，解析命令行参数
def main():
    parser = argparse.ArgumentParser(description='处理 JSON 文件，合并相同材料并去除无效值')
    parser.add_argument('input_file', type=str, help='输入 JSON 文件路径')
    parser.add_argument('output_file', type=str, help='输出 JSON 文件路径')
    args = parser.parse_args()

    # 调用处理函数
    process_json(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
