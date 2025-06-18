import json

def convert_json_to_jsonl_remove_backticks(input_filepath, output_filepath):
    """
    将 JSON 文件转换为 JSONL 文件，并移除 'answer' 字段中的所有反引号 (```)。

    Args:
        input_filepath (str): 输入 JSON 文件的路径。
        output_filepath (str): 输出 JSONL 文件的路径。
    """
    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for item in data:
                if 'answer' in item and isinstance(item['answer'], str):
                    # 移除所有反引号字符
                    item['answer'] = item['answer'].replace('```', '')
                
                json.dump(item, outfile, ensure_ascii=False)
                outfile.write('\n')
        print(f"成功将 '{input_filepath}' 转换为 '{output_filepath}'，并移除了 'answer' 字段中的反引号。")
    except FileNotFoundError:
        print(f"错误：文件 '{input_filepath}' 未找到。请检查文件路径是否正确。")
    except json.JSONDecodeError:
        print(f"错误：文件 '{input_filepath}' 不是一个有效的 JSON 文件。请检查文件内容格式。")
    except Exception as e:
        print(f"发生了一个未知错误: {e}")

# --- 使用示例 ---
# 请将 'your_input.json' 替换为您的实际 JSON 文件名
# 将 'your_output.jsonl' 替换为您希望的输出 JSONL 文件名
input_json_file = 'system_block_benchmark_v2_verilogA.json'
output_jsonl_file = 'benchmark/system_block_benchmark_verilog_v1.jsonl'

# 执行转换
convert_json_to_jsonl_remove_backticks(input_json_file, output_jsonl_file)