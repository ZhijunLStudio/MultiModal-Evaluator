import os
import json
import glob
import re

# 定义源文件夹和目标文件夹
source_folder = "benchmark_output/qwen2.5-vl-32B-origin"
target_folder = "benchmark_output/0523-qwen2.5-vl-32B-origin"

# 确保目标文件夹存在
os.makedirs(target_folder, exist_ok=True)

# 用于计算整体平均值的变量
total_precision = 0
total_recall = 0
total_f1 = 0
total_miss_rate = 0
total_fa_rate = 0
total_conn_ratio = 0
file_count = 0

# 打印指标定义
print("===== 图形连接分析指标定义 =====")
print("p (精确率): 正确识别的连接数 ÷ 模型生成的连接总数 - 显示生成连接的准确性")
print("r (召回率): 正确识别的连接数 ÷ 参考标准中的连接总数 - 显示覆盖了多少参考连接")
print("f1 (F1分数): 2 * (精确率 * 召回率) ÷ (精确率 + 召回率) - 精确率与召回率的调和平均数")
print("miss (漏报率): 参考中未被识别的连接比例 = 1 - 召回率 - 显示模型漏掉的连接占比")
print("fa (误报率): 错误生成的连接比例 = 1 - 精确率 - 显示模型错误生成的连接占比")
print("cr (连接完整性比率): 生成的连接总数 ÷ 参考的连接总数 - >1表示过度生成，<1表示生成不足")
print("===============================\n")

# 处理所有JSON文件
for json_path in glob.glob(os.path.join(source_folder, "*.json")):
    filename = os.path.basename(json_path)
    
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 查找metrics字段和连接分析数据
        metrics = None
        connection_analysis = None
        
        # 检查results中的所有键，找到以"prompt"开头的
        if "results" in data:
            prompt_keys = [k for k in data["results"].keys() if k.startswith("prompt")]
            
            # 遍历所有prompt键
            for prompt_key in prompt_keys:
                if prompt_key in data["results"] and len(data["results"][prompt_key]) > 0:
                    if "connection_analysis" in data["results"][prompt_key][0]:
                        connection_analysis = data["results"][prompt_key][0]["connection_analysis"]
                        if "metrics" in connection_analysis:
                            metrics = connection_analysis["metrics"]
                        break  # 找到分析数据后跳出循环
        
        if metrics and "precision" in metrics and "recall" in metrics and connection_analysis:
            # 提取指标值 - 确保值在0-100的范围内
            precision = metrics["precision"] * 100 if metrics["precision"] <= 1 else metrics["precision"]
            recall = metrics["recall"] * 100 if metrics["recall"] <= 1 else metrics["recall"]
            
            # 计算F1分数
            if precision > 0 and recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0
                
            # 计算漏报率和误报率
            miss_rate = 100 - recall  # 漏报率 = 1 - 召回率
            fa_rate = 100 - precision  # 误报率 = 1 - 精确率
            
            # 计算连接完整性比率 - 修复后的代码
            if "comparison" in connection_analysis:
                total_gen = connection_analysis["comparison"].get("total_gen", 0)
                total_ref = connection_analysis["comparison"].get("total_ref", 0)
            else:
                # 尝试从gen_connections和ref_connections列表中获取
                gen_connections = connection_analysis.get("gen_connections", [])
                ref_connections = connection_analysis.get("ref_connections", [])
                total_gen = len(gen_connections) if isinstance(gen_connections, list) else 0
                total_ref = len(ref_connections) if isinstance(ref_connections, list) else 0
            
            # 计算连接比率，避免除零错误
            conn_ratio = total_gen / total_ref if total_ref > 0 else 0
            
            # 更新总计
            total_precision += precision
            total_recall += recall
            total_f1 += f1_score
            total_miss_rate += miss_rate
            total_fa_rate += fa_rate
            total_conn_ratio += conn_ratio
            file_count += 1
            
            # 创建新文件名 - 替换map_xxx为多个指标
            new_filename = re.sub(r'map_[\d\.]+', 
                                f'p_{precision:.2f}_r_{recall:.2f}_f1_{f1_score:.2f}_miss_{miss_rate:.2f}_fa_{fa_rate:.2f}_cr_{conn_ratio:.2f}', 
                                filename)
            
            # 保存到新位置
            new_filepath = os.path.join(target_folder, new_filename)
            with open(new_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"处理文件: {filename} -> {new_filename}")
    
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")

# 打印整体平均值
if file_count > 0:
    print("\n===== 统计结果 =====")
    print(f"总处理文件数: {file_count}")
    print(f"平均精确率 (P): {total_precision/file_count:.4f}")
    print(f"平均召回率 (R): {total_recall/file_count:.4f}")
    print(f"平均F1分数: {total_f1/file_count:.4f}")
    print(f"平均漏报率: {total_miss_rate/file_count:.4f}")
    print(f"平均误报率: {total_fa_rate/file_count:.4f}")
    print(f"平均连接完整性比率 (CR): {total_conn_ratio/file_count:.4f}")
    print("=====================")
else:
    print("没有找到包含所需metrics字段的文件。")
