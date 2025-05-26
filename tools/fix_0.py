import os
import json
import re
import glob
import shutil
from typing import List, Tuple, Dict, Any, Set
import statistics

# 从GradingClient类中提取关键方法
def clean_node_names(node_name: str) -> str:
    """清理节点名称，移除引号、括号等"""
    if not isinstance(node_name, str):
        return str(node_name)
    
    # 首先提取纯节点名称，去除属性标签
    bracket_index = node_name.find('[')
    if bracket_index != -1:
        clean_name = node_name[:bracket_index].strip()
    else:
        clean_name = node_name.strip()
    
    # 去除分号和其他潜在的分隔符
    clean_name = clean_name.rstrip(';').strip()
    
    # 移除引号
    if (clean_name.startswith('"') and clean_name.endswith('"')) or \
       (clean_name.startswith("'") and clean_name.endswith("'")):
        clean_name = clean_name[1:-1]
    
    return clean_name.strip()

def extract_connections(text: str) -> List[Tuple[str, str, str]]:
    """提取文本中的连接关系"""
    connections = []
    
    # 按行拆分文本
    lines = text.split('\n')
    
    for line in lines:
        # 提取箭头连接 (->)
        # 修改正则表达式以处理带引号的节点名称
        arrow_pattern = re.compile(r'(?:"([^"]+)"|\'([^\']+)\'|(\w+(?:[-_]\w+)*))\s*->\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+(?:[-_]\w+)*)(?:\[.*?\])?)')
        arrow_matches = arrow_pattern.findall(line)
        
        for match in arrow_matches:
            # 处理多个捕获组，选择非空值
            src = next((s for s in match[:3] if s), "")
            tgt_parts = match[3:]
            tgt_with_attrs = next((t for t in tgt_parts if t), "")
            
            # 提取目标节点名称（去除属性标签）
            bracket_index = tgt_with_attrs.find('[')
            if bracket_index != -1:
                tgt = tgt_with_attrs[:bracket_index].strip()
            else:
                tgt = tgt_with_attrs.strip()
            
            if src and tgt:  # 确保都不为空
                connections.append((src.strip(), "->", tgt.strip()))
        
        # 提取双向连接 (<->)
        bidirectional_pattern = re.compile(r'(?:"([^"]+)"|\'([^\']+)\'|(\w+(?:[-_]\w+)*))\s*<->\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+(?:[-_]\w+)*)(?:\[.*?\])?)')
        bidir_matches = bidirectional_pattern.findall(line)
        
        for match in bidir_matches:
            src = next((s for s in match[:3] if s), "")
            tgt_parts = match[3:]
            tgt_with_attrs = next((t for t in tgt_parts if t), "")
            
            bracket_index = tgt_with_attrs.find('[')
            if bracket_index != -1:
                tgt = tgt_with_attrs[:bracket_index].strip()
            else:
                tgt = tgt_with_attrs.strip()
                
            if src and tgt:
                connections.append((src.strip(), "<->", tgt.strip()))
        
        # 提取双连字符连接 (--)
        double_dash_pattern = re.compile(r'(?:"([^"]+)"|\'([^\']+)\'|(\w+(?:[-_]\w+)*))\s*--\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+(?:[-_]\w+)*)(?:\[.*?\])?)')
        double_dash_matches = double_dash_pattern.findall(line)
        
        for match in double_dash_matches:
            src = next((s for s in match[:3] if s), "")
            tgt_parts = match[3:]
            tgt_with_attrs = next((t for t in tgt_parts if t), "")
            
            bracket_index = tgt_with_attrs.find('[')
            if bracket_index != -1:
                tgt = tgt_with_attrs[:bracket_index].strip()
            else:
                tgt = tgt_with_attrs.strip()
                
            if src and tgt:
                connections.append((src.strip(), "--", tgt.strip()))
    
    # 移除重复连接
    unique_connections = []
    seen = set()
    
    for src, conn_type, tgt in connections:
        # 去除末尾可能的分号
        src = src.rstrip(';')
        tgt = tgt.rstrip(';')
        
        connection_key = f"{src}|{conn_type}|{tgt}"
        if connection_key not in seen:
            seen.add(connection_key)
            unique_connections.append((src, conn_type, tgt))
    
    return unique_connections

def format_connection(conn: Tuple[str, str, str]) -> str:
    """格式化连接元组为字符串"""
    src = clean_node_names(conn[0])
    tgt = clean_node_names(conn[2])
    return f"{src} {conn[1]} {tgt}"

def compare_connections(gen_connections: List[Tuple[str, str, str]], 
                        ref_connections: List[Tuple[str, str, str]]) -> Dict[str, Any]:
    """比较生成和参考连接"""
    gen_conn_set = set(format_connection(c) for c in gen_connections)
    ref_conn_set = set(format_connection(c) for c in ref_connections)
    
    # 精确匹配的连接
    exact_matches = gen_conn_set.intersection(ref_conn_set)
    
    # 未匹配的连接
    unmatched_gen = [conn for conn in gen_connections 
                    if format_connection(conn) not in exact_matches]
    
    unmatched_ref = [conn for conn in ref_connections 
                    if format_connection(conn) not in exact_matches]
    
    # 计算指标
    precision = len(exact_matches) / len(gen_conn_set) if gen_conn_set else 0.0
    recall = len(exact_matches) / len(ref_conn_set) if ref_conn_set else 0.0
    
    # 保留两位小数
    precision = round(precision, 2)
    recall = round(recall, 2)
    
    result = {
        "exact_matches": list(exact_matches),
        "exact_match_count": len(exact_matches),
        "unmatched_gen": [format_connection(c) for c in unmatched_gen],
        "unmatched_ref": [format_connection(c) for c in unmatched_ref],
        "total_gen": len(gen_connections),
        "total_ref": len(ref_connections),
        "precision": precision,
        "recall": recall
    }
    
    return result

def calculate_metrics(exact_matches: int, semantic_matches: int, 
                    total_gen: int, total_ref: int) -> Dict[str, float]:
    """计算评估指标"""
    # 确保总匹配数不超过参考数量
    total_matches = min(exact_matches + semantic_matches, total_ref)
    
    # 计算精确率和召回率
    precision = total_matches / total_gen if total_gen > 0 else 0.0
    recall = total_matches / total_ref if total_ref > 0 else 0.0
    
    # 计算F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 连接完整性比率
    conn_ratio = total_gen / total_ref if total_ref > 0 else 0.0
    
    # 全对率
    perfect_match = 1.0 if abs(f1 - 1.0) < 0.001 else 0.0
    
    # 确保所有浮点数都保留两位小数
    return {
        "precision": round(min(1.0, max(0.0, precision)), 2),
        "recall": round(min(1.0, max(0.0, recall)), 2),
        "f1": round(min(1.0, max(0.0, f1)), 2),
        "conn_ratio": round(conn_ratio, 2),
        "perfect_match": round(perfect_match, 2)
    }

class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，确保浮点数保存为两位小数"""
    def encode(self, obj):
        if isinstance(obj, float):
            return format(obj, '.2f')
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        if isinstance(obj, float):
            return (format(obj, '.2f'),)
        return super().iterencode(obj, _one_shot)

def recompute_metrics(input_path: str, output_path: str):
    """重新计算F1=0的评估结果，并保存所有结果到新目录"""
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(input_path, "*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 收集统计数据
    total_fixed = 0
    original_f1_values = []
    new_f1_values = []
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_path, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否存在F1=0的情况
            has_zero_f1 = False
            if "results" in data and "prompt4" in data["results"]:
                for run in data["results"]["prompt4"]:
                    if "connection_analysis" in run and "metrics" in run["connection_analysis"]:
                        metrics = run["connection_analysis"]["metrics"]
                        if metrics["f1"] == 0:
                            has_zero_f1 = True
                            original_f1_values.append(0)
                            break
            
            # 如果没有F1=0的情况，直接复制文件
            if not has_zero_f1:
                shutil.copy2(file_path, output_file_path)
                continue
            
            # 处理F1=0的情况
            modified = False
            
            if "reference" in data and "results" in data and "prompt4" in data["results"]:
                reference_text = data["reference"]
                
                for i, run in enumerate(data["results"]["prompt4"]):
                    # 确认是F1=0的情况
                    if "connection_analysis" in run and "metrics" in run["connection_analysis"] and run["connection_analysis"]["metrics"]["f1"] == 0:
                        generation_text = run["generation"]
                        
                        # 使用改进的提取方法重新获取连接
                        gen_connections = extract_connections(generation_text)
                        ref_connections = extract_connections(reference_text)
                        
                        # 如果没有连接被提取，跳过
                        if not gen_connections or not ref_connections:
                            continue
                        
                        # 重新比较连接
                        comparison = compare_connections(gen_connections, ref_connections)
                        
                        # 重新计算指标
                        metrics = calculate_metrics(
                            comparison["exact_match_count"], 0,
                            comparison["total_gen"], comparison["total_ref"]
                        )
                        
                        # 更新结果
                        run["connection_analysis"]["comparison"] = comparison
                        run["connection_analysis"]["gen_connections"] = [format_connection(c) for c in gen_connections]
                        run["connection_analysis"]["ref_connections"] = [format_connection(c) for c in ref_connections]
                        run["connection_analysis"]["metrics"] = metrics
                        run["connection_analysis"]["final_unmatched_gen"] = comparison["unmatched_gen"]
                        run["connection_analysis"]["final_unmatched_ref"] = comparison["unmatched_ref"]
                        
                        # 更新分数
                        run["score"] = int(min(100, metrics["f1"] * 100))
                        
                        # 如果计算出的F1不为0，标记为修改成功
                        if metrics["f1"] > 0:
                            modified = True
                            total_fixed += 1
                            new_f1_values.append(metrics["f1"])
                        else:
                            new_f1_values.append(0)
            
            # 保存修改后的结果（使用自定义编码器确保浮点数为两位小数）
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
            
            if modified:
                print(f"修复成功: {file_name}")
            else:
                print(f"无法修复: {file_name}")
                
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            # 如果出错，复制原始文件
            shutil.copy2(file_path, output_file_path)
    
    # 打印统计信息
    print("\n统计结果:")
    print(f"总共修复的F1=0案例: {total_fixed}")
    
    if new_f1_values:
        avg_f1_before = sum(original_f1_values) / len(original_f1_values) if original_f1_values else 0
        avg_f1_after = sum(new_f1_values) / len(new_f1_values) if new_f1_values else 0
        print(f"修复前平均F1: {avg_f1_before:.2f}")
        print(f"修复后平均F1: {avg_f1_after:.2f}")
        
    # 计算整个数据集的指标
    calculate_dataset_metrics(output_path)

def calculate_dataset_metrics(directory_path: str):
    """计算整个数据集的指标平均值"""
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    all_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "conn_ratio": [],
        "perfect_match": []
    }
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "results" in data and "prompt4" in data["results"]:
                for run in data["results"]["prompt4"]:
                    if "connection_analysis" in run and "metrics" in run["connection_analysis"]:
                        metrics = run["connection_analysis"]["metrics"]
                        for key in all_metrics.keys():
                            if key in metrics:
                                all_metrics[key].append(float(metrics[key]))
        except Exception as e:
            print(f"读取文件 {os.path.basename(file_path)} 的指标时出错: {str(e)}")
    
    # 计算平均值
    print("\n数据集整体指标:")
    for key, values in all_metrics.items():
        if values:
            avg = sum(values) / len(values)
            print(f"平均{key}: {avg:.4f}")
            
            # 计算中位数并保留两位小数
            median = statistics.median(values)
            print(f"{key}中位数: {median:.4f}")
            
            # 计算完美匹配率
            if key == "perfect_match":
                perfect_ratio = sum(values) / len(values)
                print(f"完美匹配率: {perfect_ratio:.4f}")
        else:
            print(f"未找到{key}指标")


if __name__ == "__main__":
    # 设置输入和输出路径
    input_directory = "benchmark_output/0523-o4mini-benchmark-v2-v1.1"  # 输入原始结果目录
    output_directory = "benchmark_output/0523-o4mini-benchmark-v2-v1.1-fixed"  # 输出修复后结果目录
    
    recompute_metrics(input_directory, output_directory)
