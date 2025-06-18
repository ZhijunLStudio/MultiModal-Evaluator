import os
import json
import numpy as np
from collections import defaultdict
from tabulate import tabulate

def analyze_metrics(json_dir):
    """分析指定目录下所有JSON文件中的评估指标"""
    # 初始化指标统计
    metrics_summary = {
        'module_mappings_metrics': defaultdict(list),
        'module_metrics': defaultdict(list),
        'connection_metrics': defaultdict(list)
    }
    
    # 遍历目录下的所有JSON文件
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(json_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 收集各项指标
            for metric_type in metrics_summary.keys():
                if metric_type in data:
                    for metric_name, value in data[metric_type].items():
                        if isinstance(value, (int, float)) and metric_name in ['precision', 'recall', 'f1']:
                            metrics_summary[metric_type][metric_name].append(value)
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            continue
    
    # 计算统计结果
    results = {}
    for metric_type, metrics in metrics_summary.items():
        results[metric_type] = {}
        for metric_name, values in metrics.items():
            if values:  # 确保有数据
                results[metric_type][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': int(len(values))
                }
    
    return results

def print_results(results):
    """打印统计结果"""
    print("\n=== 评估指标统计结果 ===\n")
    
    # 准备表格数据
    headers = ['指标类型', '评估指标', '平均值', '标准差', '最小值', '最大值', '中位数', '样本数']
    table_data = []
    
    # 指标类型的中文映射
    type_names = {
        'module_metrics': '模块匹配',
        'connection_metrics': '连接匹配',
        'module_mappings_metrics': '模块映射'
    }
    
    # 指标名称的中文映射
    metric_names = {
        'precision': '精确率',
        'recall': '召回率',
        'f1': 'F1分数'
    }
    
    for metric_type, metrics in results.items():
        for metric_name, stats in metrics.items():
            if metric_name in ['precision', 'recall', 'f1']:
                row = [
                    type_names[metric_type],
                    metric_names[metric_name],
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                    f"{stats['median']:.4f}",
                    stats['count']
                ]
                table_data.append(row)
    
    # 打印表格
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def main():
    json_dir = ".cache/results"
    results = analyze_metrics(json_dir)
    print_results(results)
    
    # 保存结果到文件
    output_file = "metrics_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n详细结果已保存到 {output_file}")

if __name__ == "__main__":
    main() 