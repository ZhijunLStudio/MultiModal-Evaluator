import argparse
from src.config import Config

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="多次运行的图连接关系评估工具")
    
    parser.add_argument("--jsonl", type=str, required=True, help="JSONL数据文件路径")
    parser.add_argument("--image-root", type=str, required=True, help="图片根目录")
    parser.add_argument("--prompts", type=str, required=True, help="提示词JSON文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录（存放所有结果）")
    parser.add_argument("--summary-name", type=str, default="summary.json", help="总结果文件名")
    
    parser.add_argument("--llama-api", type=str, default="http://0.0.0.0:37000/v1", help="LLaMA Factory API基础URL")
    parser.add_argument("--llama-key", type=str, default="111", help="LLaMA Factory API密钥")
    parser.add_argument("--llama-model", type=str, default="Qwen2-VL-7B-Instruct", help="LLaMA Factory模型名称")
    
    parser.add_argument("--grading-api", type=str, default="grading-api", help="评分API基础URL")
    parser.add_argument("--grading-key", type=str, default="grading-key", help="评分API密钥")
    parser.add_argument("--grading-model", type=str, default="deepseek-v3-241226", help="评分模型名称")
    
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k采样参数")
    parser.add_argument("--max-tokens", type=int, default=1024, help="最大生成token数")
    
    parser.add_argument("--samples", type=int, default=-1, help="评估样本数量，-1表示全部")
    parser.add_argument("--workers", type=int, default=2, help="并发工作线程数")
    parser.add_argument("--prompt-keys", type=str, nargs="+", help="要使用的prompt keys列表")
    parser.add_argument("--runs", type=int, default=1, help="每个prompt运行的次数")
    parser.add_argument("--grading-lang", type=str, choices=["en", "zh"], default="en", help="评分提示语言 (en=英文, zh=中文)")
    parser.add_argument("--no-individual", action="store_true", help="不保存单独的图片结果文件")
    
    args = parser.parse_args()
    
    # 如果未指定输出目录，使用默认值
    if not args.output_dir:
        output_name = os.path.splitext(os.path.basename(args.output))[0]
        args.output_dir = f"{output_name}_individual"
        
    return args

def create_config_from_args(args):
    """从命令行参数创建配置对象"""
    config = Config(
        jsonl_path=args.jsonl,
        image_root_dir=args.image_root,
        prompt_path=args.prompts,
        output_dir=args.output_dir,
        summary_name=args.summary_name,
        
        llama_api_base=args.llama_api,
        llama_api_key=args.llama_key,
        llama_model=args.llama_model,
        
        grading_api_base=args.grading_api,
        grading_api_key=args.grading_key,
        grading_model=args.grading_model,
        grading_lang=args.grading_lang,
        
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        
        eval_samples=args.samples,
        num_workers=args.workers,
        prompt_keys=args.prompt_keys,
        runs_per_prompt=args.runs,
        
        save_individual=not args.no_individual if hasattr(args, "no_individual") else True
    )
    return config


def print_performance_dashboard(stats, config):
    """打印详细的性能监控仪表盘"""
    print("\n" + "="*50)
    print(" "*15 + "性能监控仪表盘")
    print("="*50)
    
    elapsed = time.time() - stats["start_time"]
    requests_per_second = stats["total_samples_processed"] / elapsed if elapsed > 0 else 0
    
    print(f"运行时间: {elapsed:.2f}秒 | 并发数: {config.num_workers}")
    print(f"总处理样本: {stats['total_samples_processed']} | 速率: {requests_per_second:.2f}样本/秒")
    print(f"成功样本: {stats['successful_samples']} | 失败样本: {stats['failed_samples']}")
    print(f"成功率: {(stats['successful_samples']/stats['total_samples_processed']*100):.1f}%" if stats['total_samples_processed'] > 0 else "成功率: N/A")
    
    print("\n性能分析:")
    print(f"平均推理时间: {stats['total_inference_time']/stats['successful_samples']:.2f}秒" if stats['successful_samples'] > 0 else "平均推理时间: N/A")
    print(f"平均评分时间: {stats['total_grading_time']/stats['successful_samples']:.2f}秒" if stats['successful_samples'] > 0 else "平均评分时间: N/A")
    print(f"推理时间占比: {(stats['total_inference_time']/(stats['total_inference_time']+stats['total_grading_time'])*100):.1f}%" if (stats['total_inference_time']+stats['total_grading_time']) > 0 else "推理时间占比: N/A")
    
    if stats["scores"]:
        print(f"\n评分统计:")
        print(f"平均分数: {statistics.mean(stats['scores']):.2f}")
        print(f"最高分数: {max(stats['scores']):.2f}")
        print(f"最低分数: {min(stats['scores']):.2f}")
        if len(stats["scores"]) > 1:
            print(f"分数标准差: {statistics.stdev(stats['scores']):.2f}")
    
    print("="*50)


def analyze_bottlenecks(evaluator):
    """分析并打印性能瓶颈"""
    print("\n性能瓶颈分析:")
    
    # 计算总延迟时间百分比
    inference_time = sum(r["generation"]["latency"] for r in evaluator.results if "generation" in r and "latency" in r["generation"])
    grading_time = sum(r["grading"]["latency"] for r in evaluator.results if "grading" in r and "latency" in r["grading"])
    total_time = inference_time + grading_time
    
    if total_time > 0:
        print(f"推理API占用时间: {inference_time:.2f}秒 ({inference_time/total_time*100:.1f}%)")
        print(f"评分API占用时间: {grading_time:.2f}秒 ({grading_time/total_time*100:.1f}%)")
        
        # 确定瓶颈
        if inference_time > grading_time * 1.5:
            print("主要瓶颈: 推理API - 考虑使用更快的模型或增加并发请求")
        elif grading_time > inference_time * 1.5:
            print("主要瓶颈: 评分API - 考虑使用更快的评分模型或优化评分提示")
        else:
            print("两个API的性能相近，同时优化两者可能获得更好效果")
        
        # 推荐并发数
        avg_request_time = total_time / len(evaluator.results) if evaluator.results else 0
        if avg_request_time > 0:
            recommended_workers = min(100, max(2, int(5 / avg_request_time)))
            print(f"建议并发数: {recommended_workers} (基于平均请求时间: {avg_request_time:.2f}秒)")
