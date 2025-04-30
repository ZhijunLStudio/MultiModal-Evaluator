import os
import asyncio
import traceback
from src.config import Config
from src.evaluator import Evaluator
from src.utils import parse_args, create_config_from_args, analyze_bottlenecks  # 添加 analyze_bottlenecks 到导入

async def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置对象
    config = create_config_from_args(args)
    
    print("评估配置:")
    print(f"- 数据文件: {config.jsonl_path}")
    print(f"- 图片目录: {config.image_root_dir}")
    print(f"- 提示词文件: {config.prompt_path}")
    print(f"- 输出目录: {config.output_dir}")
    print(f"- 总结果文件: {os.path.join(config.output_dir, config.summary_name)}")
    print(f"- 每个prompt运行次数: {config.runs_per_prompt}")
    print(f"- 使用并发数: {config.num_workers}")
    print(f"- 评估样本数: {config.eval_samples} (-1表示全部)")
    print(f"- 单独保存图片结果: {config.save_individual}")
    
    try:
        # 创建评估器并运行
        evaluator = Evaluator(config)
        await evaluator.run()
        
        # 分析性能瓶颈
        analyze_bottlenecks(evaluator)
        
        # 保存结果
        evaluator.save_results()
        
        print("\n评估完成!")
        
    except Exception as e:
        print(f"\n评估过程出错: {str(e)}")
        print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    # 在Windows平台上需要这个设置
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
