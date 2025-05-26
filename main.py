import os
import asyncio
import traceback
from src.config import Config
from src.evaluator import Evaluator
from src.utils import parse_args, create_config_from_args, analyze_bottlenecks

async def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create configuration object
    config = create_config_from_args(args)
    
    print("Evaluation Configuration:")
    print(f"- Data file: {config.jsonl_path}")
    print(f"- Image directory: {config.image_root_dir}")
    print(f"- Prompt file: {config.prompt_path}")
    print(f"- Output directory: {config.output_dir}")
    print(f"- Summary file: {os.path.join(config.output_dir, config.summary_name)}")
    print(f"- Runs per prompt: {config.runs_per_prompt}")
    print(f"- Number of workers: {config.num_workers}")
    print(f"- Evaluation samples: {config.eval_samples} (-1 means all)")
    print(f"- Save individual results: {config.save_individual}")
    print(f"\nAnswer API Configuration:")
    print(f"- API URL: {config.answer_api_base}")
    print(f"- Model: {config.answer_model}")
    print(f"- Temperature: {config.answer_temperature}")
    print(f"- Top-P: {config.answer_top_p}")
    print(f"- Top-K: {config.answer_top_k}")
    print(f"- Max tokens: {config.answer_max_tokens}")
    print(f"\nGrading API Configuration:")
    print(f"- API URL: {config.grading_api_base}")
    print(f"- Model: {config.grading_model}")
    print(f"- Temperature: {config.grading_temperature}")
    print(f"- Top-P: {config.grading_top_p}")
    print(f"- Max tokens: {config.grading_max_tokens}")
    
    try:
        # Create evaluator and run
        evaluator = Evaluator(config)
        await evaluator.run()
        
        # Analyze performance bottlenecks
        analyze_bottlenecks(evaluator)
        
        # Save results
        evaluator.save_results()
        
        print("\nEvaluation completed!")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    # Required for Windows platform
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
