import argparse
import time
import statistics
import os
from src.config import Config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-run image relationship evaluation tool")
    
    parser.add_argument("--jsonl", type=str, required=True, help="JSONL data file path")
    parser.add_argument("--image-root", type=str, required=True, help="Root directory for images")
    parser.add_argument("--prompts", type=str, required=True, help="Prompts JSON file path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for all results")
    parser.add_argument("--summary-name", type=str, default="summary.json", help="Name of summary result file")
    
    # Model mode selection
    parser.add_argument("--model-mode", type=str, choices=["local", "remote"], default="local", 
                        help="Model API mode: local (LLaMA Factory) or remote (OpenAI-compatible)")
    
    # Local model parameters
    parser.add_argument("--llama-api", type=str, default="http://0.0.0.0:37000/v1", help="LLaMA Factory API base URL")
    parser.add_argument("--llama-key", type=str, default="111", help="LLaMA Factory API key")
    parser.add_argument("--llama-model", type=str, default="Qwen2-VL-7B-Instruct", help="LLaMA Factory model name")
    
    # Remote model parameters
    parser.add_argument("--remote-api", type=str, default="https://api.openai.com/v1", help="Remote API base URL")
    parser.add_argument("--remote-key", type=str, default="your-api-key", help="Remote API key")
    parser.add_argument("--remote-model", type=str, default="o4-mini", help="Remote model name")
    parser.add_argument("--no-remote-params", action="store_true", help="Don't send generation parameters to remote API")
    
    # Grading parameters
    parser.add_argument("--grading-api", type=str, default="grading-api", help="Grading API base URL")
    parser.add_argument("--grading-key", type=str, default="grading-key", help="Grading API key")
    parser.add_argument("--grading-model", type=str, default="deepseek-v3-241226", help="Grading model name")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
    
    # Evaluation parameters
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples to evaluate, -1 means all")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent workers")
    parser.add_argument("--prompt-keys", type=str, nargs="+", help="List of prompt keys to use")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per prompt")
    parser.add_argument("--grading-lang", type=str, choices=["en", "zh"], default="en", help="Grading prompt language (en=English, zh=Chinese)")
    parser.add_argument("--no-individual", action="store_true", help="Don't save individual image result files")
    
    args = parser.parse_args()
    
    # If output directory not specified, use default
    if not args.output_dir:
        output_name = os.path.splitext(os.path.basename(args.output))[0]
        args.output_dir = f"{output_name}_individual"
        
    return args

def create_config_from_args(args):
    """Create configuration object from command line arguments"""
    config = Config(
        jsonl_path=args.jsonl,
        image_root_dir=args.image_root,
        prompt_path=args.prompts,
        output_dir=args.output_dir,
        summary_name=args.summary_name,
        
        model_mode=args.model_mode,
        
        llama_api_base=args.llama_api,
        llama_api_key=args.llama_key,
        llama_model=args.llama_model,
        
        remote_api_base=args.remote_api,
        remote_api_key=args.remote_key,
        remote_model=args.remote_model,
        remote_use_params=not args.no_remote_params,
        
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
    """Print detailed performance monitoring dashboard"""
    print("\n" + "="*50)
    print(" "*15 + "Performance Monitoring Dashboard")
    print("="*50)
    
    elapsed = time.time() - stats["start_time"]
    requests_per_second = stats["total_samples_processed"] / elapsed if elapsed > 0 else 0
    
    print(f"Runtime: {elapsed:.2f}s | Concurrency: {config.num_workers}")
    print(f"Total samples processed: {stats['total_samples_processed']} | Rate: {requests_per_second:.2f} samples/sec")
    print(f"Successful samples: {stats['successful_samples']} | Failed samples: {stats['failed_samples']}")
    print(f"Success rate: {(stats['successful_samples']/stats['total_samples_processed']*100):.1f}%" if stats['total_samples_processed'] > 0 else "Success rate: N/A")
    
    print("\nPerformance analysis:")
    print(f"Average inference time: {stats['total_inference_time']/stats['successful_samples']:.2f}s" if stats['successful_samples'] > 0 else "Average inference time: N/A")
    print(f"Average grading time: {stats['total_grading_time']/stats['successful_samples']:.2f}s" if stats['successful_samples'] > 0 else "Average grading time: N/A")
    print(f"Inference time ratio: {(stats['total_inference_time']/(stats['total_inference_time']+stats['total_grading_time'])*100):.1f}%" if (stats['total_inference_time']+stats['total_grading_time']) > 0 else "Inference time ratio: N/A")
    
    if stats["scores"]:
        print(f"\nScore statistics:")
        print(f"Average score: {statistics.mean(stats['scores']):.2f}")
        print(f"Maximum score: {max(stats['scores']):.2f}")
        print(f"Minimum score: {min(stats['scores']):.2f}")
        if len(stats["scores"]) > 1:
            print(f"Score standard deviation: {statistics.stdev(stats['scores']):.2f}")
    
    print("="*50)


def analyze_bottlenecks(evaluator):
    """Analyze and print performance bottlenecks"""
    print("\nPerformance Bottleneck Analysis:")
    
    # Calculate total latency time percentages
    inference_time = sum(r["generation"]["latency"] for r in evaluator.results if "generation" in r and "latency" in r["generation"])
    grading_time = sum(r["grading"]["latency"] for r in evaluator.results if "grading" in r and "latency" in r["grading"])
    total_time = inference_time + grading_time
    
    if total_time > 0:
        print(f"Inference API time: {inference_time:.2f}s ({inference_time/total_time*100:.1f}%)")
        print(f"Grading API time: {grading_time:.2f}s ({grading_time/total_time*100:.1f}%)")
        
        # Determine bottlenecks
        if inference_time > grading_time * 1.5:
            print("Main bottleneck: Inference API - Consider using a faster model or increasing concurrent requests")
        elif grading_time > inference_time * 1.5:
            print("Main bottleneck: Grading API - Consider using a faster grading model or optimizing grading prompts")
        else:
            print("Both APIs have similar performance, optimizing both may yield better results")
        
        # Recommend concurrency
        avg_request_time = total_time / len(evaluator.results) if evaluator.results else 0
        if avg_request_time > 0:
            recommended_workers = min(100, max(2, int(5 / avg_request_time)))
            print(f"Recommended workers: {recommended_workers} (based on average request time: {avg_request_time:.2f}s)")
