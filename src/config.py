from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class Config:
    # Path configuration
    jsonl_path: str = "data.jsonl"
    image_root_dir: str = "images/"
    prompt_path: str = "prompts.json"
    output_dir: str = "results/"  # Output directory
    summary_name: str = "summary.json"  # Summary file name
    
    # Model mode: 'local' or 'remote'
    model_mode: str = "local"
    
    # Local API configuration
    llama_api_base: str = "http://0.0.0.0:37000/v1"
    llama_api_key: str = "111"
    llama_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    
    # Remote API configuration
    remote_api_base: str = "https://api.openai.com/v1"
    remote_api_key: str = "your-api-key"
    remote_model: str = "o4-mini"
    remote_use_params: bool = True
    
    # Grading API configuration
    grading_api_base: str = "api-grader"
    grading_api_key: str = "your api key here"
    grading_model: str = "deepseek-v3-241226"
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096
    
    # Evaluation configuration
    eval_samples: int = -1  # -1 means evaluate all samples
    num_workers: int = 1    # Number of concurrent workers
    
    # Prompt configuration
    prompt_keys: List[str] = None  # List of prompt keys to use
    runs_per_prompt: int = 1       # Number of runs per prompt
    grading_lang: str = "en"  # Default to English
    
    # Output configuration
    verbose: bool = True
    save_individual: bool = True  # Whether to save individual image results
