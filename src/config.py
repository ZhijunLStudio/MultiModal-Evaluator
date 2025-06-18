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
    
    # Answer API configuration
    answer_api_base: str = "http://0.0.0.0:37000/v1"
    answer_api_key: str = "111"
    answer_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    
    # Answer API parameters (optional)
    answer_temperature: Optional[float] = None
    answer_top_p: Optional[float] = None
    answer_top_k: Optional[int] = None
    answer_max_tokens: Optional[int] = None
    
    # Grading API configuration
    grading_api_base: str = "api-grader"
    grading_api_key: str = "your api key here"
    grading_model: str = "deepseek-v3-241226"
    
    # Grading API parameters (optional)
    grading_temperature: Optional[float] = None
    grading_top_p: Optional[float] = None
    grading_max_tokens: Optional[int] = None
    
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
