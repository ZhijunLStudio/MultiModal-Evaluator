from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class Config:
    # 路径配置
    jsonl_path: str = "data.jsonl"
    image_root_dir: str = "images/"
    prompt_path: str = "prompts.json"
    output_dir: str = "results/"  # 输出目录
    summary_name: str = "summary.json"  # 总结果文件名
    
    # LLaMA Factory API配置
    llama_api_base: str = "http://0.0.0.0:37000/v1"
    llama_api_key: str = "111"
    llama_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    
    # 评分API配置
    grading_api_base: str = "api-grader"
    grading_api_key: str = "your api key here"
    grading_model: str = "deepseek-v3-241226"
    
    # 生成配置
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096
    
    # 评估配置
    eval_samples: int = -1  # -1表示评估所有样本
    num_workers: int = 1    # 异步并发数
    
    # prompt配置
    prompt_keys: List[str] = None  # 要使用的prompt keys列表
    runs_per_prompt: int = 1       # 每个prompt运行的次数
    grading_lang: str = "en"  # 默认使用英文
    
    # 输出配置
    verbose: bool = True
    save_individual: bool = True  # 是否保存单独的图片结果
