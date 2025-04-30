import time
from typing import Dict, Any
import aiohttp
from src.config import Config

class LlamaFactoryClient:
    def __init__(self, config: Config):
        self.config = config
        self.api_base = config.llama_api_base
        self.api_key = config.llama_api_key
        self.model = config.llama_model
    
    async def generate(self, session, prompt: str, image_base64: str) -> Dict[str, Any]:
        """调用LLaMA Factory API生成回答"""
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ]}
        ]
        
        data = {
            "model": self.config.llama_model,
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "max_tokens": self.config.max_tokens
        }
        
        try:
            async with session.post(f"{self.api_base}/chat/completions", 
                                    headers=headers, 
                                    json=data,
                                    timeout=60) as response:
                response_json = await response.json()
                end_time = time.time()
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    result = {
                        "content": response_json["choices"][0]["message"]["content"],
                        "usage": response_json.get("usage", {}),
                        "latency": end_time - start_time
                    }
                    return result
                else:
                    return {"error": "无效的API响应", "content": "", "usage": {}, "latency": end_time - start_time}
                
        except Exception as e:
            return {"error": str(e), "content": "", "usage": {}, "latency": time.time() - start_time}


    def add_metrics(self, metrics_dict, response_time, is_success):
        """添加API调用指标"""
        if not hasattr(self, 'api_metrics'):
            self.api_metrics = {
                'calls': 0,
                'success': 0,
                'failures': 0,
                'total_time': 0,
                'response_times': []
            }
        
        self.api_metrics['calls'] += 1
        if is_success:
            self.api_metrics['success'] += 1
        else:
            self.api_metrics['failures'] += 1
        
        self.api_metrics['total_time'] += response_time
        self.api_metrics['response_times'].append(response_time)

        # 同时更新全局指标
        for key in metrics_dict:
            if key in self.api_metrics:
                metrics_dict[key] += self.api_metrics[key]
