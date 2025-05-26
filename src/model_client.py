import time
from typing import Dict, Any
import aiohttp
from src.config import Config

class AnswerApiClient:
    """Client for answer generation API calls"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api_metrics = {
            'calls': 0,
            'success': 0,
            'failures': 0,
            'total_time': 0,
            'response_times': []
        }
        
    async def generate(self, session, prompt: str, image_base64: str) -> Dict[str, Any]:
        """Generate response from answer API"""
        start_time = time.time()
        
        api_base = self.config.answer_api_base
        api_key = self.config.answer_api_key
        model = self.config.answer_model
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64, "detail": "high"}}
            ]}
        ]
        
        # 准备基础参数
        data = {
            "model": model,
            "messages": messages,
        }
        
        # 添加可选生成参数 - 仅添加终端指定的参数
        if hasattr(self.config, 'answer_temperature') and self.config.answer_temperature is not None:
            data["temperature"] = self.config.answer_temperature
        if hasattr(self.config, 'answer_top_p') and self.config.answer_top_p is not None:
            data["top_p"] = self.config.answer_top_p
        if hasattr(self.config, 'answer_max_tokens') and self.config.answer_max_tokens is not None:
            data["max_tokens"] = self.config.answer_max_tokens
        if hasattr(self.config, 'answer_top_k') and self.config.answer_top_k is not None:
            data["top_k"] = self.config.answer_top_k
        
        try:
            async with session.post(f"{api_base}/chat/completions", 
                                    headers=headers, 
                                    json=data,
                                    timeout=600) as response:
                
                response_status = response.status
                if response_status != 200:
                    error_text = await response.text()
                    print(f"API Error: Status {response_status}")
                    print(f"Error details: {error_text}")
                    self.api_metrics['failures'] += 1
                    self.api_metrics['calls'] += 1
                    return {"error": f"API Error: {response_status}", "content": "", "usage": {}, "latency": time.time() - start_time}
                    
                try:
                    response_json = await response.json()
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    self.api_metrics['success'] += 1
                    self.api_metrics['calls'] += 1
                    self.api_metrics['total_time'] += response_time
                    self.api_metrics['response_times'].append(response_time)
                    
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        result = {
                            "content": response_json["choices"][0]["message"]["content"],
                            "usage": response_json.get("usage", {}),
                            "latency": response_time
                        }
                        return result
                    else:
                        print(f"Invalid response structure: {response_json}")
                        self.api_metrics['failures'] += 1
                        return {"error": "Invalid API response structure", "content": "", "usage": {}, "latency": end_time - start_time}
                except Exception as e:
                    response_text = await response.text()
                    print(f"Failed to parse API response: {e}")
                    print(f"Raw response: {response_text}")
                    self.api_metrics['failures'] += 1
                    return {"error": f"Failed to parse API response: {str(e)}", "content": "", "usage": {}, "latency": time.time() - start_time}
                
        except Exception as e:
            print(f"Answer API request error: {str(e)}")
            self.api_metrics['failures'] += 1
            self.api_metrics['calls'] += 1
            return {"error": str(e), "content": "", "usage": {}, "latency": time.time() - start_time}
            
    def add_metrics(self, metrics_dict, response_time, is_success):
        """Add API call metrics"""
        self.api_metrics['calls'] += 1
        if is_success:
            self.api_metrics['success'] += 1
        else:
            self.api_metrics['failures'] += 1
        
        self.api_metrics['total_time'] += response_time
        self.api_metrics['response_times'].append(response_time)

        # Update global metrics
        for key in metrics_dict:
            if key in self.api_metrics:
                metrics_dict[key] += self.api_metrics[key]
