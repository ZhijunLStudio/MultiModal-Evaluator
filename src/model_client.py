import time
from typing import Dict, Any
import aiohttp
from abc import ABC, abstractmethod
from src.config import Config

class ModelClient(ABC):
    """Abstract base class for model API clients"""
    
    def __init__(self, config: Config):
        self.config = config
        
    @abstractmethod
    async def generate(self, session, prompt: str, image_base64: str) -> Dict[str, Any]:
        """Generate response from the model API"""
        pass

    def add_metrics(self, metrics_dict, response_time, is_success):
        """Add API call metrics"""
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

        # Update global metrics
        for key in metrics_dict:
            if key in self.api_metrics:
                metrics_dict[key] += self.api_metrics[key]

class LocalModelClient(ModelClient):
    """Client for local model API calls using LLaMA Factory API format"""
    
    async def generate(self, session, prompt: str, image_base64: str) -> Dict[str, Any]:
        """Call local LLaMA Factory API to generate response"""
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.llama_api_key}"
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
            async with session.post(f"{self.config.llama_api_base}/chat/completions", 
                                    headers=headers, 
                                    json=data,
                                    timeout=600) as response:
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
                    return {"error": "Invalid API response", "content": "", "usage": {}, "latency": end_time - start_time}
                
        except Exception as e:
            return {"error": str(e), "content": "", "usage": {}, "latency": time.time() - start_time}

class RemoteModelClient(ModelClient):
    """Client for remote API calls using OpenAI-compatible API format"""
    
    async def generate(self, session, prompt: str, image_base64: str) -> Dict[str, Any]:
        """Call remote model API to generate response"""
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.remote_api_key}"
        }
        
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_base64, "detail": "high"}}
            ]}
        ]
        
        # Start with minimal required parameters
        data = {
            "model": self.config.remote_model,
            "messages": messages,
        }
        
        # Only add generation parameters if configured to do so
        if self.config.remote_use_params:
            data.update({
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens
            })
            
            # Add top_k only for APIs that support it
            if hasattr(self.config, 'top_k'):
                data["top_k"] = self.config.top_k
        
        try:
            # Add debug logging
            print(f"Sending request to: {self.config.remote_api_base}/chat/completions")
            print(f"Request data: {data}")
            
            async with session.post(f"{self.config.remote_api_base}/chat/completions", 
                                    headers=headers, 
                                    json=data,
                                    timeout=600) as response:
                
                response_status = response.status
                try:
                    response_json = await response.json()
                    end_time = time.time()
                    
                    if response_status != 200:
                        print(f"API Error: Status {response_status}")
                        print(f"Response: {response_json}")
                        return {"error": f"API Error: {response_status}", "content": "", "usage": {}, "latency": end_time - start_time}
                    
                    if "choices" in response_json and len(response_json["choices"]) > 0:
                        result = {
                            "content": response_json["choices"][0]["message"]["content"],
                            "usage": response_json.get("usage", {}),
                            "latency": end_time - start_time
                        }
                        return result
                    else:
                        print(f"Invalid response structure: {response_json}")
                        return {"error": "Invalid API response structure", "content": "", "usage": {}, "latency": end_time - start_time}
                except Exception as e:
                    response_text = await response.text()
                    print(f"Failed to parse API response: {e}")
                    print(f"Raw response: {response_text}")
                    return {"error": f"Failed to parse API response: {str(e)}", "content": "", "usage": {}, "latency": time.time() - start_time}
                
        except Exception as e:
            print(f"API request error: {str(e)}")
            return {"error": str(e), "content": "", "usage": {}, "latency": time.time() - start_time}
