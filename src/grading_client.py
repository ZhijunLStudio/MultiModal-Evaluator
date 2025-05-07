import time
import re
from typing import Dict, Any
import aiohttp
from src.config import Config

class GradingClient:
    def __init__(self, config: Config):
        self.config = config
        self.api_base = config.grading_api_base
        self.api_key = config.grading_api_key
        self.model = config.grading_model
        self.prompts = {}  # 将在Evaluator中设置
    
    async def grade(self, session, prompt: str, generated_answer: str, reference_answer: str) -> Dict[str, Any]:
        """调用评分API评估生成的答案"""
        # 根据语言选择评分prompt
        grading_prompt_key = f"grading_prompt_{self.config.grading_lang}"
        
        if not self.prompts:
            return {"error": "评分提示未设置", "content": "", "score": 0, "usage": {}, "latency": 0}
        
        grading_prompt = self.prompts.get(grading_prompt_key)
        if not grading_prompt:
            return {"error": f"评分提示'{grading_prompt_key}'不存在", "content": "", "score": 0, "usage": {}, "latency": 0}
            
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.grading_api_key}"
        }
        
        # 构造评分提示
        full_prompt = f"{grading_prompt}\n\nGenerated Answer:\n{generated_answer}\n\nReference Answer:\n{reference_answer}"
        
        data = {
            "model": self.config.grading_model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.2,  # 评分应该是确定性的，使用较低的温度
            "max_tokens": 1024,
        }
        
        try:
            async with session.post(f"{self.api_base}/chat/completions", 
                                    headers=headers, 
                                    json=data,
                                    timeout=60) as response:
                response_json = await response.json()
                end_time = time.time()
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0]["message"]["content"]
                    
                    # 尝试从回复中提取分数
                    score = self._extract_score(content)
                    
                    result = {
                        "content": content,
                        "score": score,
                        "usage": response_json.get("usage", {}),
                        "latency": end_time - start_time
                    }
                    return result
                else:
                    return {"error": "无效的API响应", "content": "", "score": 0, "usage": {}, "latency": end_time - start_time}
                
        except Exception as e:
            return {"error": str(e), "content": "", "score": 0, "usage": {}, "latency": time.time() - start_time}
    
    def _extract_score(self, text: str) -> float:
        """从评分文本中提取分数"""
        try:
            # 尝试多种可能的格式提取分数
            
            # 寻找格式如 "分数：85" 或 "Score: 85" 或单独的数字
            patterns = [
                r'score\s*(?:is)?(?::|=)?\s*(\d+(?:\.\d+)?)',
                r'分数[:：]\s*(\d+(?:\.\d+)?)',
                r'评分[:：]\s*(\d+(?:\.\d+)?)',
                r'得分[:：]\s*(\d+(?:\.\d+)?)',
                r'^\s*(\d+(?:\.\d+)?)\s*$'  # 单独的数字
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            
            # 如果没有匹配到具体格式，尝试提取任意数字（可能为分数）
            numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
            if numbers and 0 <= float(numbers[0]) <= 100:
                return float(numbers[0])
                
            return 0
        except:
            return 0
