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
        self.prompts = {}  # Will be set in Evaluator
    
    async def grade(self, session, prompt: str, generated_answer: str, reference_answer: str) -> Dict[str, Any]:
        """Call grading API to evaluate the generated answer"""
        # Select grading prompt based on language
        grading_prompt_key = f"grading_prompt_{self.config.grading_lang}"
        
        if not self.prompts:
            return {"error": "Grading prompts not set", "content": "", "score": 0, "usage": {}, "latency": 0}
        
        grading_prompt = self.prompts.get(grading_prompt_key)
        if not grading_prompt:
            return {"error": f"Grading prompt '{grading_prompt_key}' does not exist", "content": "", "score": 0, "usage": {}, "latency": 0}
            
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.grading_api_key}"
        }
        
        # Construct grading prompt
        full_prompt = f"{grading_prompt}\n\nGenerated Answer:\n{generated_answer}\n\nReference Answer:\n{reference_answer}"
        
        data = {
            "model": self.config.grading_model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.2,  # Grading should be deterministic, use lower temperature
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
                    
                    # Try to extract score from the response
                    score = self._extract_score(content)
                    
                    result = {
                        "content": content,
                        "score": score,
                        "usage": response_json.get("usage", {}),
                        "latency": end_time - start_time
                    }
                    return result
                else:
                    return {"error": "Invalid API response", "content": "", "score": 0, "usage": {}, "latency": end_time - start_time}
                
        except Exception as e:
            return {"error": str(e), "content": "", "score": 0, "usage": {}, "latency": time.time() - start_time}
    
    def _extract_score(self, text: str) -> float:
        """Extract score from grading text"""
        try:
            # Try multiple possible formats to extract score
            
            # Look for formats like "Score: 85" or "分数：85" or standalone numbers
            patterns = [
                r'score\s*(?:is)?(?::|=)?\s*(\d+(?:\.\d+)?)',
                r'分数[:：]\s*(\d+(?:\.\d+)?)',
                r'评分[:：]\s*(\d+(?:\.\d+)?)',
                r'得分[:：]\s*(\d+(?:\.\d+)?)',
                r'^\s*(\d+(?:\.\d+)?)\s*$'  # Standalone number
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return float(match.group(1))
            
            # If no specific format matches, try to extract any number (may be a score)
            numbers = re.findall(r'(\d+(?:\.\d+)?)', text)
            if numbers and 0 <= float(numbers[0]) <= 100:
                return float(numbers[0])
                
            return 0
        except:
            return 0
