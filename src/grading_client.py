import time
import re
from typing import Dict, Any, List, Tuple, Set
import aiohttp
from src.config import Config

class GradingClient:
    def __init__(self, config: Config):
        self.config = config
        self.api_base = config.grading_api_base
        self.api_key = config.grading_api_key
        self.model = config.grading_model
        self.prompts = {}  # Will be set in Evaluator
    
    def _extract_connections(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract connection relationships from text using regex.
        Returns list of tuples (source, connection_type, target)
        """
        connections = []
        
        # 提取形式为 A -> B 的连接
        arrow_pattern = re.compile(r'(\w+)\s*->\s*(\w+)(?:\s*\[.*?\])?', re.DOTALL)
        matches = arrow_pattern.findall(text)
        for src, tgt in matches:
            connections.append((src.strip(), "->", tgt.strip()))
        
        # 提取形式为 A - B 或 A — B 的连接 (包括多种连字符，且处理可能没有空格的情况)
        dash_pattern = re.compile(r'(\w+)[^\w\n]*[—\-–−﹣－‐⁃‑‒\u2010-\u2015][^\w\n]*(\w+)(?:\s*\[.*?\])?', re.DOTALL)
        matches = dash_pattern.findall(text)
        for src, tgt in matches:
            connections.append((src.strip(), "-", tgt.strip()))
        
        # 提取形式为 A <-> B 的连接
        bidirectional_pattern = re.compile(r'(\w+)\s*<->\s*(\w+)(?:\s*\[.*?\])?', re.DOTALL)
        matches = bidirectional_pattern.findall(text)
        for src, tgt in matches:
            connections.append((src.strip(), "<->", tgt.strip()))
        
        if not connections:
            # 添加调试信息
            print(f"Warning: No connections extracted from text. Sample: {text[:100]}...")
        
        return connections


    
    def _format_connection(self, conn: Tuple[str, str, str]) -> str:
        """Format connection tuple as string"""
        return f"{conn[0]} {conn[1]} {conn[2]}"
    
    def _compare_connections(self, gen_connections: List[Tuple[str, str, str]], 
                            ref_connections: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Compare generated and reference connections using regex"""
        # 创建连接的字符串表示集合，方便比较
        gen_conn_set = set(self._format_connection(c) for c in gen_connections)
        ref_conn_set = set(self._format_connection(c) for c in ref_connections)
        
        # 精确匹配的连接
        exact_matches = gen_conn_set.intersection(ref_conn_set)
        
        # 生成中未匹配的连接
        unmatched_gen = [conn for conn in gen_connections 
                        if self._format_connection(conn) not in exact_matches]
        
        # 参考中未匹配的连接
        unmatched_ref = [conn for conn in ref_connections 
                        if self._format_connection(conn) not in exact_matches]
        
        # 计算精确匹配指标
        precision = len(exact_matches) / len(gen_conn_set) if gen_conn_set else 0.0
        recall = len(exact_matches) / len(ref_conn_set) if ref_conn_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 将结果打包成字典
        result = {
            "exact_matches": list(exact_matches),
            "exact_match_count": len(exact_matches),
            "unmatched_gen": [self._format_connection(c) for c in unmatched_gen],
            "unmatched_ref": [self._format_connection(c) for c in unmatched_ref],
            "total_gen": len(gen_connections),
            "total_ref": len(ref_connections),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        return result
    
    async def grade(self, session, prompt: str, generated_answer: str, reference_answer: str) -> Dict[str, Any]:
        """Call grading API to evaluate the generated answer with focus on connection relationships"""
        # Select grading prompt based on language
        grading_prompt_key = f"grading_prompt_{self.config.grading_lang}"
        
        if not self.prompts:
            return {"error": "Grading prompts not set", "content": "", "score": 0, "usage": {}, "latency": 0}
        
        grading_prompt = self.prompts.get(grading_prompt_key)
        if not grading_prompt:
            return {"error": f"Grading prompt '{grading_prompt_key}' does not exist", "content": "", "score": 0, "usage": {}, "latency": 0}
            
        # 提取连接关系
        gen_connections = self._extract_connections(generated_answer)
        ref_connections = self._extract_connections(reference_answer)
        
        # 比较连接关系
        comparison = self._compare_connections(gen_connections, ref_connections)
        
        # 如果没有未匹配的连接，或者参考或生成的连接为空，则不需要进行语义匹配
        if not comparison['unmatched_gen'] or not comparison['unmatched_ref']:
            return {
                "content": "No semantic matching needed - all connections exactly matched or one set is empty.",
                "score": 100 if comparison["recall"] == 1.0 else int(comparison["recall"] * 100),
                "usage": {},
                "latency": 0,
                "connection_analysis": {
                    "comparison": comparison,
                    "gen_connections": [self._format_connection(c) for c in gen_connections],
                    "ref_connections": [self._format_connection(c) for c in ref_connections],
                    "semantic_matches": [],
                    "total_matches": comparison["exact_match_count"],
                    "metrics": {
                        "precision": comparison["precision"],
                        "recall": comparison["recall"],
                        "f1": comparison["f1"]
                    }
                }
            }
        
        # 构建差异信息，只发送未匹配的连接给大模型
        diff_info = (
            f"# Connection Relationship Analysis\n\n"
            f"## Statistics\n"
            f"- Total connections in reference: {comparison['total_ref']}\n"
            f"- Total connections in generated: {comparison['total_gen']}\n"
            f"- Exact regex matches: {comparison['exact_match_count']}\n\n"
        )
        
        if comparison['unmatched_ref']:
            diff_info += "## Unmatched Reference Connections\nThese connections are in the reference but not found in the generated answer:\n"
            for i, conn in enumerate(comparison['unmatched_ref']):
                diff_info += f"{i+1}. `{conn}`\n"
            diff_info += "\n"
            
        if comparison['unmatched_gen']:
            diff_info += "## Unmatched Generated Connections\nThese connections are in the generated answer but not found in the reference:\n"
            for i, conn in enumerate(comparison['unmatched_gen']):
                diff_info += f"{i+1}. `{conn}`\n"
            diff_info += "\n"
        
        diff_info += (
            "## Semantic Matching Task\n"
            "For each unmatched connection, determine if it is semantically equivalent to any connection in the opposite set, "
            "considering that different node names might refer to the same component in the diagram. "
            "Please list all semantically equivalent pairs in this format:\n"
            "Generated: `Node1 -> Node2` = Reference: `NodeA -> NodeB`\n\n"
        )
        
        start_time = time.time()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.grading_api_key}"
        }
        
        # 构建提示词，专注于未匹配连接的语义等价分析
        full_prompt = (
            f"{grading_prompt}\n\n"
            f"Generated Answer:\n{generated_answer}\n\n"
            f"Reference Answer:\n{reference_answer}\n\n"
            f"{diff_info}"
            f"Only identify semantically equivalent connections from the unmatched sets. "
            f"Focus on finding connections that refer to the same components but use different names or representations."
        )
        
        data = {
            "model": self.config.grading_model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            # "temperature": 0.2,  # Should be deterministic, use lower temperature
            # "max_tokens": 1024,
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
                    
                    # 提取语义匹配对
                    semantic_matches = self._extract_semantic_matches(content)
                    total_matches = comparison["exact_match_count"] + len(semantic_matches)
                    
                    # 计算包括语义匹配的指标
                    sem_precision = total_matches / comparison["total_gen"] if comparison["total_gen"] > 0 else 0
                    sem_recall = total_matches / comparison["total_ref"] if comparison["total_ref"] > 0 else 0
                    sem_f1 = 2 * sem_precision * sem_recall / (sem_precision + sem_recall) if (sem_precision + sem_recall) > 0 else 0
                    
                    # 生成分数 - 基于召回率
                    score = int(min(100, sem_recall * 100))
                    
                    result = {
                        "content": content,
                        "score": score,
                        "usage": response_json.get("usage", {}),
                        "latency": end_time - start_time,
                        "connection_analysis": {
                            "comparison": comparison,
                            "gen_connections": [self._format_connection(c) for c in gen_connections],
                            "ref_connections": [self._format_connection(c) for c in ref_connections],
                            "semantic_matches": semantic_matches,
                            "total_matches": total_matches,
                            "metrics": {
                                "regex_precision": comparison["precision"],
                                "regex_recall": comparison["recall"],
                                "regex_f1": comparison["f1"],
                                "semantic_precision": sem_precision,
                                "semantic_recall": sem_recall, 
                                "semantic_f1": sem_f1
                            }
                        }
                    }
                    return result
                else:
                    return {
                        "error": "Invalid API response", 
                        "content": "", 
                        "score": int(comparison["recall"] * 100),
                        "usage": {}, 
                        "latency": end_time - start_time,
                        "connection_analysis": {
                            "comparison": comparison,
                            "metrics": {
                                "precision": comparison["precision"],
                                "recall": comparison["recall"],
                                "f1": comparison["f1"]
                            }
                        }
                    }
                
        except Exception as e:
            return {
                "error": str(e), 
                "content": "", 
                "score": int(comparison["recall"] * 100), 
                "usage": {}, 
                "latency": time.time() - start_time,
                "connection_analysis": {
                    "comparison": comparison,
                    "metrics": {
                        "precision": comparison["precision"],
                        "recall": comparison["recall"],
                        "f1": comparison["f1"]
                    }
                }
            }
    
    def _extract_semantic_matches(self, text: str) -> List[Dict[str, str]]:
        """Extract semantic matches from grading model output"""
        matches = []
        
        # 尝试不同的匹配模式
        patterns = [
            # 标准格式: Generated: `A -> B` = Reference: `X -> Y`
            r'Generated:\s*`([^`]+)`\s*=\s*Reference:\s*`([^`]+)`',
            # 替代格式: `A -> B` is equivalent to `X -> Y`
            r'`([^`]+)`\s*(?:is|are)?\s*equivalent to\s*`([^`]+)`',
            # 表格形式: | `A -> B` | `X -> Y` |
            r'\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|'
        ]
        
        for pattern in patterns:
            pairs = re.findall(pattern, text, re.IGNORECASE)
            for gen, ref in pairs:
                matches.append({
                    "generated": gen.strip(),
                    "reference": ref.strip()
                })
        
        return matches
    
    def _extract_score(self, text: str) -> float:
        """Extract score from grading text"""
        try:
            # Try multiple possible formats to extract score
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
