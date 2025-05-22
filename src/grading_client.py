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
        
        # 将结果打包成字典
        result = {
            "exact_matches": list(exact_matches),
            "exact_match_count": len(exact_matches),
            "unmatched_gen": [self._format_connection(c) for c in unmatched_gen],
            "unmatched_ref": [self._format_connection(c) for c in unmatched_ref],
            "total_gen": len(gen_connections),
            "total_ref": len(ref_connections),
            "precision": precision,
            "recall": recall
        }
        
        return result
    
    def _extract_semantic_matches(self, text: str) -> List[Dict[str, str]]:
        """Extract semantic matches from grading model output in JSON format"""
        matches = []
        
        # 调试信息
        if self.config.verbose:
            print(f"Extracting semantic matches from text: {text[:200]}...")
        
        # 1. 尝试从 ```json ... ``` 代码块提取 JSON
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        if json_matches:
            for json_text in json_matches:
                try:
                    if self.config.verbose:
                        print(f"Found JSON text: {json_text[:100]}...")
                    
                    data = json.loads(json_text)
                    
                    # 检查是否包含 semantic_matches
                    if "semantic_matches" in data and isinstance(data["semantic_matches"], list):
                        for match in data["semantic_matches"]:
                            if isinstance(match, dict) and "generated" in match and "reference" in match:
                                gen = self._clean_match_string(match["generated"])
                                ref = self._clean_match_string(match["reference"])
                                matches.append({"generated": gen, "reference": ref})
                        
                        if matches and self.config.verbose:
                            print(f"Successfully extracted {len(matches)} semantic matches from JSON")
                        
                        # 有效匹配后返回，不再尝试其他方法
                        if matches:
                            return matches
                except json.JSONDecodeError as e:
                    if self.config.verbose:
                        print(f"JSON decode error: {str(e)}")
                    continue
                except Exception as e:
                    if self.config.verbose:
                        print(f"Error processing JSON: {str(e)}")
                    continue
        
        # 2. 如果未从代码块提取成功，尝试直接解析整个文本
        try:
            if self.config.verbose:
                print("Trying to parse entire text as JSON")
            
            data = json.loads(text)
            if "semantic_matches" in data and isinstance(data["semantic_matches"], list):
                for match in data["semantic_matches"]:
                    if isinstance(match, dict) and "generated" in match and "reference" in match:
                        gen = self._clean_match_string(match["generated"])
                        ref = self._clean_match_string(match["reference"])
                        matches.append({"generated": gen, "reference": ref})
                
                if matches and self.config.verbose:
                    print(f"Successfully extracted {len(matches)} semantic matches from full text")
                
                # 有效匹配后返回
                if matches:
                    return matches
        except json.JSONDecodeError:
            if self.config.verbose:
                print("Failed to parse entire text as JSON")
        except Exception as e:
            if self.config.verbose:
                print(f"Error processing text as JSON: {str(e)}")
        
        # 3. 如果 JSON 解析都失败，回退到正则表达式方法
        if self.config.verbose:
            print("Falling back to regex pattern matching")
        
        # 主要匹配模式
        main_pattern = r'Generated:\s*([^=\n]+?)\s*=\s*Reference:\s*([^\n]+)'
        pairs = re.findall(main_pattern, text, re.IGNORECASE)
        
        for gen, ref in pairs:
            gen_clean = self._clean_match_string(gen)
            ref_clean = self._clean_match_string(ref)
            matches.append({"generated": gen_clean, "reference": ref_clean})
        
        # 备用匹配模式
        if not matches:
            alt_patterns = [
                r'`([^`]+)`\s*(?:is|are)?\s*equivalent to\s*`([^`]+)`',
                r'\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|'
            ]
            
            for pattern in alt_patterns:
                pairs = re.findall(pattern, text, re.IGNORECASE)
                for gen, ref in pairs:
                    gen_clean = self._clean_match_string(gen)
                    ref_clean = self._clean_match_string(ref)
                    matches.append({"generated": gen_clean, "reference": ref_clean})
                
                if matches:  # 找到匹配就停止尝试
                    break
        
        # 4. 去重处理
        unique_matches = []
        seen_pairs = set()
        
        for match in matches:
            pair_id = f"{match['generated']}|{match['reference']}"
            if pair_id not in seen_pairs:
                seen_pairs.add(pair_id)
                unique_matches.append(match)
        
        if self.config.verbose:
            print(f"Final extraction result: {len(unique_matches)} unique semantic matches")
        
        return unique_matches

    def _clean_match_string(self, text: str) -> str:
        """Clean match string by removing quotes, asterisks, etc."""
        # 移除引号、星号和转义字符等
        cleaned = re.sub(r'[`*"\\]', '', text.strip())
        # 移除开头结尾的空白字符
        return cleaned.strip()


    
    def _calculate_metrics(self, exact_matches: int, semantic_matches: int, 
                          total_gen: int, total_ref: int) -> Dict[str, float]:
        """Calculate evaluation metrics correctly"""
        # 确保语义匹配数量不会导致总匹配超过参考数量
        total_matches = min(exact_matches + semantic_matches, total_ref)
        
        # 计算精确率和召回率
        precision = total_matches / total_gen if total_gen > 0 else 0.0
        recall = total_matches / total_ref if total_ref > 0 else 0.0
        
        # 计算标准AP和mAP
        # AP是为每个参考连接找到匹配的概率
        # 在我们的案例中，可以简化为正确匹配的参考连接数量比例
        ap = recall  # 在这个场景中，AP等同于召回率
        
        # 确保所有指标在[0,1]范围内
        precision = min(1.0, max(0.0, precision))
        recall = min(1.0, max(0.0, recall))
        ap = min(1.0, max(0.0, ap))
        
        return {
            "precision": precision,
            "recall": recall,
            "map": ap  # 在我们的场景中，mAP就是AP
        }
    
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
        
        # 计算初始指标(仅基于精确匹配)
        initial_metrics = self._calculate_metrics(
            comparison["exact_match_count"], 0, 
            comparison["total_gen"], comparison["total_ref"]
        )
        
        # 如果没有未匹配的连接，或者参考或生成的连接为空，则不需要进行语义匹配
        if not comparison['unmatched_gen'] or not comparison['unmatched_ref']:
            return {
                "content": "No semantic matching needed - all connections exactly matched or one set is empty.",
                "score": int(min(100, initial_metrics["map"] * 100)),
                "usage": {},
                "latency": 0,
                "connection_analysis": {
                    "comparison": comparison,
                    "gen_connections": [self._format_connection(c) for c in gen_connections],
                    "ref_connections": [self._format_connection(c) for c in ref_connections],
                    "semantic_matches": [],
                    "total_matches": comparison["exact_match_count"],
                    "metrics": initial_metrics
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
                    
                    # 提取语义匹配对（已去重）
                    semantic_matches = self._extract_semantic_matches(content)
                    
                    # 计算指标（确保不会因为重复计算而超过1）
                    metrics = self._calculate_metrics(
                        comparison["exact_match_count"], 
                        len(semantic_matches),
                        comparison["total_gen"], 
                        comparison["total_ref"]
                    )
                    
                    # 计算总匹配数（确保不超过参考总数）
                    total_matches = min(
                        comparison["exact_match_count"] + len(semantic_matches),
                        comparison["total_ref"]
                    )
                    
                    # 生成分数 - 基于 mAP
                    score = int(min(100, metrics["map"] * 100))
                    
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
                            "metrics": metrics
                        }
                    }
                    return result
                else:
                    # API 错误情况 - 仅使用正则匹配结果
                    return {
                        "error": "Invalid API response", 
                        "content": "", 
                        "score": int(min(100, initial_metrics["map"] * 100)),
                        "usage": {}, 
                        "latency": end_time - start_time,
                        "connection_analysis": {
                            "comparison": comparison,
                            "gen_connections": [self._format_connection(c) for c in gen_connections],
                            "ref_connections": [self._format_connection(c) for c in ref_connections],
                            "semantic_matches": [],
                            "total_matches": comparison["exact_match_count"],
                            "metrics": initial_metrics
                        }
                    }
                
        except Exception as e:
            # 异常情况 - 仅使用正则匹配结果
            return {
                "error": str(e), 
                "content": "", 
                "score": int(min(100, initial_metrics["map"] * 100)),
                "usage": {}, 
                "latency": time.time() - start_time,
                "connection_analysis": {
                    "comparison": comparison,
                    "gen_connections": [self._format_connection(c) for c in gen_connections],
                    "ref_connections": [self._format_connection(c) for c in ref_connections],
                    "semantic_matches": [],
                    "total_matches": comparison["exact_match_count"],
                    "metrics": initial_metrics
                }
            }
    
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
