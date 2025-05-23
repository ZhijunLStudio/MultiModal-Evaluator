import time
import re
from typing import Dict, Any, List, Tuple, Set
import aiohttp
from src.config import Config
import json

class GradingClient:
    def __init__(self, config: Config):
        self.config = config
        self.api_base = config.grading_api_base
        self.api_key = config.grading_api_key
        self.model = config.grading_model
        self.prompts = {}  # Will be set in Evaluator
    


    def _extract_node_name(self, node_text: str) -> str:
        """Extract only the node name part from text that may include attributes."""
        # 提取节点名称，忽略可能存在的属性标签
        bracket_index = node_text.find('[')
        if bracket_index != -1:
            return node_text[:bracket_index].strip()
        return node_text.strip()

    def _clean_node_names(self, node_name: str) -> str:
        """Clean node names by removing brackets, quotes, etc."""
        if not isinstance(node_name, str):
            return str(node_name)
        
        # 首先提取纯节点名称，去除属性标签
        clean_name = self._extract_node_name(node_name)
        
        # 去除分号和其他潜在的分隔符
        clean_name = clean_name.rstrip(';').strip()
        
        # 移除引号
        if (clean_name.startswith('"') and clean_name.endswith('"')) or \
        (clean_name.startswith("'") and clean_name.endswith("'")):
            clean_name = clean_name[1:-1]
        
        return clean_name.strip()

    def _extract_connections(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract connection relationships from text using regex.
        Returns list of tuples (source, connection_type, target)
        """
        connections = []
        
        # 按行拆分文本，方便处理
        lines = text.split('\n')
        
        for line in lines:
            # 提取原始连接关系
            
            # 1. 提取箭头连接 (->)，使用更精确的模式来匹配节点名称
            arrow_pattern = re.compile(r'(\w+(?:[-_]\w+)*)\s*->\s*(\w+(?:[-_]\w+)*(?:\[.*?\])?)')
            arrow_matches = re.findall(arrow_pattern, line)
            
            for src, tgt in arrow_matches:
                # 清理目标节点，移除可能的属性标签
                clean_tgt = self._extract_node_name(tgt)
                if src and clean_tgt:  # 确保都不为空
                    connections.append((src.strip(), "->", clean_tgt.strip()))
            
            # 2. 提取双向连接 (<->)
            bidirectional_pattern = re.compile(r'(\w+(?:[-_]\w+)*)\s*<->\s*(\w+(?:[-_]\w+)*(?:\[.*?\])?)')
            bidir_matches = re.findall(bidirectional_pattern, line)
            
            for src, tgt in bidir_matches:
                clean_tgt = self._extract_node_name(tgt)
                if src and clean_tgt:
                    connections.append((src.strip(), "<->", clean_tgt.strip()))
            
            # 3. 提取双连字符连接 (--)
            double_dash_pattern = re.compile(r'(\w+(?:[-_]\w+)*)\s*--\s*(\w+(?:[-_]\w+)*(?:\[.*?\])?)')
            double_dash_matches = re.findall(double_dash_pattern, line)
            
            for src, tgt in double_dash_matches:
                clean_tgt = self._extract_node_name(tgt)
                if src and clean_tgt:
                    connections.append((src.strip(), "--", clean_tgt.strip()))
        
        # 移除重复连接
        unique_connections = []
        seen = set()
        
        for src, conn_type, tgt in connections:
            # 去除末尾可能的分号
            src = src.rstrip(';')
            tgt = tgt.rstrip(';')
            
            connection_key = f"{src}|{conn_type}|{tgt}"
            if connection_key not in seen:
                seen.add(connection_key)
                unique_connections.append((src, conn_type, tgt))
        
        return unique_connections





    
    def _format_connection(self, conn: Tuple[str, str, str]) -> str:
        """Format connection tuple as string, preserving connection type"""
        src = self._clean_node_names(conn[0])
        tgt = self._clean_node_names(conn[2])
        return f"{src} {conn[1]} {tgt}"


    
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
        
        # 1. 尝试从 ```json ... ``` 代码块提取 JSON
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        if json_matches:
            for json_text in json_matches:
                try:
                    # if self.config.verbose:
                    #     print(f"Found JSON text: {json_text[:100]}...")
                    
                    data = json.loads(json_text)
                    
                    # 检查是否包含 semantic_matches
                    if "semantic_matches" in data and isinstance(data["semantic_matches"], list):
                        for match in data["semantic_matches"]:
                            if isinstance(match, dict) and "generated" in match and "reference" in match:
                                gen = self._clean_match_string(match["generated"])
                                ref = self._clean_match_string(match["reference"])
                                matches.append({"generated": gen, "reference": ref})
                        
                        # if matches and self.config.verbose:
                        #     print(f"Successfully extracted {len(matches)} semantic matches from JSON")
                        
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
            
            data = json.loads(text)
            if "semantic_matches" in data and isinstance(data["semantic_matches"], list):
                for match in data["semantic_matches"]:
                    if isinstance(match, dict) and "generated" in match and "reference" in match:
                        gen = self._clean_match_string(match["generated"])
                        ref = self._clean_match_string(match["reference"])
                        matches.append({"generated": gen, "reference": ref})
                
                # if matches and self.config.verbose:
                #     print(f"Successfully extracted {len(matches)} semantic matches from full text")
                
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
        """Calculate evaluation metrics with F1 score and other ratios"""
        # 确保总匹配数不会超过参考数量
        total_matches = min(exact_matches + semantic_matches, total_ref)
        
        # 计算精确率和召回率
        precision = total_matches / total_gen if total_gen > 0 else 0.0
        recall = total_matches / total_ref if total_ref > 0 else 0.0
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 连接完整性比率 - 生成的连接总数 ÷ 参考的连接总数
        conn_ratio = total_gen / total_ref if total_ref > 0 else 0.0
        
        # 全对率 (F1为1的情况)
        perfect_match = 1.0 if abs(f1 - 1.0) < 0.001 else 0.0
        
        # 确保结果在[0,1]范围内，并保留两位小数
        precision = round(min(1.0, max(0.0, precision)), 2)
        recall = round(min(1.0, max(0.0, recall)), 2)
        f1 = round(min(1.0, max(0.0, f1)), 2)
        conn_ratio = round(conn_ratio, 2)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conn_ratio": conn_ratio,
            "perfect_match": perfect_match
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
                "score": int(min(100, initial_metrics["f1"] * 100)),  # 使用F1作为分数基础
                "usage": {
                    "generation_tokens": 0,
                    "grading_tokens": 0,
                    "total_tokens": 0
                },
                "latency": 0,
                "connection_analysis": {
                    "comparison": comparison,
                    "gen_connections": [self._format_connection(c) for c in gen_connections],
                    "ref_connections": [self._format_connection(c) for c in ref_connections],
                    "semantic_matches": [],
                    "total_matches": comparison["exact_match_count"],
                    "metrics": initial_metrics,
                    "final_unmatched_gen": comparison["unmatched_gen"],
                    "final_unmatched_ref": comparison["unmatched_ref"]
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
        }
        
        try:
            async with session.post(f"{self.api_base}/chat/completions", 
                                    headers=headers, 
                                    json=data,
                                    timeout=600) as response:
                response_json = await response.json()
                end_time = time.time()
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0]["message"]["content"]
                    
                    # 提取语义匹配对
                    semantic_matches = self._extract_semantic_matches(content)
                    
                    # 将语义匹配转换为格式化连接，方便比较
                    semantic_gen_matches = set()
                    semantic_ref_matches = set()
                    for match in semantic_matches:
                        gen_clean = self._clean_match_string(match["generated"])
                        ref_clean = self._clean_match_string(match["reference"])
                        semantic_gen_matches.add(gen_clean)
                        semantic_ref_matches.add(ref_clean)
                    
                    # 计算最终未匹配的连接
                    # 先获取精确匹配的连接集合
                    exact_matches_set = set(comparison["exact_matches"])
                    
                    # 计算经过语义匹配后仍未匹配的连接
                    final_unmatched_gen = [conn for conn in comparison["unmatched_gen"] 
                                        if conn not in semantic_gen_matches]
                    final_unmatched_ref = [conn for conn in comparison["unmatched_ref"] 
                                        if conn not in semantic_ref_matches]
                    
                    # 计算总匹配数（确保不超过参考总数）
                    total_matches = min(
                        comparison["exact_match_count"] + len(semantic_matches),
                        comparison["total_ref"]
                    )
                    
                    # 计算新指标
                    metrics = self._calculate_metrics(
                        comparison["exact_match_count"], 
                        len(semantic_matches),
                        comparison["total_gen"], 
                        comparison["total_ref"]
                    )
                    
                    # 生成分数 - 基于 F1
                    score = int(min(100, metrics["f1"] * 100))
                    
                    result = {
                        "content": content,
                        "score": score,
                        "usage": {
                            "generation_tokens": 0,  # 在外部填充
                            "grading_tokens": response_json.get("usage", {}).get("total_tokens", 0),
                            "total_tokens": response_json.get("usage", {}).get("total_tokens", 0)
                        },
                        "latency": end_time - start_time,
                        "connection_analysis": {
                            "comparison": comparison,
                            "gen_connections": [self._format_connection(c) for c in gen_connections],
                            "ref_connections": [self._format_connection(c) for c in ref_connections],
                            "semantic_matches": semantic_matches,
                            "total_matches": total_matches,
                            "metrics": metrics,
                            "final_unmatched_gen": final_unmatched_gen,
                            "final_unmatched_ref": final_unmatched_ref
                        }
                    }
                    return result
                else:
                    # API 错误情况 - 仅使用正则匹配结果
                    return {
                        "error": "Invalid API response", 
                        "content": "", 
                        "score": int(min(100, initial_metrics["f1"] * 100)),
                        "usage": {
                            "generation_tokens": 0,
                            "grading_tokens": 0,
                            "total_tokens": 0
                        }, 
                        "latency": end_time - start_time,
                        "connection_analysis": {
                            "comparison": comparison,
                            "gen_connections": [self._format_connection(c) for c in gen_connections],
                            "ref_connections": [self._format_connection(c) for c in ref_connections],
                            "semantic_matches": [],
                            "total_matches": comparison["exact_match_count"],
                            "metrics": initial_metrics,
                            "final_unmatched_gen": comparison["unmatched_gen"],
                            "final_unmatched_ref": comparison["unmatched_ref"]
                        }
                    }
                
        except Exception as e:
            # 异常情况 - 仅使用正则匹配结果
            return {
                "error": str(e), 
                "content": "", 
                "score": int(min(100, initial_metrics["f1"] * 100)),
                "usage": {
                    "generation_tokens": 0,
                    "grading_tokens": 0,
                    "total_tokens": 0
                },
                "latency": time.time() - start_time,
                "connection_analysis": {
                    "comparison": comparison,
                    "gen_connections": [self._format_connection(c) for c in gen_connections],
                    "ref_connections": [self._format_connection(c) for c in ref_connections],
                    "semantic_matches": [],
                    "total_matches": comparison["exact_match_count"],
                    "metrics": initial_metrics,
                    "final_unmatched_gen": comparison["unmatched_gen"],
                    "final_unmatched_ref": comparison["unmatched_ref"]
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
