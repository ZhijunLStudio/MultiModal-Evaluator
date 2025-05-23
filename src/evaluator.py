import os
import json
import time
import statistics
import asyncio
import aiohttp
from typing import Dict, List, Any
from tqdm import tqdm
from dataclasses import asdict
from src.config import Config
from src.image_processor import ImageProcessor
from src.model_client import LocalModelClient, RemoteModelClient
from src.grading_client import GradingClient

class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.prompts = self._load_prompts()
        
        # Initialize the appropriate model client based on configuration
        if config.model_mode == "local":
            self.model_client = LocalModelClient(config)
        elif config.model_mode == "remote":
            self.model_client = RemoteModelClient(config)
        else:
            raise ValueError(f"Invalid model mode: {config.model_mode}. Must be 'local' or 'remote'")
            
        self.grading_client = GradingClient(config)
        self.grading_client.prompts = self.prompts
        self.results_stats = {  # Only save statistics, not full results
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "scores_by_prompt": {},
            "errors": [],
             # 新增：统计各种指标的累积值
            "metrics_accumulation": {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "conn_ratio": 0,
            "perfect_match": 0
            },
            # 新增：各指标的平均值
            "metrics_average": {}
        }
        
        
        self.individual_results = {}  # Temporary storage for individual image results
        self.results = []  # Keep all results
        self.file_locks = {}  # Initialize file locks dictionary
        
        # Validate and set prompt keys to use
        if not self.config.prompt_keys:
            self.config.prompt_keys = [k for k in self.prompts.keys() 
                                      if not k.startswith("grading_prompt")]
        else:
            for key in self.config.prompt_keys:
                if key not in self.prompts:
                    raise ValueError(f"Specified prompt key '{key}' does not exist")
                
        # Initialize statistics for each prompt
        for key in self.config.prompt_keys:
            self.results_stats["scores_by_prompt"][key] = []
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from file"""
        try:
            with open(self.config.prompt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to load prompts: {str(e)}")
    
    def _load_jsonl(self) -> List[Dict[str, Any]]:
        """Load JSONL data"""
        data = []
        try:
            with open(self.config.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            
            if self.config.eval_samples > 0:
                return data[:self.config.eval_samples]
            return data
        except Exception as e:
            raise Exception(f"Failed to load JSONL data: {str(e)}")
    
    async def _process_item_with_multiple_runs(self, session, item: Dict[str, Any], prompt_key: str) -> List[Dict[str, Any]]:
        """Process a single item with the specified prompt multiple times"""
        results = []
        
        # Build complete image path
        img_path = os.path.join(self.config.image_root_dir, item["img_folder"], item["img"])
        img_key = f"{item['img_folder']}/{item['img']}"
        
        # Check if file exists
        if not os.path.exists(img_path):
            error_result = {
                "error": f"Image does not exist: {img_path}",
                "item": {
                    "img": item["img"],
                    "img_folder": item["img_folder"]
                },
                "prompt_key": prompt_key,
                "run_id": 0
            }
            return [error_result]
        
        # Encode image (only need to do this once)
        try:
            image_base64 = ImageProcessor.encode_image(img_path)
            generation_prompt = self.prompts[prompt_key]
        except Exception as e:
            error_result = {
                "error": str(e),
                "item": {
                    "img": item["img"],
                    "img_folder": item["img_folder"]
                },
                "prompt_key": prompt_key,
                "run_id": 0
            }
            return [error_result]
        
        # Execute multiple runs
        run_results = []
        metrics_history = []  # 收集所有运行的指标
        
        for run_id in range(self.config.runs_per_prompt):
            try:
                # Call model API
                llm_start = time.time()
                llm_result = await self.model_client.generate(session, generation_prompt, image_base64)
                llm_time = time.time() - llm_start
                
                if "error" in llm_result:
                    # Error handling
                    error_result = {
                        "error": f"Generation error: {llm_result.get('error', '')}",
                        "item": {
                            "img": item["img"],
                            "img_folder": item["img_folder"]
                        },
                        "prompt_key": prompt_key,
                        "run_id": run_id
                    }
                    results.append(error_result)
                    self.results_stats["errors"].append(error_result)
                    
                    # Add to run results
                    run_results.append({
                        "run_id": run_id,
                        "error": llm_result.get('error', ''),
                        "timestamp": time.time()
                    })
                    continue
                
                # Call grading API
                grade_start = time.time()
                grading_result = await self.grading_client.grade(
                    session,
                    generation_prompt,
                    llm_result["content"],
                    item["answer"]
                )
                grade_time = time.time() - grade_start
                
                # Ensure results include latency time
                if "latency" not in llm_result:
                    llm_result["latency"] = llm_time
                if "latency" not in grading_result:
                    grading_result["latency"] = grade_time
                
                # Get score and add to metrics history
                score = grading_result.get("score", 0)
                metrics = grading_result.get("connection_analysis", {}).get("metrics", {})
                
                # 提取指标
                precision = metrics.get("precision", 0)
                recall = metrics.get("recall", 0)
                f1_score = metrics.get("f1", 0)
                conn_ratio = metrics.get("conn_ratio", 0)
                perfect_match = int(metrics.get("perfect_match", 0))
                
                # 保存指标历史
                metrics_history.append({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_score,
                    "conn_ratio": conn_ratio,
                    "perfect_match": perfect_match
                })
                
                for key in self.results_stats["metrics_accumulation"].keys():
                    if key in metrics:
                        self.results_stats["metrics_accumulation"][key] += metrics[key]
                
                # 更新token使用量
                if "usage" not in grading_result:
                    grading_result["usage"] = {}
                grading_result["usage"]["generation_tokens"] = llm_result.get("usage", {}).get("total_tokens", 0)
                grading_result["usage"]["total_tokens"] = (
                    grading_result["usage"].get("grading_tokens", 0) + 
                    grading_result["usage"]["generation_tokens"]
                )
                
                # Integrate results - keep full format for internal use
                full_result = {
                    "item": {
                        "img": item["img"],
                        "img_folder": item["img_folder"],
                        "tag": item.get("tag", "")
                    },
                    "prompt_key": prompt_key,
                    "run_id": run_id,
                    "generation": {
                        "content": llm_result["content"],
                        "usage": llm_result.get("usage", {}),
                        "latency": llm_result.get("latency", llm_time)
                    },
                    "grading": {
                        "content": grading_result.get("content", ""),
                        "score": score,
                        "usage": grading_result.get("usage", {}),
                        "latency": grading_result.get("latency", grade_time),
                        "connection_analysis": grading_result.get("connection_analysis", {})
                    },
                    "reference": item["answer"],
                    "timestamp": time.time(),
                    "total_processing_time": llm_time + grade_time
                }
                
                # Create simplified version of results for JSON file
                simplified_result = {
                    "run_id": run_id,
                    "generation": llm_result["content"],
                    "score": score,
                    "grading_feedback": grading_result.get("content", ""),
                    "connection_analysis": grading_result.get("connection_analysis", {
                        "gen_connections": [],
                        "ref_connections": [],
                        "comparison": {
                            "exact_match_count": 0,
                            "total_ref": 0,
                            "total_gen": 0,
                            "precision": 0,
                            "recall": 0
                        },
                        "semantic_matches": [],
                        "total_matches": 0,
                        "metrics": {
                            "precision": 0,
                            "recall": 0,
                            "f1": 0,
                            "conn_ratio": 0,
                            "perfect_match": 0
                        },
                        "final_unmatched_gen": [],
                        "final_unmatched_ref": []
                    }),
                    "latency": {
                        "generation": llm_time,
                        "grading": grade_time,
                        "total": llm_time + grade_time
                    },
                    "timestamp": time.time(),
                    "token_usage": {
                        "generation_tokens": llm_result.get("usage", {}).get("total_tokens", 0),
                        "grading_tokens": grading_result.get("usage", {}).get("grading_tokens", 0),
                        "total_tokens": (
                            llm_result.get("usage", {}).get("total_tokens", 0) + 
                            grading_result.get("usage", {}).get("grading_tokens", 0)
                        )
                    }
                }
                
                # Add to run results
                run_results.append(simplified_result)
                
                # Add to internal results list
                results.append(full_result)
                self.results.append(full_result)
                
                # Update statistics
                self.results_stats["scores_by_prompt"][prompt_key].append(score)
                
            except Exception as e:
                import traceback
                # Error handling
                error_result = {
                    "error": f"{str(e)}\n{traceback.format_exc()}",
                    "item": {
                        "img": item["img"],
                        "img_folder": item["img_folder"]
                    },
                    "prompt_key": prompt_key,
                    "run_id": run_id
                }
                results.append(error_result)
                self.results_stats["errors"].append(error_result)
                
                # Add to run results
                run_results.append({
                    "run_id": run_id,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        # Use synchronization lock to save individual image results, ensuring we don't overwrite results from other prompts
        if self.config.save_individual:
            try:
                # Ensure output directory exists
                os.makedirs(self.config.output_dir, exist_ok=True)
                
                # 提取最后一次运行的指标用于文件名
                if metrics_history:
                    last_metrics = metrics_history[-1]
                    precision = last_metrics.get("precision", 0)
                    recall = last_metrics.get("recall", 0)
                    f1_score = last_metrics.get("f1", 0)
                    conn_ratio = last_metrics.get("conn_ratio", 0)
                    perfect_match = int(last_metrics.get("perfect_match", 0))
                    
                    # 将指标值格式化为文件名（保留两位小数）
                    metrics_str = f"p_{precision:.2f}_r_{recall:.2f}_f1_{f1_score:.2f}_cr_{conn_ratio:.2f}_pm_{perfect_match}"
                else:
                    metrics_str = "p_0.00_r_0.00_f1_0.00_cr_0.00_pm_0"
                
                # 创建文件名
                img_name = item["img"]  # 只使用图像文件名，不包含文件夹
                result_file = os.path.join(self.config.output_dir, f"{img_name}_{prompt_key}_{metrics_str}.json")

                
                # Use lock to get access
                if result_file not in self.file_locks:
                    self.file_locks[result_file] = asyncio.Lock()
                
                async with self.file_locks[result_file]:
                    # Read the latest file content
                    current_img_results = {}
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, 'r', encoding='utf-8') as f:
                                current_img_results = json.load(f)
                        except Exception:
                            # If file is corrupted, create a new one
                            current_img_results = {}
                    
                    # If it's a new file, add basic information
                    if not current_img_results:
                        current_img_results = {
                            "image": {
                                "path": img_key,
                                "file": item["img"],
                                "folder": item["img_folder"],
                                "tag": item.get("tag", ""),
                                "metrics_history": metrics_history  # 存储指标历史
                            },
                            "reference": item["answer"],
                            "results": {}
                        }
                    
                    # Ensure results field exists
                    if "results" not in current_img_results:
                        current_img_results["results"] = {}
                        
                    # Add results for current prompt, preserving existing results for other prompts
                    current_img_results["results"][prompt_key] = run_results
                    current_img_results["last_updated"] = time.time()
                    
                    # Save complete results
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(current_img_results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                import traceback
                print(f"Error saving individual results: {str(e)}\n{traceback.format_exc()}")
        
        return results




    async def run(self) -> None:
        """Run evaluation - process samples sequentially"""
        data = self._load_jsonl()
        print(f"Loaded {len(data)} items for evaluation")
        print(f"Using these prompt keys: {self.config.prompt_keys}")
        print(f"Each prompt will run {self.config.runs_per_prompt} times")
        print(f"Using model mode: {self.config.model_mode}")
        
        # If individual results need to be saved, ensure output directory exists
        if self.config.save_individual and not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize performance statistics dictionary
        stats = {
            "start_time": time.time(),
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "total_inference_time": 0,
            "total_grading_time": 0,
            "scores": [],
            "recent_scores": [],
            "recent_inference_times": [],
            "recent_grading_times": []
        }
        
        # Calculate total tasks
        total_tasks = len(data) * len(self.config.prompt_keys) * self.config.runs_per_prompt
        
        # Create progress bar
        progress_bar = tqdm(total=total_tasks, desc="Evaluation Progress")
        
        async with aiohttp.ClientSession() as session:
            # Limit concurrency
            semaphore = asyncio.Semaphore(self.config.num_workers)
            
            # Flag for periodic summary updates
            last_summary_time = time.time()
            summary_interval = 300  # Update summary every 5 minutes
            
            # Process each sample sequentially
            for i, item in enumerate(data):
                # At this level we can process different prompts concurrently
                async def process_prompt(prompt_key):
                    async with semaphore:
                        results = await self._process_item_with_multiple_runs(session, item, prompt_key)
                        
                        # Update performance statistics
                        for result in results:
                            stats["total_samples"] += 1
                            
                            if "error" not in result:
                                stats["successful_samples"] += 1
                                self.results_stats["successful_samples"] += 1
                                
                                # Collect times and scores
                                inference_time = result.get("generation", {}).get("latency", 0)
                                grading_time = result.get("grading", {}).get("latency", 0)
                                score = result.get("grading", {}).get("score", 0)
                                
                                stats["total_inference_time"] += inference_time
                                stats["total_grading_time"] += grading_time
                                stats["scores"].append(score)
                                
                                # Keep recent statistics
                                stats["recent_inference_times"].append(inference_time)
                                if len(stats["recent_inference_times"]) > 5:
                                    stats["recent_inference_times"].pop(0)
                                    
                                stats["recent_grading_times"].append(grading_time)
                                if len(stats["recent_grading_times"]) > 5:
                                    stats["recent_grading_times"].pop(0)
                                    
                                stats["recent_scores"].append(score)
                                if len(stats["recent_scores"]) > 5:
                                    stats["recent_scores"].pop(0)
                            else:
                                stats["failed_samples"] += 1
                                self.results_stats["failed_samples"] += 1
                        
                        # Update progress bar
                        
                        
                        # 更新进度bar的部分
                        if self.results_stats["successful_samples"] > 0:
                            # 计算当前指标平均值
                            current_metrics = {}
                            for key, value in self.results_stats["metrics_accumulation"].items():
                                current_metrics[key] = value / self.results_stats["successful_samples"]
                            
                            # 计算其他统计信息
                            elapsed = time.time() - stats["start_time"]
                            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
                            recent_avg_score = sum(stats["recent_scores"]) / len(stats["recent_scores"]) if stats["recent_scores"] else 0
                            samples_per_second = stats["total_samples"] / elapsed if elapsed > 0 else 0
                            avg_inference = stats["total_inference_time"] / stats["successful_samples"] if stats["successful_samples"] > 0 else 0
                            avg_grading = stats["total_grading_time"] / stats["successful_samples"] if stats["successful_samples"] > 0 else 0
                            recent_avg_inference = sum(stats["recent_inference_times"]) / len(stats["recent_inference_times"]) if stats["recent_inference_times"] else 0
                            recent_avg_grading = sum(stats["recent_grading_times"]) / len(stats["recent_grading_times"]) if stats["recent_grading_times"] else 0
                            
                            progress_desc = (
                                f"Completed: {stats['total_samples']}/{total_tasks} | "
                                f"Score: {avg_score:.1f} | F1: {current_metrics.get('f1', 0):.2f} | "
                                f"P: {current_metrics.get('precision', 0):.2f} | R: {current_metrics.get('recall', 0):.2f} | "
                                f"Time: {avg_inference+avg_grading:.1f}s | {samples_per_second:.2f}/sec"
                            )
                            progress_bar.set_description(progress_desc)
                            progress_bar.update(len(results))
                        
                        return results
                
                # Create tasks for each prompt
                prompt_tasks = [process_prompt(prompt_key) for prompt_key in self.config.prompt_keys]
                
                # Run all prompt tasks for the current sample concurrently
                await asyncio.gather(*prompt_tasks)
                
                # Periodically update summary file
                current_time = time.time()
                if current_time - last_summary_time > summary_interval:
                    self._update_summary()
                    last_summary_time = current_time
                    
                # Force garbage collection every 10 samples
                if i % 10 == 0:
                    import gc
                    gc.collect()
            
            progress_bar.close()
            
        # Final summary update
        self._update_summary()
        
        
    def calculate_metrics_average(self):
        """Calculate average metrics from accumulated values"""
        successful_count = self.results_stats["successful_samples"]
        if successful_count > 0:
            self.results_stats["metrics_average"] = {
                key: round(value / successful_count, 4)
                for key, value in self.results_stats["metrics_accumulation"].items()
            }
        else:
            self.results_stats["metrics_average"] = {
                key: 0.0 for key in self.results_stats["metrics_accumulation"].keys()
            }


    def _update_summary(self):
        """Update summary file, saving only key statistics"""
        try:
            # 确保输出目录存在
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # 计算最新的指标平均值
            self.calculate_metrics_average()
            
            # 计算统计信息，为每个prompt
            prompt_stats = {}
            for prompt_key, scores in self.results_stats["scores_by_prompt"].items():
                if scores:
                    prompt_stats[prompt_key] = {
                        "samples": len(scores),
                        "average_score": statistics.mean(scores),
                        "median_score": statistics.median(scores) if len(scores) > 0 else 0,
                        "min_score": min(scores) if scores else 0,
                        "max_score": max(scores) if scores else 0,
                        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
                    }
            
            # 计算总体统计信息
            all_scores = []
            for scores in self.results_stats["scores_by_prompt"].values():
                all_scores.extend(scores)
                
            # 分数分布
            bins = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
            score_distribution = {bin_range: 0 for bin_range in bins}
            
            for score in all_scores:
                for i, bin_range in enumerate(bins):
                    lower = i * 10
                    upper = (i + 1) * 10
                    if lower <= score < upper or (score == 100 and upper == 100):
                        score_distribution[bin_range] += 1
                        break
                        
            # 整体统计
            stats = {
                "total_samples": self.results_stats["total_samples"],
                "valid_samples": self.results_stats["successful_samples"],
                "error_samples": self.results_stats["failed_samples"],
                "unique_images": len(set(r["item"]["img"] for r in self.results if "item" in r and "img" in r["item"])),
                "average_score": statistics.mean(all_scores) if all_scores else 0,
                "median_score": statistics.median(all_scores) if all_scores else 0,
                "min_score": min(all_scores) if all_scores else 0,
                "max_score": max(all_scores) if all_scores else 0,
                "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0
            }
            
            # 简化错误信息
            simplified_errors = []
            for error in self.results_stats["errors"][-20:]:  # 只保留最近的20个错误
                error_info = {
                    "prompt_key": error.get("prompt_key", "unknown"),
                    "error": error.get("error", "")[:200]  # 截断错误信息
                }
                
                # 安全获取图像信息
                if "item" in error:
                    if isinstance(error["item"], dict):
                        error_info["img"] = error["item"].get("img", "unknown")
                    else:
                        error_info["img"] = str(error["item"])
                else:
                    error_info["img"] = "unknown"
                    
                simplified_errors.append(error_info)
            
            # 准备汇总数据
            summary_data = {
                "config": {k: v for k, v in asdict(self.config).items() if k != 'grading_api_key'},
                "analysis": {
                    "stats": stats,
                    "score_distribution": score_distribution,
                    "prompt_stats": prompt_stats,
                    "recent_errors": simplified_errors,
                    # 添加平均指标
                    "metrics_average": self.results_stats["metrics_average"]
                },
                "performance": {
                    "average_tokens_per_request": sum(r.get("generation", {}).get("usage", {}).get("total_tokens", 0) for r in self.results) / len(self.results) if self.results else 0,
                    "total_tokens_used": sum(r.get("generation", {}).get("usage", {}).get("total_tokens", 0) for r in self.results),
                    "average_latency": sum(r.get("total_processing_time", 0) for r in self.results) / len(self.results) if self.results else 0,
                    "total_processing_time": sum(r.get("total_processing_time", 0) for r in self.results),
                },
                "timestamp": time.time()
            }
            
            # 保存汇总文件
            summary_path = os.path.join(self.config.output_dir, self.config.summary_name)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
                    
        except Exception as e:
            import traceback
            print(f"Error updating summary file: {str(e)}\n{traceback.format_exc()}")
            
            
    def save_metrics_summary(self):
        """Save a separate metrics summary file for visualization tools"""
        metrics_path = os.path.join(self.config.output_dir, "metrics.json")
        
        # 收集所有样本的详细指标
        sample_metrics = []
        for result in self.results:
            if "grading" in result and "connection_analysis" in result["grading"] and "metrics" in result["grading"]["connection_analysis"]:
                metrics = result["grading"]["connection_analysis"]["metrics"]
                sample_metrics.append({
                    "file": result["item"]["img"],
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1": metrics.get("f1", 0),
                    "conn_ratio": metrics.get("conn_ratio", 0),
                    "perfect_match": metrics.get("perfect_match", 0),
                    "prompt": result["prompt_key"]
                })
        
        # 生成汇总数据
        metrics_data = {
            "averages": self.results_stats["metrics_average"],
            "samples": sample_metrics,
            "prompt_metrics": {}
        }
        
        # 按提示词计算指标
        for prompt_key in self.config.prompt_keys:
            prompt_results = [r for r in sample_metrics if r["prompt"] == prompt_key]
            if prompt_results:
                metrics_data["prompt_metrics"][prompt_key] = {
                    "precision": sum(r["precision"] for r in prompt_results) / len(prompt_results),
                    "recall": sum(r["recall"] for r in prompt_results) / len(prompt_results),
                    "f1": sum(r["f1"] for r in prompt_results) / len(prompt_results),
                    "conn_ratio": sum(r["conn_ratio"] for r in prompt_results) / len(prompt_results),
                    "perfect_match": sum(r["perfect_match"] for r in prompt_results) / len(prompt_results)
                }
        
        # 保存指标文件
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, ensure_ascii=False, indent=2)
            print(f"Detailed metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"Error saving metrics summary: {str(e)}")



    
    def save_results(self):
        """Final results saving - now only updates final summary and outputs statistics"""
        # 计算最新的指标平均值
        self.calculate_metrics_average()
        
        # 更新总结文件
        self._update_summary()
        
        # 保存详细指标
        self.save_metrics_summary()
        
        # 计算实际保存的文件数
        saved_files = 0
        if os.path.exists(self.config.output_dir):
            saved_files = len([f for f in os.listdir(self.config.output_dir) 
                            if f.endswith('.json') and f != self.config.summary_name and f != 'metrics.json'])
        
        print(f"Summary results saved to: {os.path.join(self.config.output_dir, self.config.summary_name)}")
        print(f"Detailed metrics saved to: {os.path.join(self.config.output_dir, 'metrics.json')}")
        print(f"Saved {saved_files} individual image results to {self.config.output_dir} directory")
        
        # 分析总结并打印
        all_scores = []
        for scores in self.results_stats["scores_by_prompt"].values():
            all_scores.extend(scores)
                
        if self.config.verbose and all_scores:
            # 原有评估结果摘要
            print("\n" + "="*50)
            print(" "*15 + "EVALUATION RESULTS SUMMARY")
            print("="*50)
            print(f"Total samples: {self.results_stats['successful_samples'] + self.results_stats['failed_samples']}")
            print(f"Valid samples: {self.results_stats['successful_samples']}")
            print(f"Error samples: {self.results_stats['failed_samples']}")
            print(f"Overall average score: {statistics.mean(all_scores):.2f}")
            
            # 添加指标平均值输出
            print("\n" + "="*50)
            print(" "*15 + "CONNECTION METRICS SUMMARY")
            print("="*50)
            avg_metrics = self.results_stats["metrics_average"]
            print(f"Precision (P): {avg_metrics.get('precision', 0):.4f}")
            print(f"Recall (R): {avg_metrics.get('recall', 0):.4f}")
            print(f"F1 Score: {avg_metrics.get('f1', 0):.4f}")
            print(f"Connection Ratio (CR): {avg_metrics.get('conn_ratio', 0):.4f}")
            print(f"Perfect Match Rate (PM): {avg_metrics.get('perfect_match', 0):.4f} ({int(avg_metrics.get('perfect_match', 0)*100)}%)")
            
            # 指标解释
            print("\n" + "="*50)
            print(" "*15 + "METRICS EXPLANATION")
            print("="*50)
            print("P (precision): Correct connections ÷ Total generated connections")
            print("R (recall): Correct connections ÷ Total reference connections")
            print("F1: Harmonic mean of precision and recall")
            print("CR (conn_ratio): Total generated connections ÷ Total reference connections")
            print("PM (perfect_match): Percentage of samples with F1=1 (perfect match)")
            
            # 分数分布
            bins = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
            score_distribution = {bin_range: 0 for bin_range in bins}
            
            for score in all_scores:
                for i, bin_range in enumerate(bins):
                    lower = i * 10
                    upper = (i + 1) * 10
                    if lower <= score < upper or (score == 100 and upper == 100):
                        score_distribution[bin_range] += 1
                        break
            
            print("\n" + "="*50)
            print(" "*15 + "SCORE DISTRIBUTION")
            print("="*50)
            for range_key, count in score_distribution.items():
                print(f"{range_key}: {count} samples ({count/len(all_scores)*100:.1f}%)")
            
            print("\n" + "="*50)
            print(" "*15 + "PROMPT STATISTICS")
            print("="*50)
            for prompt_key, scores in self.results_stats["scores_by_prompt"].items():
                if scores:
                    avg_score = statistics.mean(scores)
                    print(f"{prompt_key}: Average score {avg_score:.2f} ({len(scores)} samples)")
            
            print("\n" + "="*50)




