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
            "errors": []
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
        map_scores = []  # 收集所有运行的MAP分数
        
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
                
                # Get score and add to map scores list
                score = grading_result.get("score", 0)
                map_score = grading_result.get("connection_analysis", {}).get("metrics", {}).get("map", 0)
                map_scores.append(map_score)
                
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
                            "map": 0
                        }
                    }),
                    "latency": {
                        "generation": llm_time,
                        "grading": grade_time,
                        "total": llm_time + grade_time
                    },
                    "timestamp": time.time(),
                    "token_usage": {
                        "prompt_tokens": llm_result.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": llm_result.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": llm_result.get("usage", {}).get("total_tokens", 0)
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
                
                # 创建包含所有MAP分数的字符串
                map_str = "_".join([f"{int(s*100)}" for s in map_scores]) if map_scores else "0"
                map_filename_part = f"map_{map_str}"
                
                # 创建文件名
                img_name = item["img"]  # 只使用图像文件名，不包含文件夹
                result_file = os.path.join(self.config.output_dir, f"{img_name}_{prompt_key}_{map_filename_part}.json")

                
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
                                "map_scores": map_scores  # 存储MAP分数而非原始分数
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
                            f"Avg Score: {avg_score:.1f} | Recent: {recent_avg_score:.1f} | "
                            f"Inference: {avg_inference:.1f}s/recent{recent_avg_inference:.1f}s | "
                            f"Grading: {avg_grading:.1f}s/recent{recent_avg_grading:.1f}s | "
                            f"{samples_per_second:.2f} samples/sec"
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

    def _update_summary(self):
        """Update summary file, saving only key statistics"""
        try:
            # Ensure output directory exists
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Calculate statistics for each prompt
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
            
            # Calculate overall statistics
            all_scores = []
            for scores in self.results_stats["scores_by_prompt"].values():
                all_scores.extend(scores)
                
            # Score distribution
            bins = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
            score_distribution = {bin_range: 0 for bin_range in bins}
            
            for score in all_scores:
                for i, bin_range in enumerate(bins):
                    lower = i * 10
                    upper = (i + 1) * 10
                    if lower <= score < upper or (score == 100 and upper == 100):
                        score_distribution[bin_range] += 1
                        break
                        
            # Overall statistics
            stats = {
                "total_samples": self.results_stats["successful_samples"] + self.results_stats["failed_samples"],
                "valid_samples": self.results_stats["successful_samples"],
                "error_samples": self.results_stats["failed_samples"],
                "unique_images": 0,  # This value needs to be calculated separately
                "average_score": statistics.mean(all_scores) if all_scores else 0,
                "median_score": statistics.median(all_scores) if all_scores else 0,
                "min_score": min(all_scores) if all_scores else 0,
                "max_score": max(all_scores) if all_scores else 0,
                "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0
            }
            
            # Simplified error information - safely handle different error structures
            simplified_errors = []
            for error in self.results_stats["errors"][-20:]:  # Only keep the most recent 20 errors
                error_info = {
                    "prompt_key": error.get("prompt_key", "unknown"),
                    "error": error.get("error", "")[:200]  # Truncate error message
                }
                
                # Safely get image information
                if "item" in error:
                    if isinstance(error["item"], dict):
                        error_info["img"] = error["item"].get("img", "unknown")
                    else:
                        error_info["img"] = str(error["item"])
                else:
                    error_info["img"] = "unknown"
                    
                simplified_errors.append(error_info)
            
            # Prepare summary data
            summary_data = {
                "config": {k: v for k, v in asdict(self.config).items() if k != 'grading_api_key'},  # Exclude sensitive information
                "analysis": {
                    "stats": stats,
                    "score_distribution": score_distribution,
                    "prompt_stats": prompt_stats,
                    "recent_errors": simplified_errors
                },
                "performance": {
                    "average_tokens_per_request": sum(r.get("generation", {}).get("usage", {}).get("total_tokens", 0) for r in self.results) / len(self.results) if self.results else 0,
                    "total_tokens_used": sum(r.get("generation", {}).get("usage", {}).get("total_tokens", 0) for r in self.results),
                    "average_latency": sum(r.get("total_processing_time", 0) for r in self.results) / len(self.results) if self.results else 0,
                    "total_processing_time": sum(r.get("total_processing_time", 0) for r in self.results),
                },
                "timestamp": time.time()
            }
            
            # Save summary file
            summary_path = os.path.join(self.config.output_dir, self.config.summary_name)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            import traceback
            print(f"Error updating summary file: {str(e)}\n{traceback.format_exc()}")
    
    def save_results(self):
        """Final results saving - now only updates final summary"""
        self._update_summary()
        
        # Calculate actual number of saved files
        saved_files = 0
        if os.path.exists(self.config.output_dir):
            saved_files = len([f for f in os.listdir(self.config.output_dir) 
                             if f.endswith('.json') and f != self.config.summary_name])
        
        print(f"Summary results saved to: {os.path.join(self.config.output_dir, self.config.summary_name)}")
        print(f"Saved {saved_files} individual image results to {self.config.output_dir} directory")
        
        # Analyze summary and print
        all_scores = []
        for scores in self.results_stats["scores_by_prompt"].values():
            all_scores.extend(scores)
            
        if self.config.verbose and all_scores:
            print("\n=== Evaluation Results Summary ===")
            print(f"Total samples: {self.results_stats['successful_samples'] + self.results_stats['failed_samples']}")
            print(f"Valid samples: {self.results_stats['successful_samples']}")
            print(f"Error samples: {self.results_stats['failed_samples']}")
            print(f"Overall average score: {statistics.mean(all_scores):.2f}")
            
            # Score distribution
            bins = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
            score_distribution = {bin_range: 0 for bin_range in bins}
            
            for score in all_scores:
                for i, bin_range in enumerate(bins):
                    lower = i * 10
                    upper = (i + 1) * 10
                    if lower <= score < upper or (score == 100 and upper == 100):
                        score_distribution[bin_range] += 1
                        break
            
            print("\nScore distribution:")
            for range_key, count in score_distribution.items():
                print(f"{range_key}: {count} samples")
            
            print("\nStatistics by prompt:")
            for prompt_key, scores in self.results_stats["scores_by_prompt"].items():
                if scores:
                    avg_score = statistics.mean(scores)
                    print(f"{prompt_key}: Average score {avg_score:.2f} ({len(scores)} samples)")
