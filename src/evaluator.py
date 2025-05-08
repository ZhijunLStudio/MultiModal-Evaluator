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
from src.llama_factory_client import LlamaFactoryClient
from src.grading_client import GradingClient

class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.prompts = self._load_prompts()
        self.llama_client = LlamaFactoryClient(config)
        self.grading_client = GradingClient(config)
        self.grading_client.prompts = self.prompts
        self.results_stats = {  # 只保存统计信息，不保存完整结果
            "total_samples": 0,
            "successful_samples": 0,
            "failed_samples": 0,
            "scores_by_prompt": {},
            "errors": []
        }
        self.individual_results = {}  # 临时存储单个图片的结果
        self.results = []  # 完整保留所有结果
        
        # 验证并设置要使用的prompt keys
        if not self.config.prompt_keys:
            self.config.prompt_keys = [k for k in self.prompts.keys() 
                                      if not k.startswith("grading_prompt")]
        else:
            for key in self.config.prompt_keys:
                if key not in self.prompts:
                    raise ValueError(f"指定的prompt key '{key}'不存在")
                
        # 为每个prompt初始化统计数据
        for key in self.config.prompt_keys:
            self.results_stats["scores_by_prompt"][key] = []
    
    def _load_prompts(self) -> Dict[str, str]:
        """加载提示词"""
        try:
            with open(self.config.prompt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"加载提示词失败: {str(e)}")
    
    def _load_jsonl(self) -> List[Dict[str, Any]]:
        """加载JSONL数据"""
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
            raise Exception(f"加载JSONL数据失败: {str(e)}")
    
    async def _process_item_with_multiple_runs(self, session, item: Dict[str, Any], prompt_key: str) -> List[Dict[str, Any]]:
        """对单个数据项使用指定prompt运行多次"""
        results = []
        
        # 构建图片完整路径
        img_path = os.path.join(self.config.image_root_dir, item["img_folder"], item["img"])
        img_key = f"{item['img_folder']}/{item['img']}"
        
        # 确认文件存在
        if not os.path.exists(img_path):
            error_result = {
                "error": f"图片不存在: {img_path}",
                "item": item,
                "prompt_key": prompt_key,
                "run_id": 0
            }
            return [error_result]
        
        # 编码图片 (只需要做一次)
        try:
            image_base64 = ImageProcessor.encode_image(img_path)
            generation_prompt = self.prompts[prompt_key]
        except Exception as e:
            error_result = {
                "error": str(e),
                "item": item,
                "prompt_key": prompt_key,
                "run_id": 0
            }
            return [error_result]
        
        # 初始化当前图片的结果（如果不存在）
        safe_img_key = img_key.replace('/', '_').replace('\\', '_')
        result_file = os.path.join(self.config.output_dir, f"{safe_img_key}.json")
        
        # 检查是否有现有结果文件，如果有则加载
        current_img_results = {}
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    current_img_results = json.load(f)
            except Exception:
                # 如果文件损坏，创建新的
                current_img_results = {}
        
        # 如果是新文件，添加基本信息
        if not current_img_results:
            current_img_results = {
                "image": {
                    "path": img_key,
                    "file": item["img"],
                    "folder": item["img_folder"],
                    "tag": item.get("tag", "")
                },
                "reference": item["answer"],
                "results": {}
            }
        
        # 如果这个prompt_key还没有结果，初始化它
        if prompt_key not in current_img_results["results"]:
            current_img_results["results"][prompt_key] = []
        
        # 执行多次运行
        for run_id in range(self.config.runs_per_prompt):
            try:
                # 调用LLaMA Factory API
                llm_start = time.time()
                llm_result = await self.llama_client.generate(session, generation_prompt, image_base64)
                llm_time = time.time() - llm_start
                
                if "error" in llm_result:
                    error_result = {
                        "error": f"生成错误: {llm_result.get('error', '')}",
                        "prompt_key": prompt_key,
                        "run_id": run_id
                    }
                    results.append(error_result)
                    self.results_stats["errors"].append(error_result)
                    
                    # 添加到当前图片的错误结果中
                    current_img_results["results"][prompt_key].append({
                        "run_id": run_id,
                        "error": llm_result.get('error', ''),
                        "timestamp": time.time()
                    })
                    continue
                
                # 调用评分API
                grade_start = time.time()
                grading_result = await self.grading_client.grade(
                    session,
                    generation_prompt,
                    llm_result["content"],
                    item["answer"]
                )
                grade_time = time.time() - grade_start
                
                # 确保结果包含延迟时间
                if "latency" not in llm_result:
                    llm_result["latency"] = llm_time
                if "latency" not in grading_result:
                    grading_result["latency"] = grade_time
                
                # 整合结果 - 保留内部使用的完整格式
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
                        "score": grading_result.get("score", 0),
                        "usage": grading_result.get("usage", {}),
                        "latency": grading_result.get("latency", grade_time)
                    },
                    "reference": item["answer"],
                    "timestamp": time.time(),
                    "total_processing_time": llm_time + grade_time
                }
                
                # 为JSON文件创建简化版本的结果
                simplified_result = {
                    "run_id": run_id,
                    "generation": llm_result["content"],
                    "score": grading_result.get("score", 0),
                    "grading_feedback": grading_result.get("content", ""),
                    "latency": {
                        "generation": llm_time,
                        "grading": grade_time,
                        "total": llm_time + grade_time
                    },
                    "timestamp": time.time()
                }
                
                # 添加到当前图片的结果中
                current_img_results["results"][prompt_key].append(simplified_result)
                
                # 添加到内部结果列表
                results.append(full_result)
                self.results.append(full_result)
                
                # 更新统计信息
                score = grading_result.get("score", 0)
                self.results_stats["scores_by_prompt"][prompt_key].append(score)
                
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                error_result = {
                    "error": error_msg,
                    "item": {
                        "img": item["img"],
                        "img_folder": item["img_folder"]
                    },
                    "prompt_key": prompt_key,
                    "run_id": run_id
                }
                results.append(error_result)
                self.results_stats["errors"].append(error_result)
                
                # 添加到当前图片的错误结果中
                current_img_results["results"][prompt_key].append({
                    "run_id": run_id,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        # 立即保存单个图片的结果
        if self.config.save_individual:
            try:
                # 确保输出目录存在
                os.makedirs(self.config.output_dir, exist_ok=True)
                
                # 添加最后更新时间
                current_img_results["last_updated"] = time.time()
                
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(current_img_results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"保存单独结果出错: {str(e)}")
        
        return results


    async def run(self) -> None:
        """运行评估 - 按顺序处理样本"""
        data = self._load_jsonl()
        print(f"已加载 {len(data)} 条数据进行评估")
        print(f"将使用以下prompt keys: {self.config.prompt_keys}")
        print(f"每个prompt将运行 {self.config.runs_per_prompt} 次")
        
        # 如果需要保存单独的结果，确保输出目录存在
        if self.config.save_individual and not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 初始化性能统计字典
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
        
        # 计算总任务数
        total_tasks = len(data) * len(self.config.prompt_keys) * self.config.runs_per_prompt
        
        # 创建进度条
        progress_bar = tqdm(total=total_tasks, desc="评估进度")
        
        async with aiohttp.ClientSession() as session:
            # 限制并发数量
            semaphore = asyncio.Semaphore(self.config.num_workers)
            
            # 定期更新摘要的标志
            last_summary_time = time.time()
            summary_interval = 300  # 每5分钟更新一次摘要
            
            # 按顺序处理每个样本
            for i, item in enumerate(data):
                # 在这一层可以并发处理不同的prompt
                async def process_prompt(prompt_key):
                    async with semaphore:
                        results = await self._process_item_with_multiple_runs(session, item, prompt_key)
                        
                        # 更新性能统计
                        for result in results:
                            stats["total_samples"] += 1
                            
                            if "error" not in result:
                                stats["successful_samples"] += 1
                                self.results_stats["successful_samples"] += 1
                                
                                # 收集时间和分数
                                inference_time = result.get("generation", {}).get("latency", 0)
                                grading_time = result.get("grading", {}).get("latency", 0)
                                score = result.get("grading", {}).get("score", 0)
                                
                                stats["total_inference_time"] += inference_time
                                stats["total_grading_time"] += grading_time
                                stats["scores"].append(score)
                                
                                # 保持最近统计
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
                        
                        # 更新进度条
                        elapsed = time.time() - stats["start_time"]
                        avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
                        recent_avg_score = sum(stats["recent_scores"]) / len(stats["recent_scores"]) if stats["recent_scores"] else 0
                        samples_per_second = stats["total_samples"] / elapsed if elapsed > 0 else 0
                        avg_inference = stats["total_inference_time"] / stats["successful_samples"] if stats["successful_samples"] > 0 else 0
                        avg_grading = stats["total_grading_time"] / stats["successful_samples"] if stats["successful_samples"] > 0 else 0
                        recent_avg_inference = sum(stats["recent_inference_times"]) / len(stats["recent_inference_times"]) if stats["recent_inference_times"] else 0
                        recent_avg_grading = sum(stats["recent_grading_times"]) / len(stats["recent_grading_times"]) if stats["recent_grading_times"] else 0
                        
                        progress_desc = (
                            f"已完成: {stats['total_samples']}/{total_tasks} | "
                            f"总分: {avg_score:.1f} | 最近: {recent_avg_score:.1f} | "
                            f"推理: {avg_inference:.1f}s/最近{recent_avg_inference:.1f}s | "
                            f"评分: {avg_grading:.1f}s/最近{recent_avg_grading:.1f}s | "
                            f"{samples_per_second:.2f}样本/秒"
                        )
                        progress_bar.set_description(progress_desc)
                        progress_bar.update(len(results))
                        
                        return results
                
                # 为每个prompt创建任务
                prompt_tasks = [process_prompt(prompt_key) for prompt_key in self.config.prompt_keys]
                
                # 并发执行当前样本的所有prompt任务
                await asyncio.gather(*prompt_tasks)
                
                # 定期更新摘要文件
                current_time = time.time()
                if current_time - last_summary_time > summary_interval:
                    self._update_summary()
                    last_summary_time = current_time
                    
                # 每10个样本执行一次强制垃圾回收
                if i % 10 == 0:
                    import gc
                    gc.collect()
            
            progress_bar.close()
            
        # 最后一次更新摘要
        self._update_summary()

    def _update_summary(self):
        """更新摘要文件，只保存关键统计信息"""
        try:
            # 确保输出目录存在
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # 计算各提示词的统计信息
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
            
            # 计算总体统计
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
                        
            # 总体统计信息
            stats = {
                "total_samples": self.results_stats["successful_samples"] + self.results_stats["failed_samples"],
                "valid_samples": self.results_stats["successful_samples"],
                "error_samples": self.results_stats["failed_samples"],
                "unique_images": 0,  # 这个值需要单独计算
                "average_score": statistics.mean(all_scores) if all_scores else 0,
                "median_score": statistics.median(all_scores) if all_scores else 0,
                "min_score": min(all_scores) if all_scores else 0,
                "max_score": max(all_scores) if all_scores else 0,
                "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0
            }
            
            # 简化的错误信息
            simplified_errors = []
            for error in self.results_stats["errors"][-20:]:  # 只保留最近20条错误
                simplified_errors.append({
                    "img": error["item"]["img"],
                    "prompt_key": error["prompt_key"],
                    "error": error["error"][:200]  # 截取错误信息
                })
            
            # 准备摘要数据
            summary_data = {
                "config": {k: v for k, v in asdict(self.config).items() if k != 'grading_api_key'},  # 排除敏感信息
                "analysis": {
                    "stats": stats,
                    "score_distribution": score_distribution,
                    "prompt_stats": prompt_stats,
                    "recent_errors": simplified_errors
                },
                "timestamp": time.time()
            }
            
            # 保存摘要文件
            summary_path = os.path.join(self.config.output_dir, self.config.summary_name)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            import traceback
            print(f"更新摘要文件出错: {str(e)}\n{traceback.format_exc()}")
    
    def save_results(self):
        """最终保存结果 - 现在只需要更新最终摘要"""
        self._update_summary()
        
        # 计算实际保存的文件数量
        saved_files = 0
        if os.path.exists(self.config.output_dir):
            saved_files = len([f for f in os.listdir(self.config.output_dir) 
                             if f.endswith('.json') and f != self.config.summary_name])
        
        print(f"总结果已保存至: {os.path.join(self.config.output_dir, self.config.summary_name)}")
        print(f"已保存 {saved_files} 个单独的图片结果到 {self.config.output_dir} 目录")
        
        # 分析摘要并打印
        all_scores = []
        for scores in self.results_stats["scores_by_prompt"].values():
            all_scores.extend(scores)
            
        if self.config.verbose and all_scores:
            print("\n=== 评估结果摘要 ===")
            print(f"总样本数: {self.results_stats['successful_samples'] + self.results_stats['failed_samples']}")
            print(f"有效样本数: {self.results_stats['successful_samples']}")
            print(f"错误样本数: {self.results_stats['failed_samples']}")
            print(f"总平均分: {statistics.mean(all_scores):.2f}")
            
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
            
            print("\n分数分布:")
            for range_key, count in score_distribution.items():
                print(f"{range_key}: {count} 样本")
            
            print("\n按提示词统计:")
            for prompt_key, scores in self.results_stats["scores_by_prompt"].items():
                if scores:
                    avg_score = statistics.mean(scores)
                    print(f"{prompt_key}: 平均分 {avg_score:.2f} ({len(scores)} 样本)")
