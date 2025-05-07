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
        self.results = []
        self.individual_results = {}  # 按图片ID存储单独的结果
        
        # 验证并设置要使用的prompt keys
        if not self.config.prompt_keys:
            # 默认使用除grading_prompt开头的提示之外的所有提示
            self.config.prompt_keys = [k for k in self.prompts.keys() 
                                      if not k.startswith("grading_prompt")]
        else:
            # 验证指定的prompt keys是否存在
            for key in self.config.prompt_keys:
                if key not in self.prompts:
                    raise ValueError(f"指定的prompt key '{key}'不存在")
    
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
        
        # 执行多次运行
        for run_id in range(self.config.runs_per_prompt):
            try:
                # 调用LLaMA Factory API
                llm_start = time.time()
                llm_result = await self.llama_client.generate(session, generation_prompt, image_base64)
                llm_time = time.time() - llm_start
                
                if "error" in llm_result:
                    error_result = {
                        "error": f"生成错误: {llm_result['error']}",
                        "item": item,
                        "prompt_key": prompt_key,
                        "run_id": run_id
                    }
                    results.append(error_result)
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
                
                # 整合结果
                result = {
                    "item": {
                        "img": item["img"],
                        "img_folder": item["img_folder"],
                        "tag": item.get("tag", "")
                    },
                    "prompt_key": prompt_key,
                    "prompt": generation_prompt,
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
                
                # 添加到单个图片结果字典中
                if img_key not in self.individual_results:
                    self.individual_results[img_key] = []
                self.individual_results[img_key].append(result)
                
                results.append(result)
                
                # 立即保存当前图片的结果，但不打印消息
                if self.config.save_individual:
                    # 确保输出目录存在
                    os.makedirs(self.config.output_dir, exist_ok=True)
                    
                    # 生成有效的文件名
                    safe_img_key = img_key.replace('/', '_').replace('\\', '_')
                    output_file = os.path.join(self.config.output_dir, f"{safe_img_key}.json")
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(self.individual_results[img_key], f, ensure_ascii=False, indent=2)
                    
                    # 不打印消息，避免干扰进度条
                    # print(f"已保存图片结果: {img_key}, 分数: {grading_result.get('score', 0)}")
                
            except Exception as e:
                import traceback
                error_result = {
                    "error": f"{str(e)}\n{traceback.format_exc()}",
                    "item": item,
                    "prompt_key": prompt_key,
                    "run_id": run_id
                }
                results.append(error_result)
        
        return results




    async def run(self) -> None:
        """运行评估"""
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
        
        async with aiohttp.ClientSession() as session:
            # 限制并发数量
            semaphore = asyncio.Semaphore(self.config.num_workers)
            
            # 计算正确的任务总数
            total_tasks = len(data) * len(self.config.prompt_keys) * self.config.runs_per_prompt
            
            # 创建进度条
            progress_bar = tqdm(total=total_tasks, desc="评估进度")
            
            async def process_with_semaphore(item, prompt_key):
                async with semaphore:
                    results = await self._process_item_with_multiple_runs(session, item, prompt_key)
                    
                    # 更新性能统计
                    for result in results:
                        stats["total_samples"] += 1
                        
                        if "error" not in result:
                            stats["successful_samples"] += 1
                            
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
            
            # 关键问题修复：确保创建所有任务并等待它们完成
            tasks = []
            for item in data:
                for prompt_key in self.config.prompt_keys:
                    # 每个提示词运行多次
                    for _ in range(self.config.runs_per_prompt):
                        tasks.append(process_with_semaphore(item, prompt_key))
            
            # 执行所有任务并收集结果
            all_results = await asyncio.gather(*tasks)
            
            # 将结果列表展平
            self.results = []
            for result_list in all_results:
                self.results.extend(result_list)
            
            progress_bar.close()

    

    
    def analyze_results(self) -> Dict[str, Any]:
        """分析结果"""
        # 确保我们有结果
        if not self.results:
            # 返回默认的空分析
            return {
                "stats": {
                    "total_samples": 0,
                    "valid_samples": 0,
                    "error_samples": 0,
                    "unique_images": 0,
                    "average_score": 0,
                    "median_score": 0,
                    "min_score": 0,
                    "max_score": 0,
                    "std_dev": 0
                },
                "score_distribution": {key: 0 for key in ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]},
                "prompt_stats": {}
            }
        
        # 提取所有有效的分数
        valid_results = [r for r in self.results if "error" not in r 
                        and "grading" in r and "score" in r.get("grading", {})]
        
        if not valid_results:
            # 如果没有有效结果，返回包含stats字段的错误信息
            return {
                "error": "没有有效的评分结果",
                "stats": {
                    "total_samples": len(self.results),
                    "valid_samples": 0,
                    "error_samples": len(self.results),
                    "unique_images": 0,
                    "average_score": 0,
                    "median_score": 0,
                    "min_score": 0,
                    "max_score": 0,
                    "std_dev": 0
                },
                "score_distribution": {key: 0 for key in ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90","90-100"]},
                "prompt_stats": {}
            }
        
        # 按样本和prompt分组计算平均分
        grouped_scores = {}
        for result in valid_results:
            # 使用字符串作为键，而不是元组
            item_key = f"{result['item']['img']}::{result['prompt_key']}"
            if item_key not in grouped_scores:
                grouped_scores[item_key] = []
            grouped_scores[item_key].append(result["grading"]["score"])
        
        # 计算每个组合的平均分
        average_scores = {}
        for k, v in grouped_scores.items():
            average_scores[k] = statistics.mean(v)
        
        # 提取所有平均分
        all_avg_scores = list(average_scores.values())
        
        # 按prompt分组的统计
        prompt_stats = {}
        for prompt_key in self.config.prompt_keys:
            # 找出与当前prompt_key匹配的键
            prompt_scores = [score for key, score in average_scores.items() 
                            if key.split("::")[1] == prompt_key]
            if prompt_scores:
                prompt_stats[prompt_key] = {
                    "samples": len(prompt_scores),
                    "average_score": statistics.mean(prompt_scores),
                    "median_score": statistics.median(prompt_scores),
                    "min_score": min(prompt_scores),
                    "max_score": max(prompt_scores),
                    "std_dev": statistics.stdev(prompt_scores) if len(prompt_scores) > 1 else 0
                }
        
        # 计算基本统计量
        stats = {
            "total_samples": len(self.results),
            "valid_samples": len(valid_results),
            "error_samples": len(self.results) - len(valid_results),
            "unique_images": len(set(r["item"]["img"] for r in valid_results)),
            "average_score": statistics.mean(all_avg_scores) if all_avg_scores else 0,
            "median_score": statistics.median(all_avg_scores) if all_avg_scores else 0,
            "min_score": min(all_avg_scores) if all_avg_scores else 0,
            "max_score": max(all_avg_scores) if all_avg_scores else 0,
            "std_dev": statistics.stdev(all_avg_scores) if len(all_avg_scores) > 1 else 0
        }
        
        # 计算分数分布
        bins = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
        score_distribution = {bin_range: 0 for bin_range in bins}
        
        for score in all_avg_scores:
            for i, bin_range in enumerate(bins):
                lower = i * 10
                upper = (i + 1) * 10
                if lower <= score < upper or (score == 100 and upper == 100):
                    score_distribution[bin_range] += 1
                    break
        
        # 将average_scores中的键格式化为可读形式
        formatted_scores = {}
        for key, value in average_scores.items():
            img, prompt = key.split("::")
            formatted_scores[f"{img} ({prompt})"] = value
        
        return {
            "stats": stats,
            "score_distribution": score_distribution,
            "prompt_stats": prompt_stats,
            "average_scores": formatted_scores
        }
    
    def save_results(self) -> None:
        """保存结果到JSON文件"""
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 生成分析报告
        analysis = self.analyze_results()
        
        # 准备保存数据
        output_data = {
            "config": asdict(self.config),
            "analysis": analysis,
            "results": self.results
        }
        
        # 保存总结果到文件
        summary_path = os.path.join(self.config.output_dir, self.config.summary_name)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"总结果已保存至: {summary_path}")
        
        # 保存单独的图片结果
        if self.config.save_individual:
            for img_key, results in self.individual_results.items():
                # 生成有效的文件名
                safe_img_key = img_key.replace('/', '_').replace('\\', '_')
                output_file = os.path.join(self.config.output_dir, f"{safe_img_key}.json")
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"已保存 {len(self.individual_results)} 个单独的图片结果到 {self.config.output_dir} 目录")
        
        # 打印分析摘要
        if self.config.verbose:
            print("\n=== 评估结果摘要 ===")
            print(f"总样本数: {analysis['stats']['total_samples']}")
            print(f"有效样本数: {analysis['stats']['valid_samples']}")
            print(f"唯一图片数: {analysis['stats']['unique_images']}")
            print(f"总平均分: {analysis['stats']['average_score']:.2f}")
            
            print("\n分数分布:")
            for range_key, count in analysis['score_distribution'].items():
                print(f"{range_key}: {count} 样本")
            
            print("\n按提示词统计:")
            for prompt_key, stats in analysis['prompt_stats'].items():
                print(f"{prompt_key}: 平均分 {stats['average_score']:.2f} ({stats['samples']} 样本)")
