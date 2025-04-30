# MultiModal-Evaluator

一个强大的多模态模型评测工具，专注于评估视觉-语言模型(VLM)在图像理解任务上的性能。该工具支持使用多种提示词异步并行处理大量图像样本，通过LLM自动评分，并提供详细的性能分析报告。

## 🌟 主要功能

-   🚀 **并行处理**: 通过异步IO高效处理大量图像样本
-   🔄 **多次运行**: 支持使用同一提示词多次运行，增加评测稳定性
-   📊 **自动评分**: 使用大语言模型自动评估生成回答的质量
-   📈 **详细分析**: 提供分数分布、统计数据和按提示词分类的性能报告
-   💾 **灵活输出**: 保存每张图片的单独评测结果和综合分析报告

## 📋 数据准备

1.  **JSONL数据文件**

    准备一个JSONL格式的数据文件，每行包含一个独立的JSON对象。JSONL文件中的每条记录必须至少包含以下字段:

    ```json
    {"img": "generate_single_flowcharts_0422_000001.png", "img_folder": "flowchart_images", "answer": "maingraph\n  prajnelic", "tag": "flowchart"}
    {"img": "generate_single_flowcharts_0422_000002.png", "img_folder": "flowchart_images", "answer": "digraph G {\n A -> B -> C;\n A -> D -> C;\n}", "tag": "flowchart"}
    {"img": "system_diagram_0422_000015.png", "img_folder": "system_diagrams", "answer": "用户 -> 前端系统 -> API网关 -> 微服务架构 -> 数据库", "tag": "system"}
    ```

    各字段说明:
    -   `img`: 图片文件名（必填）
    -   `img_folder`: 图片所在子文件夹，相对于`image_root`参数指定的根目录（必填）
    -   `answer`: 参考答案或标准答案文本，用于评分比较（必填）
    -   `tag`: 分类标签，用于结果分析和分组（可选）

2.  **图片文件组织**

    将所有图片按照以下结构组织:

    ```text
    image_root/
      ├── folder1/
      │   ├── image1.jpg
      │   └── image2.jpg
      ├── folder2/
      │   ├── image3.jpg
      │   └── image4.jpg
      └── ...
    ```

3.  **提示词文件**

    创建一个JSON格式的提示词文件，包含用于图像理解的提示词和评分提示词:

    ```json
    {
        "prompt1": "请描述这张图片中的内容，特别关注...",
        "prompt2": "分析这张图片中的对象关系，包括...",
        "grading_prompt_en": "Evaluate the generated answer against the reference...",
        "grading_prompt_zh": "请评估生成的回答与参考答案的一致性..."
    }
    ```

## 🚀 使用方法

### 基本用法

```bash
python evaluator.py \
  --jsonl path/to/data.jsonl \
  --image-root path/to/images \
  --prompts path/to/prompts.json \
  --output-dir results/my_evaluation \
  --prompt-keys prompt1 prompt2 \
  --grading-lang zh
```

### 高级参数

```bash
python evaluator.py \
  --jsonl path/to/data.jsonl \
  --image-root path/to/images \
  --prompts path/to/prompts.json \
  --output-dir results/my_evaluation \
  --summary-name final_report.json \
  --llama-api http://your-api-endpoint/v1 \
  --llama-key your-api-key \
  --llama-model your-model-name \
  --grading-api https://grading-api-endpoint \
  --grading-key grading-api-key \
  --grading-model grading-model-name \
  --temperature 0.5 \
  --top-p 0.9 \
  --max-tokens 2048 \
  --samples 100 \
  --workers 4 \
  --runs 3 \
  --prompt-keys prompt1 prompt2 \
  --grading-lang zh \
  --no-individual
```

## 📝 To-Do 列表

- [ ] 添加更多评分策略支持
- [ ] 实现可视化分析结果的网页界面
- [ ] 支持更多模型API格式
- [ ] 添加评分结果的人工验证工具
- [ ] 实现基准数据集对比功能
- [ ] 支持批处理模式以提高API效率
- [ ] 添加更多统计分析方法
- [ ] 支持自定义评分提示词模板