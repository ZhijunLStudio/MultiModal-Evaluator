# MultiModal-Evaluator

多模态模型评测工具，专注于评估视觉-语言模型(VLM)在图像理解任务上的性能。该工具支持使用多种提示词异步并行处理大量图像样本，通过LLM自动评分，并提供详细的性能分析报告。支持本地模型和远程API调用两种模式。

## 🌟 主要功能

-   🚀 **并行处理**: 通过异步IO高效处理大量图像样本
-   🔄 **多次运行**: 支持使用同一提示词多次运行，增加评测稳定性
-   📊 **自动评分**: 使用大语言模型自动评估生成回答的质量
-   📈 **详细分析**: 提供分数分布、统计数据和按提示词分类的性能报告
-   💾 **灵活输出**: 保存每张图片的单独评测结果和综合分析报告
-   🔌 **双模式支持**: 支持本地模型调用和远程API调用两种模式

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

### 本地模型模式

使用本地模型（通过LLaMA Factory API）进行评估:

```bash
python main.py \
  --jsonl data/benchmark.jsonl \
  --image-root images/ \
  --prompts config/prompts.json \
  --output-dir results/local_evaluation \
  --model-mode local \
  --llama-api http://0.0.0.0:37000/v1 \
  --llama-key your-api-key \
  --llama-model Qwen/Qwen2.5-VL-72B-Instruct \
  --prompt-keys prompt1 prompt2 \
  --temperature 0.2 \
  --workers 4 \
  --grading-lang zh
```

### 远程API模式
使用远程API（如OpenAI API）进行评估:
```bash
python main.py \
  --jsonl data/benchmark.jsonl \
  --image-root images/ \
  --prompts config/prompts.json \
  --output-dir results/remote_evaluation \
  --model-mode remote \
  --remote-api https://api.openai.com/v1 \
  --remote-key your-openai-api-key \
  --remote-model o4-mini \
  --prompt-keys prompt1 \
  --workers 4 \
  --no-remote-params \
  --grading-lang zh
```

### 完整参数列表
```bash
python main.py \
  --jsonl path/to/data.jsonl \         # JSONL数据文件路径
  --image-root path/to/images \        # 图片根目录
  --prompts path/to/prompts.json \     # 提示词JSON文件
  --output-dir results/my_evaluation \ # 输出结果目录
  --summary-name final_report.json \   # 摘要文件名称
  
  # 模型模式选择
  --model-mode [local|remote] \        # 模型模式: local=本地模型, remote=远程API
  
  # 本地模型参数 (model_mode=local时使用)
  --llama-api http://your-api/v1 \     # LLaMA Factory API基础URL
  --llama-key your-api-key \           # LLaMA Factory API密钥
  --llama-model model-name \           # 模型名称
  
  # 远程API参数 (model_mode=remote时使用)
  --remote-api https://api.example.com/v1 \ # 远程API基础URL
  --remote-key your-remote-api-key \        # 远程API密钥
  --remote-model model-name \                # 远程模型名称
  --no-remote-params \                       # 不向远程API发送生成参数
  
  # 评分API参数
  --grading-api https://grading-api \  # 评分API基础URL
  --grading-key grading-api-key \      # 评分API密钥
  --grading-model grading-model \      # 评分模型名称
  
  # 生成参数
  --temperature 0.5 \                  # 温度参数
  --top-p 0.9 \                        # Top-p采样参数
  --top-k 50 \                         # Top-k采样参数
  --max-tokens 2048 \                  # 最大生成token数
  
  # 评估配置
  --samples 100 \                      # 评估样本数量
  --workers 4 \                        # 并发工作线程数
  --runs 3 \                           # 每个提示词运行次数
  --prompt-keys prompt1 prompt2 \      # 要使用的提示词列表
  --grading-lang [en|zh] \             # 评分提示词语言
  --no-individual                      # 不保存单独结果文件
```

📝 To-Do 列表
- [x] 支持远程API模型调用（如OpenAI API）
- [ ] 添加更多评分策略支持
- [ ] 实现可视化分析结果的网页界面
- [ ] 添加评分结果的人工验证工具
- [ ] 实现基准数据集对比功能
- [ ] 支持批处理模式以提高API效率
- [ ] 添加更多统计分析方法
- [ ] 支持自定义评分提示词模板