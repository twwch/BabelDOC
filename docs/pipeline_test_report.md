# 多阶段翻译流水线 - 测试流程报告

## 1. 功能概述

根据需求文档 `prd/20251126-v1.md`，实现了多阶段翻译流水线，包含以下功能：

### 1.1 翻译流程

```
PDF输入 → PDF解析 → IL创建 → 段落提取
    ↓
┌─────────────────────────────────────────────────────┐
│  Stage 1: Translator (多模型并行翻译)                │
│  - 支持配置多个翻译模型                              │
│  - 并行调用所有翻译模型                              │
│  - 记录每个模型的翻译结果和token消耗                  │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  Stage 2: Polisher (多模型润色)                      │
│  - 对每个翻译结果进行润色优化                         │
│  - 支持配置多个润色模型                              │
│  - 记录润色过程数据                                  │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  Stage 3: Evaluation (多模型评估)                    │
│  - 五维度评估: Accuracy, Fluency, Consistency,       │
│    Terminology, Completeness                         │
│  - 10分制评分                                        │
│  - 支持配置多个评估模型                              │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  Stage 4: Assembly (组装)                            │
│  - 根据评估分数选择最佳翻译结果                       │
│  - 取平均分最高模型的翻译结果                         │
│  - 生成最终PDF输出                                   │
└─────────────────────────────────────────────────────┘
    ↓
输出: 译文PDF + 对照PDF + 过程数据JSON + 评估报告
```

### 1.2 模型配置格式

```json
{
    "model_name": "gpt-4o-mini",
    "model_type": "Translator | Polisher | Evaluation",
    "config": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-xxx"
    }
}
```

### 1.3 输出文件

| 文件 | 说明 |
|-----|------|
| `*_mono.pdf` | 纯译文PDF |
| `*_dual.pdf` | 原文+译文对照PDF |
| `pipeline_process_data.json` | 完整过程数据 |
| `evaluation_report.pdf` | 评估报告(PDF格式) |

### 1.4 PDF评估报告格式

评估报告PDF包含：
- **第一页**: 汇总表格
  - 每个评估模型对各翻译/润色模型的5维度评分
  - 维度: Accuracy, Fluency, Consistency, Terminology, Completeness
  - 各维度平均分及总平均分
  - 模型总排名
- **后续页面**: 详细评估结果
  - 每个段落的原文和翻译结果
  - 每个段落的评估分数详情

---

## 2. 代码结构

```
babeldoc/translator/pipeline/
├── __init__.py           # 模块导出
├── models.py             # 数据模型定义
│   ├── ModelConfig       # 模型配置
│   ├── PipelineConfig    # 流水线配置
│   ├── TranslationResult # 翻译结果
│   ├── PolishResult      # 润色结果
│   ├── EvaluationScores  # 评估分数
│   ├── EvaluationResult  # 评估结果
│   ├── ParagraphProcessData  # 段落过程数据
│   └── PipelineProcessData   # 完整过程数据
├── translator.py         # PipelineTranslator实现
│   ├── ModelClient       # OpenAI兼容客户端
│   └── PipelineTranslator # 主翻译器(继承BaseTranslator)
└── report_generator.py   # PDF评估报告生成器
    ├── EvaluationReportGenerator  # 报告生成器类
    └── generate_evaluation_report_pdf  # 快捷生成函数
```

---

## 3. 集成方式

`PipelineTranslator` 继承自 `BaseTranslator`，可直接替换 `OpenAITranslator`：

```python
from babeldoc.format.pdf.high_level import translate
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.translator.pipeline import (
    ModelConfig, ModelType, PipelineConfig, PipelineTranslator
)

# 1. 创建流水线配置
pipeline_config = PipelineConfig(
    translators=[
        ModelConfig(
            model_name="gpt-4o-mini",
            model_type=ModelType.TRANSLATOR,
            base_url="https://api.openai.com/v1",
            api_key="sk-xxx",
        ),
        ModelConfig(
            model_name="deepseek-chat",
            model_type=ModelType.TRANSLATOR,
            base_url="https://api.deepseek.com/v1",
            api_key="sk-xxx",
        ),
    ],
    polishers=[...],
    evaluators=[...],
    lang_in="en",
    lang_out="zh",
)

# 2. 创建PipelineTranslator
pipeline_translator = PipelineTranslator(
    pipeline_config=pipeline_config,
    output_dir="./output",
)

# 3. 使用TranslationConfig
translation_config = TranslationConfig(
    input_file="input.pdf",
    translator=pipeline_translator,  # 关键：使用pipeline translator
    lang_in="en",
    lang_out="zh",
    output_dir="./output",
)

# 4. 运行翻译
result = translate(translation_config)

# 5. 保存过程数据和报告
pipeline_translator.save_process_data("./output")
pipeline_translator.generate_report("./output")
```

---

## 4. 测试用例

### 4.1 单元测试

| 测试类 | 测试内容 |
|-------|---------|
| `TestModelConfig` | 模型配置序列化/反序列化 |
| `TestPipelineConfig` | 流水线配置序列化/反序列化、JSON文件加载 |
| `TestTranslationResult` | 翻译结果数据结构 |
| `TestEvaluationScores` | 评估分数计算（平均分） |
| `TestParagraphProcessData` | 段落过程数据 |
| `TestPipelineProcessData` | 完整过程数据、最佳模型选择 |
| `TestPipelineTranslatorMocked` | PipelineTranslator（Mock API） |
| `TestPipelineTranslatorIntegration` | BaseTranslator接口兼容性 |

### 4.2 运行测试

```bash
# 运行所有pipeline测试
uv run pytest tests/test_pipeline.py -v

# 运行特定测试
uv run pytest tests/test_pipeline.py::TestPipelineTranslatorMocked -v

# 带覆盖率
uv run pytest tests/test_pipeline.py -v --cov=babeldoc.translator.pipeline
```

---

## 5. 使用示例

### 5.1 命令行示例

```bash
# 设置API密钥
export OPENAI_API_KEY=sk-xxx
export DEEPSEEK_API_KEY=sk-xxx

# 运行示例脚本
uv run python examples/run_pipeline_example.py --file input.pdf --output ./output
```

### 5.2 配置文件示例

见 `examples/pipeline_config.json`：

```json
{
    "translators": [
        {
            "model_name": "gpt-4o-mini",
            "model_type": "Translator",
            "config": {
                "base_url": "https://api.openai.com/v1",
                "api_key": "YOUR_API_KEY"
            }
        }
    ],
    "polishers": [...],
    "evaluators": [...],
    "lang_in": "en",
    "lang_out": "zh",
    "qps": 4,
    "max_workers": 4
}
```

---

## 6. 过程数据格式

### 6.1 pipeline_process_data.json

```json
{
    "config": {
        "translators": [...],
        "polishers": [...],
        "evaluators": [...],
        "lang_in": "en",
        "lang_out": "zh"
    },
    "paragraphs": [
        {
            "paragraph_id": "p_1",
            "source_text": "原文...",
            "translations": [
                {
                    "model_name": "gpt-4o-mini",
                    "source_text": "原文...",
                    "translated_text": "翻译...",
                    "token_count": 50,
                    "prompt_tokens": 30,
                    "completion_tokens": 20
                }
            ],
            "polishes": [
                {
                    "model_name": "gpt-4o",
                    "source_translation": "翻译...",
                    "polished_text": "润色后...",
                    "original_source": "原文...",
                    "from_translator": "gpt-4o-mini"
                }
            ],
            "evaluations": [
                {
                    "evaluator_model": "gpt-4o",
                    "target_model": "gpt-4o-mini",
                    "target_type": "translator",
                    "scores": {
                        "accuracy": 8.5,
                        "fluency": 9.0,
                        "consistency": 8.0,
                        "terminology": 8.5,
                        "completeness": 9.0,
                        "average": 8.6
                    },
                    "reasoning": "评估理由..."
                }
            ],
            "selected_result": "最终选择的翻译结果",
            "selected_model": "gpt-4o-mini",
            "selected_type": "translator"
        }
    ],
    "summary": {
        "total_paragraphs": 100,
        "total_tokens": 50000,
        "model_scores": {
            "gpt-4o-mini": 8.6,
            "deepseek-chat": 8.2
        }
    }
}
```

---

## 7. 评估维度说明

| 维度 | 英文 | 说明 | 评分范围 |
|-----|------|------|---------|
| 准确性 | Accuracy | 翻译是否准确传达原文含义 | 1-10 |
| 流畅性 | Fluency | 译文是否通顺自然 | 1-10 |
| 一致性 | Consistency | 术语和风格是否统一 | 1-10 |
| 术语 | Terminology | 专业术语是否正确翻译 | 1-10 |
| 完整性 | Completeness | 是否完整翻译，无遗漏 | 1-10 |

**最终选择逻辑**：取所有评估模型对某个翻译/润色结果的平均分，选择平均分最高的结果。

---

## 8. 注意事项

1. **API兼容性**：支持所有OpenAI兼容的API端点（如DeepSeek、GLM等）
2. **并发控制**：通过 `max_workers` 参数控制并发数
3. **错误处理**：单个模型失败不影响其他模型，会记录错误信息
4. **缓存**：继承BaseTranslator的缓存机制
5. **资源消耗**：多模型会增加API调用次数，注意成本控制

---

## 9. 后续优化方向

1. [ ] 支持从配置文件加载流水线配置
2. [x] 添加评估报告PDF生成
3. [ ] 支持自定义评估Prompt
4. [ ] 添加模型性能对比图表
5. [ ] 支持异步流式输出
