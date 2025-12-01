#!/usr/bin/env python3
"""
独立测试脚本 - 测试完整的多阶段翻译流程

使用方法:
    1. 修改下面的配置参数
    2. 运行: uv run python scripts/run_pipeline_test.py
"""

import os
import asyncio
from pathlib import Path

# ============================================================
# 配置参数 - 请根据需要修改
# ============================================================

# 输入PDF文件路径
INPUT_PDF = "/Users/chtw/codes/iweaver/BabelDOC/examples/docs/20251201.pdf"

# 输出目录
OUTPUT_DIR = "/Users/chtw/codes/iweaver/BabelDOC/output/pipeline2"

# 语言设置
LANG_IN = "en"
LANG_OUT = "cn"


api_key = os.getenv("OPENAI_API_KEY")

# API配置 - 翻译模型
TRANSLATORS = [
    {
        "model_name": "deepseek-ai/DeepSeek-V3.2-Exp",
        "base_url": "https://api.modelverse.cn/v1/",
        "api_key": api_key,
    },

]

# API配置 - 润色模型 (可选，留空则跳过润色阶段)
POLISHERS = [
    {
        "model_name": "gemini-2.5-flash",
        "base_url": "https://api.modelverse.cn/v1/",
        "api_key": api_key
    },
]

# API配置 - 评估模型 (可选，留空则跳过评估阶段)
EVALUATORS = [
    {
        "model_name": "gemini-2.5-flash",
        "base_url": "https://api.modelverse.cn/v1/",
        "api_key": api_key
    },
]

# 并发设置
MAX_WORKERS = 32

# ============================================================
# 以下是执行代码，一般不需要修改
# ============================================================

from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn

from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.format.pdf.high_level import async_translate
from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
from babeldoc.translator.pipeline import (
    ModelConfig,
    ModelType,
    PipelineConfig,
    PipelineTranslator,
)


async def run_translation(translation_config, pipeline_translator):
    """使用异步方式运行翻译，显示进度条"""

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    translate_task_id = progress.add_task("总进度", total=100)
    stage_tasks = {}
    result = None

    with progress:
        async for event in async_translate(translation_config):
            if event["type"] == "stage_summary":
                # 初始化阶段任务
                for stage_info in event.get("stages", []):
                    stage_name = stage_info["name"]
                    if stage_name not in stage_tasks:
                        stage_tasks[stage_name] = progress.add_task(
                            f"  {stage_name}",
                            total=100,
                            visible=False,
                        )

            elif event["type"] == "progress_start":
                stage = event["stage"]
                if stage in stage_tasks:
                    progress.update(
                        stage_tasks[stage],
                        visible=True,
                        total=event.get("stage_total", 100),
                    )

            elif event["type"] == "progress_update":
                stage = event["stage"]
                if stage in stage_tasks:
                    progress.update(
                        stage_tasks[stage],
                        completed=event["stage_current"],
                        total=event["stage_total"],
                    )
                progress.update(
                    translate_task_id,
                    completed=event["overall_progress"],
                )

            elif event["type"] == "progress_end":
                stage = event["stage"]
                if stage in stage_tasks:
                    progress.update(
                        stage_tasks[stage],
                        completed=event["stage_total"],
                        total=event["stage_total"],
                    )
                progress.update(
                    translate_task_id,
                    completed=event["overall_progress"],
                )

            elif event["type"] == "finish":
                result = event.get("translate_result")

            elif event["type"] == "error":
                print(f"\n错误: {event.get('error')}")
                raise Exception(event.get("error"))

    return result


def main():
    print("=" * 60)
    print("多阶段翻译流水线测试")
    print("=" * 60)

    # 检查输入文件
    input_path = Path(INPUT_PDF)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {INPUT_PDF}")
        return 1

    # 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # 构建模型配置
    translator_configs = [
        ModelConfig(
            model_name=t["model_name"],
            model_type=ModelType.TRANSLATOR,
            base_url=t["base_url"],
            api_key=t["api_key"],
        )
        for t in TRANSLATORS
    ]

    polisher_configs = [
        ModelConfig(
            model_name=p["model_name"],
            model_type=ModelType.POLISHER,
            base_url=p["base_url"],
            api_key=p["api_key"],
        )
        for p in POLISHERS
    ]

    evaluator_configs = [
        ModelConfig(
            model_name=e["model_name"],
            model_type=ModelType.EVALUATION,
            base_url=e["base_url"],
            api_key=e["api_key"],
        )
        for e in EVALUATORS
    ]

    # 创建流水线配置
    pipeline_config = PipelineConfig(
        translators=translator_configs,
        polishers=polisher_configs,
        evaluators=evaluator_configs,
        lang_in=LANG_IN,
        lang_out=LANG_OUT,
        max_workers=MAX_WORKERS,
    )

    print(f"\n配置信息:")
    print(f"  输入文件: {input_path}")
    print(f"  输出目录: {output_path}")
    print(f"  语言: {LANG_IN} → {LANG_OUT}")
    print(f"  翻译模型: {[t['model_name'] for t in TRANSLATORS]}")
    print(f"  润色模型: {[p['model_name'] for p in POLISHERS] or '无'}")
    print(f"  评估模型: {[e['model_name'] for e in EVALUATORS] or '无'}")

    # 创建翻译器
    pipeline_translator = PipelineTranslator(
        pipeline_config=pipeline_config,
        output_dir=output_path,
    )

    # 加载文档布局模型
    print(f"\n加载文档布局模型...")
    doc_layout_model = DocLayoutModel.load_onnx()

    # 创建翻译配置
    translation_config = TranslationConfig(
        input_file=input_path,
        translator=pipeline_translator,
        lang_in=LANG_IN,
        lang_out=LANG_OUT,
        doc_layout_model=doc_layout_model,
        output_dir=output_path,
        auto_extract_glossary=False,  # 禁用自动术语提取
        pool_max_workers=MAX_WORKERS,  # 并发数
        watermark_output_mode=WatermarkOutputMode.NoWatermark,  # 去掉水印
    )

    print(f"\n开始翻译...")
    print("-" * 60)

    # 执行翻译（使用异步方式，显示进度条）
    # 注意：finalize_batch() 和 update_docs_with_best_results()
    # 已在 high_level.py 的翻译流程中自动调用
    try:
        result = asyncio.run(run_translation(translation_config, pipeline_translator))

        print("\n" + "=" * 60)
        print("翻译流程完成!")
        print("=" * 60)

        # 显示输出文件
        if result:
            print(f"\n输出文件:")
            if result.mono_pdf_path:
                print(f"  译文PDF: {result.mono_pdf_path}")
            if result.dual_pdf_path:
                print(f"  对照PDF: {result.dual_pdf_path}")

        # 保存过程数据
        json_path = pipeline_translator.save_process_data()
        print(f"  过程数据: {json_path}")

        # 生成评估报告
        report_path = pipeline_translator.generate_report()
        print(f"  评估报告: {report_path}")

        # 显示统计信息
        process_data = pipeline_translator.get_process_data()
        print(f"\n统计信息:")
        print(f"  总段落数: {process_data.total_paragraphs}")
        print(f"  总Token数: {process_data.total_tokens}")

        if process_data.model_scores:
            print(f"\n模型得分:")
            for model, score in sorted(
                process_data.model_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                print(f"  {model}: {score:.2f}")

            best = process_data.get_best_model()
            if best:
                print(f"\n最佳模型: {best}")

    except Exception as e:
        print(f"\n翻译失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
