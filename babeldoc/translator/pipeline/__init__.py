# Multi-stage translation pipeline
# Includes: Translator, Polisher, Evaluation, Assembly stages

from babeldoc.translator.pipeline.models import (
    EvaluationDimension,
    EvaluationResult,
    EvaluationScores,
    ModelConfig,
    ModelType,
    ParagraphProcessData,
    PipelineConfig,
    PipelineProcessData,
    PolishResult,
    TokenUsage,
    TranslationResult,
)
from babeldoc.translator.pipeline.translator import PipelineTranslator
from babeldoc.translator.pipeline.report_generator import (
    generate_evaluation_report_html,
    extract_actual_source_text,
    extract_actual_translation_text,
    extract_raw_translation_output,
    strip_markdown_code_block,
)

__all__ = [
    # Config
    "ModelConfig",
    "ModelType",
    "PipelineConfig",
    # Token usage
    "TokenUsage",
    # Results
    "TranslationResult",
    "PolishResult",
    "EvaluationResult",
    "EvaluationScores",
    "EvaluationDimension",
    # Process data
    "ParagraphProcessData",
    "PipelineProcessData",
    # Main translator
    "PipelineTranslator",
    # Report generator
    "generate_evaluation_report_html",
    "extract_actual_source_text",
    "extract_actual_translation_text",
    "extract_raw_translation_output",
    "strip_markdown_code_block",
]
