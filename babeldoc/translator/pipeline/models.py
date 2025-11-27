"""Data models for multi-stage translation pipeline."""

import json
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any


class ModelType(str, Enum):
    """Model type enum."""

    TRANSLATOR = "Translator"
    POLISHER = "Polisher"
    EVALUATION = "Evaluation"


@dataclass
class TokenUsage:
    """Detailed token usage statistics for an API call."""

    input_tokens: int = 0  # 输入 tokens (prompt)
    output_tokens: int = 0  # 输出 tokens (completion)
    cache_read_tokens: int = 0  # 从缓存读取的 tokens
    cache_write_tokens: int = 0  # 写入缓存的 tokens
    total_tokens: int = 0  # 总 tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_api_response(cls, usage: Any) -> "TokenUsage":
        """Create TokenUsage from OpenAI API response usage object."""
        if usage is None:
            return cls()

        # 标准 OpenAI 格式
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        # 缓存相关 (一些 API 提供商支持)
        cache_read_tokens = 0
        cache_write_tokens = 0

        # 检查 prompt_tokens_details (OpenAI 格式)
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            details = usage.prompt_tokens_details
            cache_read_tokens = getattr(details, "cached_tokens", 0) or 0

        # 检查 cache_read_input_tokens (Anthropic/其他格式)
        if hasattr(usage, "cache_read_input_tokens"):
            cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
        if hasattr(usage, "cache_creation_input_tokens"):
            cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            total_tokens=total_tokens,
        )


class EvaluationDimension(str, Enum):
    """Evaluation dimensions for translation quality assessment."""

    ACCURACY = "Accuracy"  # 准确性：翻译是否准确传达原文含义
    FLUENCY = "Fluency"  # 流畅性：译文是否通顺自然
    CONSISTENCY = "Consistency"  # 一致性：术语和风格是否统一
    TERMINOLOGY = "Terminology"  # 术语：专业术语是否正确
    COMPLETENESS = "Completeness"  # 完整性：是否完整翻译，无遗漏


@dataclass
class ModelConfig:
    """Model configuration for a single model."""

    model_name: str
    model_type: ModelType
    base_url: str
    api_key: str
    extra_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "config": {
                "base_url": self.base_url,
                "api_key": self.api_key,
                **self.extra_config,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        config = data.get("config", {})
        return cls(
            model_name=data["model_name"],
            model_type=ModelType(data["model_type"]),
            base_url=config.get("base_url", ""),
            api_key=config.get("api_key", ""),
            extra_config={
                k: v for k, v in config.items() if k not in ("base_url", "api_key")
            },
        )


@dataclass
class PipelineConfig:
    """Configuration for the entire translation pipeline."""

    translators: list[ModelConfig] = field(default_factory=list)
    polishers: list[ModelConfig] = field(default_factory=list)
    evaluators: list[ModelConfig] = field(default_factory=list)

    # Shared settings
    lang_in: str = "en"
    lang_out: str = "zh"
    qps: int = 4
    ignore_cache: bool = False
    max_workers: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "translators": [t.to_dict() for t in self.translators],
            "polishers": [p.to_dict() for p in self.polishers],
            "evaluators": [e.to_dict() for e in self.evaluators],
            "lang_in": self.lang_in,
            "lang_out": self.lang_out,
            "qps": self.qps,
            "ignore_cache": self.ignore_cache,
            "max_workers": self.max_workers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            translators=[ModelConfig.from_dict(t) for t in data.get("translators", [])],
            polishers=[ModelConfig.from_dict(p) for p in data.get("polishers", [])],
            evaluators=[ModelConfig.from_dict(e) for e in data.get("evaluators", [])],
            lang_in=data.get("lang_in", "en"),
            lang_out=data.get("lang_out", "zh"),
            qps=data.get("qps", 4),
            ignore_cache=data.get("ignore_cache", False),
            max_workers=data.get("max_workers", 4),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PipelineConfig":
        """Load config from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_json(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


@dataclass
class TranslationResult:
    """Result from a single translation model."""

    model_name: str
    source_text: str  # 存需要翻译的文本
    processed_text: str  # 存翻译后去除 json 格式的文本（用于显示和评估）
    raw_json: str = ""  # 大模型返回的原始 JSON，去除 markdown 代码块（用于更新 IL 文档）
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)

    # 向后兼容的属性
    @property
    def token_count(self) -> int:
        return self.token_usage.total_tokens

    @property
    def prompt_tokens(self) -> int:
        return self.token_usage.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.token_usage.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "source_text": self.source_text,
            "processed_text": self.processed_text,
            "raw_json": self.raw_json,
            "token_usage": self.token_usage.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class PolishResult:
    """Result from a single polish model."""

    model_name: str
    source_text: str  # 存需要翻译的文本（原文）
    processed_text: str  # 存润色后去除 json 格式的文本（用于显示和评估）
    from_translator: str  # 原翻译来自哪个模型
    raw_json: str = ""  # 润色器返回的原始 JSON，去除 markdown 代码块（用于更新 IL 文档）
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)

    # 向后兼容的属性
    @property
    def token_count(self) -> int:
        return self.token_usage.total_tokens

    @property
    def prompt_tokens(self) -> int:
        return self.token_usage.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.token_usage.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "source_text": self.source_text,
            "processed_text": self.processed_text,
            "from_translator": self.from_translator,
            "raw_json": self.raw_json,
            "token_usage": self.token_usage.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class EvaluationScores:
    """Scores for all evaluation dimensions."""

    accuracy: float = 0.0
    fluency: float = 0.0
    consistency: float = 0.0
    terminology: float = 0.0
    completeness: float = 0.0

    @property
    def average(self) -> float:
        """Calculate average score across all dimensions."""
        return (
            self.accuracy
            + self.fluency
            + self.consistency
            + self.terminology
            + self.completeness
        ) / 5.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "fluency": self.fluency,
            "consistency": self.consistency,
            "terminology": self.terminology,
            "completeness": self.completeness,
            "average": self.average,
        }


@dataclass
class EvaluationResult:
    """Result from evaluation of a translation/polish result."""

    evaluator_model: str
    target_model: str  # 被评估的模型名称
    target_type: str  # "translator" or "polisher"
    source_text: str  # 存需要翻译的文本（原文）
    processed_text: str  # 存翻译/润色后去除 json 格式的文本（用于显示和评估）
    raw_json: str = ""  # 从翻译/润色结果 copy 的原始 JSON（用于更新 IL 文档）
    scores: EvaluationScores = field(default_factory=EvaluationScores)
    reasoning: str = ""  # 评估理由
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)

    # 向后兼容的属性
    @property
    def token_count(self) -> int:
        return self.token_usage.total_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator_model": self.evaluator_model,
            "target_model": self.target_model,
            "target_type": self.target_type,
            "source_text": self.source_text,
            "processed_text": self.processed_text,
            "raw_json": self.raw_json,
            "scores": self.scores.to_dict(),
            "reasoning": self.reasoning,
            "token_usage": self.token_usage.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class ParagraphProcessData:
    """Process data for a single paragraph."""

    paragraph_id: str
    source_text: str
    translations: list[TranslationResult] = field(default_factory=list)
    polishes: list[PolishResult] = field(default_factory=list)
    evaluations: list[EvaluationResult] = field(default_factory=list)
    selected_result: str = ""  # 最终选择的翻译结果 (用于报告显示)
    selected_model: str = ""  # 最终选择的模型
    selected_type: str = ""  # "translator" or "polisher"
    selected_raw_text: str = ""  # 最终选择的原始文本（用于更新 IL 文档）
    # 批次内 id 到段落 debug_id 的映射，用于润色后更新对应段落
    batch_id_to_debug_id: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "source_text": self.source_text,
            "translations": [t.to_dict() for t in self.translations],
            "polishes": [p.to_dict() for p in self.polishes],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "selected_result": self.selected_result,
            "selected_model": self.selected_model,
            "selected_type": self.selected_type,
            "selected_raw_text": self.selected_raw_text,
            "batch_id_to_debug_id": self.batch_id_to_debug_id,
        }


@dataclass
class PipelineProcessData:
    """Complete process data from the translation pipeline."""

    config: PipelineConfig
    paragraphs: list[ParagraphProcessData] = field(default_factory=list)

    # Summary statistics
    total_paragraphs: int = 0
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    model_scores: dict[str, float] = field(default_factory=dict)  # model -> avg score

    # 按模型统计的 token 使用
    model_token_usage: dict[str, TokenUsage] = field(default_factory=dict)

    # 向后兼容
    @property
    def total_tokens(self) -> int:
        return self.total_token_usage.total_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "paragraphs": [p.to_dict() for p in self.paragraphs],
            "summary": {
                "total_paragraphs": self.total_paragraphs,
                "total_token_usage": self.total_token_usage.to_dict(),
                "model_token_usage": {
                    model: usage.to_dict()
                    for model, usage in self.model_token_usage.items()
                },
                "model_scores": self.model_scores,
            },
        }

    def get_best_model(self) -> str | None:
        """Get the model with highest average score."""
        if not self.model_scores:
            return None
        return max(self.model_scores.items(), key=lambda x: x[1])[0]

    def add_token_usage(self, model_name: str, usage: TokenUsage) -> None:
        """Add token usage for a model and update total."""
        # 更新总计
        self.total_token_usage.input_tokens += usage.input_tokens
        self.total_token_usage.output_tokens += usage.output_tokens
        self.total_token_usage.cache_read_tokens += usage.cache_read_tokens
        self.total_token_usage.cache_write_tokens += usage.cache_write_tokens
        self.total_token_usage.total_tokens += usage.total_tokens

        # 更新模型统计
        if model_name not in self.model_token_usage:
            self.model_token_usage[model_name] = TokenUsage()

        model_usage = self.model_token_usage[model_name]
        model_usage.input_tokens += usage.input_tokens
        model_usage.output_tokens += usage.output_tokens
        model_usage.cache_read_tokens += usage.cache_read_tokens
        model_usage.cache_write_tokens += usage.cache_write_tokens
        model_usage.total_tokens += usage.total_tokens

    def save_to_json(self, path: str | Path) -> None:
        """Save process data to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
