"""Tests for multi-stage translation pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from babeldoc.translator.pipeline import (
    EvaluationResult,
    EvaluationScores,
    ModelConfig,
    ModelType,
    ParagraphProcessData,
    PipelineConfig,
    PipelineProcessData,
    PipelineTranslator,
    PolishResult,
    TokenUsage,
    TranslationResult,
    generate_evaluation_report_html,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_to_dict(self):
        config = ModelConfig(
            model_name="gpt-4o-mini",
            model_type=ModelType.TRANSLATOR,
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        result = config.to_dict()

        assert result["model_name"] == "gpt-4o-mini"
        assert result["model_type"] == "Translator"
        assert result["config"]["base_url"] == "https://api.openai.com/v1"
        assert result["config"]["api_key"] == "sk-test"

    def test_from_dict(self):
        data = {
            "model_name": "deepseek-chat",
            "model_type": "Polisher",
            "config": {
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "sk-test",
            },
        }
        config = ModelConfig.from_dict(data)

        assert config.model_name == "deepseek-chat"
        assert config.model_type == ModelType.POLISHER
        assert config.base_url == "https://api.deepseek.com/v1"


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_to_dict_and_from_dict(self):
        config = PipelineConfig(
            translators=[
                ModelConfig(
                    model_name="gpt-4o-mini",
                    model_type=ModelType.TRANSLATOR,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                )
            ],
            polishers=[
                ModelConfig(
                    model_name="gpt-4o",
                    model_type=ModelType.POLISHER,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                )
            ],
            lang_in="en",
            lang_out="zh",
        )

        # Convert to dict and back
        data = config.to_dict()
        restored = PipelineConfig.from_dict(data)

        assert len(restored.translators) == 1
        assert len(restored.polishers) == 1
        assert restored.lang_in == "en"
        assert restored.lang_out == "zh"

    def test_from_json_file(self, tmp_path):
        config_data = {
            "translators": [
                {
                    "model_name": "gpt-4o-mini",
                    "model_type": "Translator",
                    "config": {
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-test",
                    },
                }
            ],
            "polishers": [],
            "evaluators": [],
            "lang_in": "en",
            "lang_out": "zh",
        }

        json_path = tmp_path / "config.json"
        with open(json_path, "w") as f:
            json.dump(config_data, f)

        config = PipelineConfig.from_json_file(json_path)
        assert len(config.translators) == 1
        assert config.translators[0].model_name == "gpt-4o-mini"


class TestTranslationResult:
    """Tests for TranslationResult."""

    def test_to_dict(self):
        token_usage = TokenUsage(
            input_tokens=30,
            output_tokens=20,
            total_tokens=50,
        )
        result = TranslationResult(
            model_name="gpt-4o-mini",
            source_text="Hello, world!",
            processed_text="你好，世界！",
            token_usage=token_usage,
        )

        data = result.to_dict()
        assert data["model_name"] == "gpt-4o-mini"
        assert data["source_text"] == "Hello, world!"
        assert data["processed_text"] == "你好，世界！"
        assert data["token_usage"]["total_tokens"] == 50
        assert data["token_usage"]["input_tokens"] == 30
        assert data["token_usage"]["output_tokens"] == 20


class TestEvaluationScores:
    """Tests for EvaluationScores."""

    def test_average_calculation(self):
        scores = EvaluationScores(
            accuracy=8.0,
            fluency=9.0,
            consistency=8.5,
            terminology=7.5,
            completeness=9.0,
        )

        # Average should be (8 + 9 + 8.5 + 7.5 + 9) / 5 = 8.4
        assert scores.average == pytest.approx(8.4)

    def test_to_dict(self):
        scores = EvaluationScores(accuracy=8.0, fluency=9.0)
        data = scores.to_dict()

        assert "accuracy" in data
        assert "average" in data


class TestParagraphProcessData:
    """Tests for ParagraphProcessData."""

    def test_to_dict(self):
        para_data = ParagraphProcessData(
            paragraph_id="p_1",
            source_text="Test paragraph",
            translations=[
                TranslationResult(
                    model_name="gpt-4o-mini",
                    source_text="Test paragraph",
                    processed_text="测试段落",
                )
            ],
            selected_result="测试段落",
            selected_model="gpt-4o-mini",
            selected_type="translator",
        )

        data = para_data.to_dict()
        assert data["paragraph_id"] == "p_1"
        assert len(data["translations"]) == 1
        assert data["selected_model"] == "gpt-4o-mini"


class TestPipelineProcessData:
    """Tests for PipelineProcessData."""

    def test_get_best_model(self):
        config = PipelineConfig()
        data = PipelineProcessData(
            config=config,
            model_scores={
                "gpt-4o-mini": 7.5,
                "deepseek-chat": 8.2,
                "gpt-4o": 8.8,
            },
        )

        assert data.get_best_model() == "gpt-4o"

    def test_get_best_model_empty(self):
        config = PipelineConfig()
        data = PipelineProcessData(config=config)

        assert data.get_best_model() is None

    def test_save_to_json(self, tmp_path):
        config = PipelineConfig(lang_in="en", lang_out="zh")
        token_usage = TokenUsage(
            input_tokens=800,
            output_tokens=200,
            total_tokens=1000,
        )
        data = PipelineProcessData(
            config=config,
            total_paragraphs=5,
            total_token_usage=token_usage,
        )

        json_path = tmp_path / "process_data.json"
        data.save_to_json(json_path)

        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
            assert loaded["summary"]["total_paragraphs"] == 5
            assert loaded["summary"]["total_token_usage"]["total_tokens"] == 1000


class TestPipelineTranslatorMocked:
    """Tests for PipelineTranslator with mocked API calls."""

    @patch("babeldoc.translator.pipeline.translator.openai.OpenAI")
    def test_initialization(self, mock_openai):
        config = PipelineConfig(
            translators=[
                ModelConfig(
                    model_name="gpt-4o-mini",
                    model_type=ModelType.TRANSLATOR,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                )
            ],
            lang_in="en",
            lang_out="zh",
        )

        translator = PipelineTranslator(config)

        assert len(translator.translator_clients) == 1
        assert len(translator.polisher_clients) == 0
        assert len(translator.evaluator_clients) == 0
        assert translator.lang_in == "en"
        assert translator.lang_out == "zh"

    @patch("babeldoc.translator.pipeline.translator.openai.OpenAI")
    def test_llm_translate(self, mock_openai):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "测试翻译"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 20

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        config = PipelineConfig(
            translators=[
                ModelConfig(
                    model_name="gpt-4o-mini",
                    model_type=ModelType.TRANSLATOR,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                )
            ],
            lang_in="en",
            lang_out="zh",
            ignore_cache=True,  # Disable cache to ensure do_llm_translate is called
        )

        translator = PipelineTranslator(config)
        # Use llm_translate which records paragraphs for batch processing
        result = translator.llm_translate("Test text")

        assert result == "测试翻译"

        # Check process data - llm_translate records the translation for batch processing
        process_data = translator.get_process_data()
        assert process_data.total_paragraphs == 1
        assert len(process_data.paragraphs) == 1
        assert process_data.paragraphs[0].source_text == "Test text"

    def test_save_process_data(self, tmp_path):
        with patch("babeldoc.translator.pipeline.translator.openai.OpenAI"):
            config = PipelineConfig(lang_in="en", lang_out="zh")
            translator = PipelineTranslator(config, output_dir=tmp_path)

            # Manually add some test data
            translator._process_data.paragraphs.append(
                ParagraphProcessData(
                    paragraph_id="p_1",
                    source_text="Test",
                    selected_result="测试",
                    selected_model="gpt-4o-mini",
                )
            )
            translator._process_data.total_paragraphs = 1

            json_path = translator.save_process_data()
            assert json_path is not None
            assert Path(json_path).exists()

    def test_generate_report(self, tmp_path):
        with patch("babeldoc.translator.pipeline.translator.openai.OpenAI"):
            config = PipelineConfig(
                translators=[
                    ModelConfig(
                        model_name="gpt-4o-mini",
                        model_type=ModelType.TRANSLATOR,
                        base_url="https://api.openai.com/v1",
                        api_key="sk-test",
                    )
                ],
                lang_in="en",
                lang_out="zh",
            )
            translator = PipelineTranslator(config, output_dir=tmp_path)
            translator._model_scores["gpt-4o-mini"] = [8.5, 9.0, 8.0]

            report_path = translator.generate_report()
            assert report_path is not None
            assert Path(report_path).exists()
            # Report is now HTML format
            assert report_path.endswith(".html")
            assert Path(report_path).stat().st_size > 0


class TestPipelineTranslatorIntegration:
    """Integration tests for PipelineTranslator with BaseTranslator interface."""

    @patch("babeldoc.translator.pipeline.translator.openai.OpenAI")
    def test_inherits_base_translator(self, mock_openai):
        from babeldoc.translator.translator import BaseTranslator

        config = PipelineConfig(lang_in="en", lang_out="zh")
        translator = PipelineTranslator(config)

        assert isinstance(translator, BaseTranslator)
        assert translator.name == "pipeline"

    @patch("babeldoc.translator.pipeline.translator.openai.OpenAI")
    def test_placeholder_methods(self, mock_openai):
        config = PipelineConfig(lang_in="en", lang_out="zh")
        translator = PipelineTranslator(config)

        # Test formula placeholder
        placeholder, pattern = translator.get_formular_placeholder(1)
        assert "{v1}" in placeholder

        # Test rich text placeholders
        left, _ = translator.get_rich_text_left_placeholder(1)
        right, _ = translator.get_rich_text_right_placeholder(1)
        assert "style" in left
        assert "style" in right

    @patch("babeldoc.translator.pipeline.translator.openai.OpenAI")
    def test_update_docs_with_best_results(self, mock_openai):
        """Test updating IL docs with best translation results."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "初始翻译"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 20

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        config = PipelineConfig(
            translators=[
                ModelConfig(
                    model_name="gpt-4o-mini",
                    model_type=ModelType.TRANSLATOR,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                )
            ],
            lang_in="en",
            lang_out="zh",
            ignore_cache=True,  # Disable cache to ensure translation runs
        )

        translator = PipelineTranslator(config)
        # Use llm_translate which records paragraphs for batch processing
        result = translator.llm_translate("Test text")

        assert result == "初始翻译"

        # Simulate updating the selected result to a better polisher result after evaluation
        para_data = translator._process_data.paragraphs[0]
        para_data.selected_result = "更好的翻译"
        para_data.selected_type = "polisher"  # Set type to polisher to trigger update
        para_data.selected_raw_text = "更好的翻译"  # Set selected_raw_text which is used for IL update

        # Create a mock IL document structure
        mock_paragraph = MagicMock()
        mock_paragraph.unicode = "初始翻译"

        mock_page = MagicMock()
        mock_page.pdf_paragraph = [mock_paragraph]

        mock_docs = MagicMock()
        mock_docs.page = [mock_page]

        # Call update_docs_with_best_results
        updated_count = translator.update_docs_with_best_results(mock_docs)

        assert updated_count == 1
        assert mock_paragraph.unicode == "更好的翻译"

    @patch("babeldoc.translator.pipeline.translator.openai.OpenAI")
    def test_update_docs_no_change_when_same(self, mock_openai):
        """Test that update skips paragraphs when result is the same."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "翻译结果"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 20

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        config = PipelineConfig(
            translators=[
                ModelConfig(
                    model_name="gpt-4o-mini",
                    model_type=ModelType.TRANSLATOR,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                )
            ],
            lang_in="en",
            lang_out="zh",
            ignore_cache=True,  # Disable cache to ensure translation runs
        )

        translator = PipelineTranslator(config)
        # Use llm_translate which records paragraphs for batch processing
        result = translator.llm_translate("Test text")

        # Don't change the selected result - it stays the same as initial
        mock_paragraph = MagicMock()
        mock_paragraph.unicode = "翻译结果"

        mock_page = MagicMock()
        mock_page.pdf_paragraph = [mock_paragraph]

        mock_docs = MagicMock()
        mock_docs.page = [mock_page]

        updated_count = translator.update_docs_with_best_results(mock_docs)

        # Should be 0 because the result is the same
        assert updated_count == 0


class TestEvaluationReportGenerator:
    """Tests for HTML report generation."""

    def test_generate_html_report(self, tmp_path):
        """Test generating HTML evaluation report."""
        # Create test data
        config = PipelineConfig(
            translators=[
                ModelConfig(
                    model_name="gpt-4o-mini",
                    model_type=ModelType.TRANSLATOR,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                )
            ],
            evaluators=[
                ModelConfig(
                    model_name="gpt-4o",
                    model_type=ModelType.EVALUATION,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                )
            ],
            lang_in="en",
            lang_out="zh",
        )

        process_data = PipelineProcessData(
            config=config,
            total_paragraphs=2,
            total_token_usage=TokenUsage(input_tokens=400, output_tokens=100, total_tokens=500),
            model_scores={"gpt-4o-mini": 8.5},
        )

        # Add paragraph data with evaluations
        para1 = ParagraphProcessData(
            paragraph_id="p_1",
            source_text="Hello world",
            translations=[
                TranslationResult(
                    model_name="gpt-4o-mini",
                    source_text="Hello world",
                    processed_text="你好世界",
                    token_usage=TokenUsage(input_tokens=15, output_tokens=5, total_tokens=20),
                )
            ],
            evaluations=[
                EvaluationResult(
                    evaluator_model="gpt-4o",
                    target_model="gpt-4o-mini",
                    target_type="translator",
                    source_text="Hello world",
                    processed_text="你好世界",
                    scores=EvaluationScores(
                        accuracy=9.0,
                        fluency=8.5,
                        consistency=8.0,
                        terminology=8.5,
                        completeness=9.0,
                    ),
                    reasoning="Good translation",
                )
            ],
            selected_result="你好世界",
            selected_model="gpt-4o-mini",
            selected_type="translator",
        )
        process_data.paragraphs.append(para1)

        # Generate HTML report
        html_path = tmp_path / "test_report.html"
        result_path = generate_evaluation_report_html(process_data, html_path)

        assert result_path == str(html_path)
        assert html_path.exists()
        assert html_path.stat().st_size > 0

    def test_generate_html_report_basic(self, tmp_path):
        """Test HTML report generation with minimal data."""
        config = PipelineConfig(lang_in="en", lang_out="zh")
        process_data = PipelineProcessData(
            config=config,
            total_paragraphs=1,
            total_token_usage=TokenUsage(input_tokens=80, output_tokens=20, total_tokens=100),
        )

        html_path = tmp_path / "report.html"
        result = generate_evaluation_report_html(process_data, html_path)

        assert Path(result).exists()

    def test_generate_report_with_multiple_models(self, tmp_path):
        """Test HTML report with multiple translator and evaluator models."""
        config = PipelineConfig(
            translators=[
                ModelConfig(
                    model_name="gpt-4o-mini",
                    model_type=ModelType.TRANSLATOR,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                ),
                ModelConfig(
                    model_name="deepseek-chat",
                    model_type=ModelType.TRANSLATOR,
                    base_url="https://api.deepseek.com/v1",
                    api_key="sk-test",
                ),
            ],
            evaluators=[
                ModelConfig(
                    model_name="gpt-4o",
                    model_type=ModelType.EVALUATION,
                    base_url="https://api.openai.com/v1",
                    api_key="sk-test",
                ),
                ModelConfig(
                    model_name="claude-3",
                    model_type=ModelType.EVALUATION,
                    base_url="https://api.anthropic.com/v1",
                    api_key="sk-test",
                ),
            ],
            lang_in="en",
            lang_out="zh",
        )

        process_data = PipelineProcessData(
            config=config,
            total_paragraphs=1,
            total_token_usage=TokenUsage(input_tokens=150, output_tokens=50, total_tokens=200),
            model_scores={"gpt-4o-mini": 8.5, "deepseek-chat": 8.2},
        )

        # Add evaluations from multiple evaluators for multiple models
        para = ParagraphProcessData(
            paragraph_id="p_1",
            source_text="Test text",
            translations=[
                TranslationResult(
                    model_name="gpt-4o-mini",
                    source_text="Test text",
                    processed_text="测试文本1",
                ),
                TranslationResult(
                    model_name="deepseek-chat",
                    source_text="Test text",
                    processed_text="测试文本2",
                ),
            ],
            evaluations=[
                # gpt-4o evaluating gpt-4o-mini
                EvaluationResult(
                    evaluator_model="gpt-4o",
                    target_model="gpt-4o-mini",
                    target_type="translator",
                    source_text="Test text",
                    processed_text="测试文本1",
                    scores=EvaluationScores(
                        accuracy=9.0, fluency=8.5, consistency=8.0,
                        terminology=8.5, completeness=9.0
                    ),
                ),
                # gpt-4o evaluating deepseek
                EvaluationResult(
                    evaluator_model="gpt-4o",
                    target_model="deepseek-chat",
                    target_type="translator",
                    source_text="Test text",
                    processed_text="测试文本2",
                    scores=EvaluationScores(
                        accuracy=8.5, fluency=8.0, consistency=7.5,
                        terminology=8.0, completeness=8.5
                    ),
                ),
                # claude-3 evaluating gpt-4o-mini
                EvaluationResult(
                    evaluator_model="claude-3",
                    target_model="gpt-4o-mini",
                    target_type="translator",
                    source_text="Test text",
                    processed_text="测试文本1",
                    scores=EvaluationScores(
                        accuracy=8.5, fluency=9.0, consistency=8.5,
                        terminology=8.0, completeness=8.5
                    ),
                ),
                # claude-3 evaluating deepseek
                EvaluationResult(
                    evaluator_model="claude-3",
                    target_model="deepseek-chat",
                    target_type="translator",
                    source_text="Test text",
                    processed_text="测试文本2",
                    scores=EvaluationScores(
                        accuracy=8.0, fluency=8.5, consistency=8.0,
                        terminology=7.5, completeness=8.0
                    ),
                ),
            ],
            selected_result="测试文本1",
            selected_model="gpt-4o-mini",
            selected_type="translator",
        )
        process_data.paragraphs.append(para)

        html_path = tmp_path / "multi_model_report.html"
        result_path = generate_evaluation_report_html(process_data, html_path)

        assert html_path.exists()
        # Verify HTML has content
        assert html_path.stat().st_size > 500  # Should be reasonably large


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
