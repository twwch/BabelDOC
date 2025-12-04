"""
PipelineTranslator - Multi-stage translation with multiple models.

This translator integrates into BabelDOC's existing translation flow by
inheriting from BaseTranslator. It performs:
1. Translation with multiple models in parallel
2. Polish/refinement with polisher models
3. Evaluation across 5 dimensions
4. Selection of best result based on evaluation scores
"""

import json
import logging
import re
import threading
from collections import defaultdict
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any

import httpx
import openai
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from babeldoc.translator.pipeline.cache import EvaluationCache
from babeldoc.translator.pipeline.cache import PolishCache
from babeldoc.translator.pipeline.cache import TranslationCache
from babeldoc.translator.pipeline.cache import _compute_hash
from babeldoc.translator.pipeline.models import EvaluationResult
from babeldoc.translator.pipeline.models import EvaluationScores
from babeldoc.translator.pipeline.models import ModelConfig
from babeldoc.translator.pipeline.models import ParagraphProcessData
from babeldoc.translator.pipeline.models import PipelineConfig
from babeldoc.translator.pipeline.models import PipelineProcessData
from babeldoc.translator.pipeline.models import PolishResult
from babeldoc.translator.pipeline.models import TokenUsage
from babeldoc.translator.pipeline.models import TranslationResult
from babeldoc.translator.pipeline.report_generator import extract_actual_source_text
from babeldoc.translator.pipeline.report_generator import (
    extract_actual_translation_text,
)
from babeldoc.translator.pipeline.report_generator import (
    generate_evaluation_report_html,
)
from babeldoc.translator.pipeline.report_generator import strip_markdown_code_block
from babeldoc.translator.translator import BaseTranslator
from babeldoc.utils.priority_thread_pool_executor import PriorityThreadPoolExecutor

logger = logging.getLogger(__name__)


def _safe_parse_score(value: Any, default: float = 5.0) -> float:
    """
    Safely parse a score value to float.

    Handles cases where the model returns non-numeric values like "N/A", "N", etc.

    Args:
        value: The value to parse (could be int, float, str, or None)
        default: Default value if parsing fails

    Returns:
        Parsed float score, or default if parsing fails
    """
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        # Try to parse as number
        try:
            return float(value)
        except ValueError:
            # Handle common non-numeric responses
            return default
    return default


class ModelClient:
    """OpenAI-compatible client wrapper for a single model."""

    def __init__(self, config: ModelConfig):
        self.config = config
        # 不在初始化时创建 client，而是按需为每个线程创建
        self._thread_local = threading.local()

    def _get_client(self) -> openai.OpenAI:
        """获取当前线程的 OpenAI client（线程安全）。"""
        if not hasattr(self._thread_local, "client"):
            self._thread_local.client = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                http_client=httpx.Client(
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                    timeout=120,
                ),
            )
        return self._thread_local.client

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature = None,
        max_tokens = None,
        response_format: dict[str, str] | None = None,
    ) -> tuple[str, TokenUsage, list[dict[str, str]]]:
        """
        Send chat completion request.

        Returns:
            tuple: (content, TokenUsage, messages) - messages 用于记录到结果的 metadata
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        client = self._get_client()
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()
        token_usage = TokenUsage.from_api_response(response.usage)

        return content, token_usage, messages


class PipelineTranslator(BaseTranslator):
    """
    Multi-stage translation translator that integrates with BabelDOC.

    Inherits from BaseTranslator so it can be used as a drop-in replacement
    for OpenAITranslator in the existing translation flow.
    """

    name = "pipeline"

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        output_dir: Path | str | None = None,
    ):
        # Initialize BaseTranslator
        super().__init__(
            lang_in=pipeline_config.lang_in,
            lang_out=pipeline_config.lang_out,
            ignore_cache=pipeline_config.ignore_cache,
        )

        self.pipeline_config = pipeline_config
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_workers = pipeline_config.max_workers

        # Initialize model clients
        self.translator_clients = [
            ModelClient(cfg) for cfg in pipeline_config.translators
        ]
        self.polisher_clients = [
            ModelClient(cfg) for cfg in pipeline_config.polishers
        ]
        self.evaluator_clients = [
            ModelClient(cfg) for cfg in pipeline_config.evaluators
        ]

        # Process data tracking
        self._process_data = PipelineProcessData(config=pipeline_config)
        self._paragraph_counter = 0
        self._model_scores: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

        # Token counters (for compatibility)
        self.model = "pipeline"

        # 批量处理模式：收集所有段落的翻译结果，最后统一润色和评估
        self._pending_paragraphs: list[ParagraphProcessData] = []
        self._batch_mode = True  # 启用批量处理模式

        # 用于跟踪翻译结果对应的模型名称
        # key: translated_text, value: model_name
        # 在 do_llm_translate 中设置，在 _record_translation_for_batch 中使用
        self._translation_model_map: dict[str, str] = {}

        # 用于存储每个段落的 translate_input，供润色更新使用
        # key: paragraph.debug_id, value: translate_input
        self._pipeline_translate_inputs: dict[str, Any] = {}

        # 用于临时存储最新一批翻译的所有结果（多翻译模型时）
        # 在 do_llm_translate 中设置，在 _record_translation_for_batch 中使用后清空
        self._latest_translations: list[TranslationResult] | None = None

        logger.info(
            f"PipelineTranslator initialized with "
            f"{len(self.translator_clients)} translators, "
            f"{len(self.polisher_clients)} polishers, "
            f"{len(self.evaluator_clients)} evaluators"
        )

    def do_translate(self, text: str, rate_limit_params: dict | None = None) -> str:
        """Simple translation (falls back to first translator)."""
        return self.do_llm_translate(text, rate_limit_params)

    def llm_translate(self, text: str, ignore_cache: bool = False, rate_limit_params: dict | None = None) -> str:
        """
        Override BaseTranslator.llm_translate to track translation results for batch processing.

        Even when using cache, we need to record the translation result for later
        polish/evaluation stages.
        """
        if text is None:
            return None

        # 提取 batch_id_to_debug_id 映射（如果有）
        batch_id_to_debug_id = None
        if rate_limit_params:
            batch_id_to_debug_id = rate_limit_params.get("batch_id_to_debug_id")

        # Call parent's llm_translate (which may use cache)
        translation = super().llm_translate(text, ignore_cache, rate_limit_params)

        # Record the translation result for batch processing
        # This is called regardless of whether cache was hit
        self._record_translation_for_batch(text, translation, batch_id_to_debug_id)

        return translation

    def _record_translation_for_batch(
        self,
        source_text: str,
        translated_text: str,
        batch_id_to_debug_id: dict[int, str] | None = None,
    ) -> None:
        """Record all translation results for batch processing (polish/evaluation).

        When multiple translators are configured, this method records ALL translation
        results from _latest_translations, not just the one returned to ILTranslatorLLMOnly.
        """
        if not self._batch_mode:
            return

        if not translated_text:
            return

        with self._lock:
            self._paragraph_counter += 1
            paragraph_id = f"p_{self._paragraph_counter}"
            # Get all translations from the latest run
            latest_translations = getattr(self, '_latest_translations', None)

        # 提取实际的源文本（去掉 prompt 包装）用于显示和评估
        actual_source = extract_actual_source_text(source_text)

        # Create paragraph process data
        para_data = ParagraphProcessData(
            paragraph_id=paragraph_id,
            source_text=actual_source,
        )

        # 保存 batch_id_to_debug_id 映射
        if batch_id_to_debug_id:
            para_data.batch_id_to_debug_id = batch_id_to_debug_id

        # Record all translation results from all configured translators
        translation_results: list[TranslationResult] = []

        if latest_translations:
            # Use all translations from the multi-model run
            for t in latest_translations:
                if t.processed_text:
                    translation_results.append(t)
            logger.info(f"[Pipeline] 段落 {paragraph_id}: 记录 {len(translation_results)} 个翻译结果 "
                       f"(模型: {[t.model_name for t in translation_results]})")
        else:
            # Fallback: single translation (cache hit or single translator)
            model_name = self._translation_model_map.get(translated_text)
            if model_name is None:
                model_name = (
                    self.translator_clients[0].config.model_name
                    if self.translator_clients
                    else "unknown"
                )
            raw_json = strip_markdown_code_block(translated_text)
            actual_translation = extract_actual_translation_text(translated_text)
            translation_results.append(TranslationResult(
                model_name=model_name,
                source_text=actual_source,
                processed_text=actual_translation,
                raw_json=raw_json,
            ))

        para_data.translations = translation_results

        # 默认选择第一个翻译结果
        if translation_results:
            first = translation_results[0]
            para_data.selected_result = first.processed_text
            para_data.selected_model = first.model_name
            para_data.selected_type = "translator"
            para_data.selected_raw_text = first.raw_json

        with self._lock:
            self._pending_paragraphs.append(para_data)
            self._process_data.paragraphs.append(para_data)
            self._process_data.total_paragraphs += 1
            # Clear latest translations for next batch
            self._latest_translations = None

        logger.debug(f"[Pipeline] 段落 {paragraph_id}: 记录翻译结果 (待批量润色/评估)")

    def do_llm_translate(self, text: str, rate_limit_params: dict | None = None) -> str:
        """
        Actual translation implementation - calls the translation API.

        Note: This method is called by BaseTranslator.llm_translate() when cache misses.
        The batch processing recording is handled in llm_translate() override, not here.

        When multiple translators are configured, all translators run in parallel.
        All translation results are stored for later polish/evaluation stages.
        Only the first successful result is returned (for IL document update).
        """
        if text is None:
            return None

        # Run translation with all translator models
        logger.info(f"[Pipeline] do_llm_translate: running translation with "
                   f"{len(self.translator_clients)} translator(s)")
        translations = self._run_translations(text)

        # Update token counts for translations
        with self._lock:
            for t in translations:
                self._process_data.add_token_usage(t.model_name, t.token_usage)

        # Store all successful translations for later batch processing
        # The first successful result will be returned for IL document update
        first_result = None
        with self._lock:
            for t in translations:
                if t.processed_text:
                    # 记录这个翻译结果对应的模型名称（用 raw_json 作为 key）
                    self._translation_model_map[t.raw_json] = t.model_name
                    logger.info(f"[Pipeline] Translation from {t.model_name}: "
                               f"{t.processed_text[:100]}...")
                    if first_result is None:
                        first_result = t.raw_json

            # Store all translations for batch processing
            # This will be picked up by _record_translation_for_batch
            self._latest_translations = translations

        return first_result if first_result else ""

    def _process_paragraph_full(
        self,
        para_data: ParagraphProcessData,
        text: str,
        translations: list[TranslationResult],
    ) -> str:
        """完整处理单个段落（非批量模式）"""
        paragraph_id = para_data.paragraph_id

        # Stage 2: Polish
        polishes: list[PolishResult] = []
        if self.polisher_clients:
            logger.info(f"[Pipeline] 段落 {paragraph_id}: 润色中...")
            polishes = self._run_polishes(text, translations)
            para_data.polishes = polishes

        # Stage 3: Evaluation
        evaluations: list[EvaluationResult] = []
        if self.evaluator_clients:
            logger.info(f"[Pipeline] 段落 {paragraph_id}: 评估中...")
            evaluations = self._run_evaluations(text, translations, polishes)
            para_data.evaluations = evaluations

        # Stage 4: Select best
        display_text, raw_json, selected_model, selected_type = self._select_best(
            translations, polishes, evaluations
        )
        para_data.selected_result = display_text
        para_data.selected_raw_text = raw_json  # 存储选中结果的原始 JSON
        para_data.selected_model = selected_model
        para_data.selected_type = selected_type

        with self._lock:
            self._process_data.paragraphs.append(para_data)
            self._process_data.total_paragraphs += 1
            for p in polishes:
                self._process_data.add_token_usage(p.model_name, p.token_usage)
            for e in evaluations:
                self._process_data.add_token_usage(e.evaluator_model, e.token_usage)

        # 返回选中结果的原始 JSON
        return raw_json or display_text

    def finalize_batch(self, progress_monitor=None, pool_max_workers: int | None = None) -> None:
        """
        完成批量处理：对所有已翻译的段落执行评估、优化和最终选择。

        Workflow:
        1. Stage 2: 初始评估 (Model B) - 评估 Model A 的翻译，给出优化建议
        2. Stage 3: 优化 (Model C) - 如果分数 < 9.5，使用建议进行优化
        3. Stage 4: 对比评估 (Model B) - 评估 A vs C
        4. Stage 5: 选择 - 选择分数更高的结果

        Args:
            progress_monitor: 可选的进度监控器，用于显示进度条
            pool_max_workers: 可选的并发数，默认使用 pipeline_config.max_workers
        """
        logger.info(f"[Pipeline] finalize_batch called: batch_mode={self._batch_mode}, "
                    f"pending_paragraphs={len(self._pending_paragraphs)}, "
                    f"polisher_clients={len(self.polisher_clients)}, "
                    f"evaluator_clients={len(self.evaluator_clients)}")

        if not self._batch_mode:
            logger.warning("[Pipeline] finalize_batch called but batch_mode is disabled")
            return

        if not self._pending_paragraphs:
            logger.info("[Pipeline] No pending paragraphs to finalize")
            return

        # 使用传入的并发数或默认值
        workers = pool_max_workers if pool_max_workers is not None else self.max_workers
        logger.info(f"[Pipeline] 使用并发数: {workers}")

        total_paragraphs = len(self._pending_paragraphs)
        logger.info("[Pipeline] ========== 开始批量后处理 ==========")
        logger.info(f"[Pipeline] 待处理段落数: {total_paragraphs}")

        # Stage 2: 初始评估 (获取建议)
        if self.evaluator_clients:
            logger.info(f"[Pipeline] Stage 2: 初始评估 ({len(self.evaluator_clients)} 个评估模型)")
            # 这一步会评估当前的翻译结果 (Model A)，并生成 suggestions
            self._run_batch_evaluation(workers, progress_monitor, stage_name="Initial Evaluation")
            logger.info("[Pipeline] 初始评估完成")

        # Stage 3: 条件优化 (根据建议优化)
        if self.polisher_clients:
            logger.info(f"[Pipeline] Stage 3: 条件优化 ({len(self.polisher_clients)} 个润色模型)")
            # 筛选需要优化的段落 (score < 9.5)
            paragraphs_to_optimize = []
            for para in self._pending_paragraphs:
                # 获取该段落的最佳评估分数
                best_score = 0.0
                suggestions = ""
                
                # 查找针对翻译结果的评估
                for ev in para.evaluations:
                    if ev.target_type == "translator" and ev.scores.average > best_score:
                        best_score = ev.scores.average
                        suggestions = ev.suggestions
                        scores = ev.scores

                # 如果分数不够高且有建议，则进行优化
                if best_score < 9.5:
                    paragraphs_to_optimize.append((para, suggestions, scores))
            
            if paragraphs_to_optimize:
                logger.info(f"[Pipeline] 需要优化的段落数: {len(paragraphs_to_optimize)}")
                self._run_batch_optimization(workers, paragraphs_to_optimize, progress_monitor)
                logger.info("[Pipeline] 优化完成")
            else:
                logger.info("[Pipeline] 没有需要优化的段落")

        # Stage 4: 对比评估 (A vs C)
        # 如果进行了优化，我们需要评估优化后的结果，并与原结果对比
        # 这里再次运行批量评估，_run_batch_evaluation 会自动收集所有 candidates (包括新的 polishes) 进行对比
        if self.evaluator_clients and self.polisher_clients:
            logger.info(f"[Pipeline] Stage 4: 对比评估 (A vs C)")
            self._run_batch_evaluation(workers, progress_monitor, stage_name="Comparative Evaluation")
            logger.info("[Pipeline] 对比评估完成")

        # Stage 5: 选择最佳结果
        logger.info("[Pipeline] Stage 5: 选择最佳结果")
        for para_data in self._pending_paragraphs:
            display_text, raw_json, selected_model, selected_type = self._select_best(
                para_data.translations,
                para_data.polishes,
                para_data.evaluations,
            )
            para_data.selected_result = display_text
            para_data.selected_raw_text = raw_json  # 存储选中结果的原始 JSON
            para_data.selected_model = selected_model
            para_data.selected_type = selected_type
            logger.info(
                f"[Pipeline] 段落 {para_data.paragraph_id}: 选择 {selected_type} ({selected_model})"
            )

        # 计算模型平均分并记录统计
        self._process_data.model_scores = {
            model: sum(scores) / len(scores) if scores else 0
            for model, scores in self._model_scores.items()
        }

        logger.info("[Pipeline] ========== 批量后处理完成 ==========")
        logger.info(f"[Pipeline] 处理段落数: {len(self._pending_paragraphs)}")
        logger.info(f"[Pipeline] 模型评分: {self._process_data.model_scores}")

        self._pending_paragraphs.clear()

    def _run_batch_optimization(
        self, 
        workers: int, 
        paragraphs_to_optimize: list[tuple[ParagraphProcessData, str, EvaluationScores]], 
        progress_monitor=None
    ) -> None:
        """并发执行批量优化 (Stage 3)"""
        if not self.polisher_clients:
            return

        # 构建优化任务
        all_tasks: list[tuple[ParagraphProcessData, ModelClient, TranslationResult, str, EvaluationScores]] = []
        for para_data, suggestions, scores in paragraphs_to_optimize:
            # 使用第一个翻译结果作为基准进行优化
            if para_data.translations:
                trans = para_data.translations[0]
                if trans.processed_text:
                    for client in self.polisher_clients:
                        all_tasks.append((para_data, client, trans, suggestions, scores))

        # 用于跟踪每个段落的润色结果
        para_polishes: dict[str, list[PolishResult]] = {
            p[0].paragraph_id: [] for p in paragraphs_to_optimize
        }
        completed_paragraphs: set[str] = set()
        total_tasks = len(paragraphs_to_optimize)

        def optimize_single(
            para_data: ParagraphProcessData,
            client: ModelClient,
            trans: TranslationResult,
            suggestions: str,
            scores: EvaluationScores
        ) -> tuple[str, PolishResult]:
            """执行单个优化任务"""
            model_name = client.config.model_name
            input_json = trans.raw_json if trans.raw_json else trans.processed_text

            # Check cache (include suggestions in key)
            cache = PolishCache(model_name)
            # 使用 suggestions 作为额外的 key 部分
            extra_key = f"{trans.model_name}|opt|{_compute_hash(suggestions)}"
            cached = cache.get_polish(para_data.source_text, input_json, extra_key)
            
            if cached:
                polished_display, raw_json = cached
                logger.debug(f"[Pipeline] Optimization cache hit: {model_name}")
                return para_data.paragraph_id, PolishResult(
                    model_name=model_name,
                    source_text=para_data.source_text,
                    processed_text=polished_display,
                    from_translator=trans.model_name,
                    raw_json=raw_json,
                    metadata={"from_cache": True, "optimized": True},
                )

            # Format scores for the prompt
            scores_text = (
                f"- Accuracy: {scores.accuracy}/10\n"
                f"- Fluency: {scores.fluency}/10\n"
                f"- Consistency: {scores.consistency}/10\n"
                f"- Terminology: {scores.terminology}/10\n"
                f"- Completeness: {scores.completeness}/10"
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translation editor. "
                        f"Optimize the following {self.lang_out} translation based on the evaluation scores and suggestions.\n\n"
                        f"## Evaluation Scores\n{scores_text}\n\n"
                        f"## Optimization Goal\n"
                        f"Focus on improving the dimensions with lower scores while maintaining the strengths of the original translation.\n\n"
                        f"## CRITICAL RULES:\n"
                        f"1. Input is a JSON array. Output MUST be a JSON array with the SAME number of elements.\n"
                        f"2. Each element MUST keep the same 'id' value from the input.\n"
                        f"3. Replace 'input' field with 'output' field containing your optimized translation.\n"
                        f"4. Keep ALL <style id='N'>...</style> tags exactly as they appear.\n"
                        f"5. Keep ALL {{vN}} placeholders exactly as they appear.\n"
                        f"6. Only modify the actual translated text content, not the structure.\n\n"
                        f"## Example:\n"
                        f"Input: [{{'id': 0, 'output': 'text1'}}]\n"
                        f"Output: [{{'id': 0, 'output': 'optimized1'}}]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original ({self.lang_in}):\n{para_data.source_text}\n\n"
                        f"Current Translation (JSON array):\n{input_json}\n\n"
                        f"Optimization Suggestions:\n{suggestions}"
                    ),
                },
            ]
            
            try:
                content, token_usage, req_messages = client.chat(messages)
                raw_json = strip_markdown_code_block(content)
                processed_text = extract_actual_translation_text(raw_json)

                if processed_text:
                    cache.set_polish(
                        para_data.source_text,
                        input_json,
                        extra_key,
                        processed_text,
                        raw_json,
                    )

                result = PolishResult(
                    model_name=model_name,
                    source_text=para_data.source_text,
                    processed_text=processed_text,
                    from_translator=trans.model_name,
                    raw_json=raw_json,
                    token_usage=token_usage,
                    metadata={"request_messages": req_messages, "response": content, "optimized": True},
                )
            except Exception as e:
                logger.exception(f"Optimization failed: {model_name}")
                result = PolishResult(
                    model_name=model_name,
                    source_text=para_data.source_text,
                    processed_text=trans.processed_text,
                    from_translator=trans.model_name,
                    raw_json=trans.raw_json,
                    metadata={"error": str(e)},
                )
            return para_data.paragraph_id, result

        if progress_monitor:
            with progress_monitor.stage_start("Optimize Translations", total_tasks) as pbar:
                with PriorityThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(optimize_single, p, c, t, s, sc): (p, c, t, s, sc)
                        for p, c, t, s, sc in all_tasks
                    }
                    for future in as_completed(futures):
                        para_id, result = future.result()
                        with self._lock:
                            para_polishes[para_id].append(result)
                            self._process_data.add_token_usage(result.model_name, result.token_usage)
                            if len(para_polishes[para_id]) >= len(self.polisher_clients) and para_id not in completed_paragraphs:
                                completed_paragraphs.add(para_id)
                                pbar.advance(1)
        else:
            with PriorityThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(optimize_single, p, c, t, s, sc): (p, c, t, s, sc)
                    for p, c, t, s, sc in all_tasks
                }
                for future in as_completed(futures):
                    para_id, result = future.result()
                    with self._lock:
                        para_polishes[para_id].append(result)
                        self._process_data.add_token_usage(result.model_name, result.token_usage)
                        if len(para_polishes[para_id]) >= len(self.polisher_clients) and para_id not in completed_paragraphs:
                            completed_paragraphs.add(para_id)
                            logger.info(f"[Pipeline] 优化进度: {len(completed_paragraphs)}/{total_tasks}")

        # 将结果写回各段落 (追加到 polishes 列表)
        for para_data in self._pending_paragraphs:
            if para_data.paragraph_id in para_polishes:
                para_data.polishes.extend(para_polishes[para_data.paragraph_id])

    def _run_batch_evaluation(self, workers: int, progress_monitor=None, stage_name="Evaluate Translations") -> None:
        """并发执行批量评估 - 对比评估模式，每个段落的所有翻译/润色结果一起评估。"""
        total_paragraphs = len(self._pending_paragraphs)

        if not self.evaluator_clients:
            return

        # 构建评估任务: (para_data, client)
        # 每个段落只需要一个评估任务，评估模型会对比所有候选翻译
        all_tasks: list[tuple[ParagraphProcessData, ModelClient]] = []
        for para_data in self._pending_paragraphs:
            for client in self.evaluator_clients:
                all_tasks.append((para_data, client))

        # 用于跟踪每个段落的评估结果
        para_evaluations: dict[str, list[EvaluationResult]] = {
            p.paragraph_id: [] for p in self._pending_paragraphs
        }
        completed_paragraphs: set[str] = set()

        def evaluate_paragraph_comparative(
            para_data: ParagraphProcessData,
            client: ModelClient,
        ) -> tuple[str, list[EvaluationResult]]:
            """对比评估一个段落的所有翻译/润色结果，返回 (paragraph_id, results)"""
            model_name = client.config.model_name

            # 收集所有候选翻译
            # 使用唯一 ID 来避免相同模型名称的冲突
            candidates: list[dict[str, str]] = []

            # 添加翻译结果 - 使用 "translator:{model_name}" 格式
            for trans in para_data.translations:
                if trans.processed_text:
                    unique_id = f"translator:{trans.model_name}"
                    candidates.append({
                        "model_id": unique_id,
                        "display_name": trans.model_name,  # 用于报告显示
                        "type": "translator",
                        "translation": trans.processed_text,  # 用于显示和评估
                        "raw_json": trans.raw_json,  # 翻译器返回的原始 JSON
                    })

            # 添加润色结果 - 使用 "polisher:{from_translator}+{polisher_model}" 格式
            for polish in para_data.polishes:
                if polish.processed_text:
                    unique_id = f"polisher:{polish.from_translator}+{polish.model_name}"
                    candidates.append({
                        "model_id": unique_id,
                        "display_name": f"{polish.from_translator}+{polish.model_name}",  # 用于报告显示
                        "type": "polisher",
                        "translation": polish.processed_text,  # 用于显示和评估
                        "raw_json": polish.raw_json,  # 润色器返回的原始 JSON
                    })

            if not candidates:
                return para_data.paragraph_id, []

            # 构建候选翻译的 key 用于缓存（排序以保证一致性）
            candidates_key = "|".join(sorted(c["model_id"] + ":" + c["translation"] for c in candidates))

            # Check cache first
            cache = EvaluationCache(model_name)
            cached = cache.get_evaluation(para_data.source_text, candidates_key)
            if cached:
                logger.debug(f"[Pipeline] Evaluation cache hit: {model_name}")
                results: list[EvaluationResult] = []
                for cand in candidates:
                    model_id = cand["model_id"]
                    if model_id in cached:
                        model_eval = cached[model_id]
                        scores = EvaluationScores(
                            accuracy=_safe_parse_score(model_eval.get("accuracy")),
                            fluency=_safe_parse_score(model_eval.get("fluency")),
                            consistency=_safe_parse_score(model_eval.get("consistency")),
                            terminology=_safe_parse_score(model_eval.get("terminology")),
                            completeness=_safe_parse_score(model_eval.get("completeness")),
                        )
                        reasoning = model_eval.get("reasoning", "")
                        suggestions = model_eval.get("suggestions", "")

                        # Track scores for model ranking
                        with self._lock:
                            self._model_scores[model_id].append(scores.average)

                        results.append(EvaluationResult(
                            evaluator_model=model_name,
                            target_model=model_id,
                            target_type=cand["type"],
                            source_text=para_data.source_text,
                            processed_text=cand["translation"],
                            raw_json=cand["raw_json"],
                            scores=scores,
                            reasoning=reasoning,
                            suggestions=suggestions,
                            metadata={"from_cache": True},
                        ))
                return para_data.paragraph_id, results

            # 构建对比评估的 prompt
            candidates_text = ""
            for i, cand in enumerate(candidates):
                candidates_text += f"\n### 候选翻译 {i+1} (ID: {cand['model_id']}, 类型: {cand['type']})\n{cand['translation']}\n"

            # 构建期望的 JSON 输出格式
            output_format = {
                cand["model_id"]: {
                    "accuracy": "1-10分",
                    "fluency": "1-10分",
                    "consistency": "1-10分",
                    "terminology": "1-10分",
                    "completeness": "1-10分",
                    "reasoning": "简要说明打分理由（中文）",
                    "suggestions": "优化建议（中文）",
                }
                for cand in candidates
            }

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"你是一名专业的翻译质量评估专家。请对比评估多个从 {self.lang_in} 到 {self.lang_out} 的翻译候选结果。\n\n"
                        "## 评分维度（1-10分）\n"
                        "- accuracy（准确性）：翻译是否准确传达原文含义，有无误译、漏译\n"
                        "- fluency（流畅性）：译文是否通顺自然，符合目标语言表达习惯\n"
                        "- consistency（一致性）：术语和风格是否前后统一\n"
                        "- terminology（术语）：专业术语翻译是否准确、规范\n"
                        "- completeness（完整性）：是否完整翻译原文内容，无遗漏或冗余\n\n"
                        "## 评分要求\n"
                        "1. 请将所有候选翻译相互对比，翻译质量更好的应该获得更高分数\n"
                        "2. 分数应有区分度，避免所有候选都给相同分数\n"
                        "3. 评分应客观公正，基于实际翻译质量而非主观偏好\n"
                        "4. 注意：原文可能因技术原因被截断，如果译文比原文内容更完整，这是正常的，不应因此扣分\n"
                        "5. 如果原文很短或被截断，重点评估译文的流畅性和准确性，放宽对完整性的要求\n\n"
                        "## 输出要求\n"
                        "- reasoning 字段：用中文简要说明打分理由，指出翻译的优点和不足\n"
                        "- suggestions 字段：用中文给出具体的优化建议，说明如何改进翻译\n\n"
                        f"## 输出格式\n请输出 JSON 对象，每个 model_id 为键：\n{json.dumps(output_format, indent=2, ensure_ascii=False)}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"## 原文\n{para_data.source_text}\n\n## 候选翻译{candidates_text}",
                },
            ]

            results: list[EvaluationResult] = []
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    content, token_usage, req_messages = client.chat(
                        messages, response_format={"type": "json_object"}
                    )

                    # 解析对比评估结果
                    eval_data = json.loads(content)

                    # Cache the evaluation result
                    cache.set_evaluation(para_data.source_text, candidates_key, eval_data)

                    for cand in candidates:
                        model_id = cand["model_id"]
                        if model_id in eval_data:
                            model_eval = eval_data[model_id]
                            scores = EvaluationScores(
                                accuracy=_safe_parse_score(model_eval.get("accuracy")),
                                fluency=_safe_parse_score(model_eval.get("fluency")),
                                consistency=_safe_parse_score(model_eval.get("consistency")),
                                terminology=_safe_parse_score(model_eval.get("terminology")),
                                completeness=_safe_parse_score(model_eval.get("completeness")),
                            )
                            reasoning = model_eval.get("reasoning", "")
                            suggestions = model_eval.get("suggestions", "")

                            # Track scores for model ranking
                            with self._lock:
                                self._model_scores[model_id].append(scores.average)

                            results.append(EvaluationResult(
                                evaluator_model=model_name,
                                target_model=model_id,
                                target_type=cand["type"],
                                source_text=para_data.source_text,
                                processed_text=cand["translation"],  # 用于显示和评估
                                raw_json=cand["raw_json"],  # 从翻译/润色结果 copy 的原始 JSON
                                scores=scores,
                                reasoning=reasoning,
                                suggestions=suggestions,
                                token_usage=TokenUsage(
                                    input_tokens=token_usage.input_tokens // len(candidates),
                                    output_tokens=token_usage.output_tokens // len(candidates),
                                    total_tokens=token_usage.total_tokens // len(candidates),
                                ),
                                metadata={"request_messages": req_messages, "response": content},
                            ))
                        else:
                            # 模型没有返回该候选的评分，使用默认值
                            results.append(EvaluationResult(
                                evaluator_model=model_name,
                                target_model=model_id,
                                target_type=cand["type"],
                                source_text=para_data.source_text,
                                processed_text=cand["translation"],  # 用于显示和评估
                                raw_json=cand["raw_json"],  # 从翻译/润色结果 copy 的原始 JSON
                                metadata={"error": f"No evaluation returned for {model_id}"},
                            ))

                    # 成功，跳出重试循环
                    break

                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Evaluation attempt {attempt + 1}/{max_retries} failed: "
                            f"{client.config.model_name}, retrying..."
                        )
                        continue
                    else:
                        logger.exception(
                            f"Comparative evaluation failed after {max_retries} attempts: "
                            f"{client.config.model_name}"
                        )
                        # 为所有候选创建错误结果
                        for cand in candidates:
                            results.append(EvaluationResult(
                                evaluator_model=client.config.model_name,
                                target_model=cand["model_id"],
                                target_type=cand["type"],
                                source_text=para_data.source_text,
                                processed_text=cand["translation"],  # 用于显示和评估
                                raw_json=cand["raw_json"],  # 从翻译/润色结果 copy 的原始 JSON
                                metadata={"error": str(last_error), "attempts": max_retries},
                            ))

            return para_data.paragraph_id, results

        if progress_monitor:
            with progress_monitor.stage_start(stage_name, total_paragraphs) as pbar:
                with PriorityThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(evaluate_paragraph_comparative, para_data, client): para_data
                        for para_data, client in all_tasks
                    }
                    for future in as_completed(futures):
                        para_id, results = future.result()
                        with self._lock:
                            para_evaluations[para_id].extend(results)
                            for result in results:
                                self._process_data.add_token_usage(result.evaluator_model, result.token_usage)
                            # 检查该段落是否完成所有评估任务
                            if para_id not in completed_paragraphs:
                                completed_paragraphs.add(para_id)
                                pbar.advance(1)
        else:
            with PriorityThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(evaluate_paragraph_comparative, para_data, client): para_data
                    for para_data, client in all_tasks
                }
                for future in as_completed(futures):
                    para_id, results = future.result()
                    with self._lock:
                        para_evaluations[para_id].extend(results)
                        for result in results:
                            self._process_data.add_token_usage(result.evaluator_model, result.token_usage)
                        if para_id not in completed_paragraphs:
                            completed_paragraphs.add(para_id)
                            logger.info(f"[Pipeline] 评估进度: {len(completed_paragraphs)}/{total_paragraphs}")

        # 将结果写回各段落
        for para_data in self._pending_paragraphs:
            para_data.evaluations = para_evaluations[para_data.paragraph_id]

    def _run_translations(self, text: str) -> list[TranslationResult]:
        """Run translation with all translator models in parallel.

        When used with ILTranslatorLLMOnly, the text parameter is already a complete
        prompt built by _build_llm_prompt, containing role instructions, contextual hints,
        glossary tables, and JSON input format. We should use it directly as the user message.

        Uses text hash based caching to avoid redundant API calls.
        """
        results: list[TranslationResult] = []

        if not self.translator_clients:
            return results

        def translate_with_model(client: ModelClient) -> TranslationResult:
            model_name = client.config.model_name

            # Check cache first
            cache = TranslationCache(model_name)
            cached = cache.get_translation(text)
            if cached:
                processed_text, raw_json = cached
                logger.debug(f"[Pipeline] Translation cache hit: {model_name}")
                return TranslationResult(
                    model_name=model_name,
                    source_text=text,
                    processed_text=processed_text,
                    raw_json=raw_json,
                    metadata={"from_cache": True},
                )

            # The text from ILTranslatorLLMOnly already contains the complete prompt
            # with all instructions, so we use it directly as user content
            messages = [
                {"role": "user", "content": text},
            ]
            
            # Stage 1.5: Fix Check - Retry if translation equals source text
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    content, token_usage, req_messages = client.chat(messages)
                    raw_json = strip_markdown_code_block(content)
                    processed_text = extract_actual_translation_text(raw_json)

                    # Check if translation is identical to source (ignoring whitespace)
                    # We need to extract actual source from the prompt to compare
                    actual_source = extract_actual_source_text(text)
                    if processed_text.strip() == actual_source.strip() and len(actual_source.strip()) > 5:
                        logger.warning(f"[Pipeline] Translation equals source (attempt {attempt+1}/{max_retries+1}): {model_name}")
                        if attempt < max_retries:
                            # Add a system message to encourage translation
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user", 
                                "content": "The translation is identical to the source text. Please translate it into the target language."
                            })
                            continue
                    
                    # Cache the result
                    if processed_text:
                        cache.set_translation(text, processed_text, raw_json)

                    return TranslationResult(
                        model_name=model_name,
                        source_text=text,
                        processed_text=processed_text,
                        raw_json=raw_json,
                        token_usage=token_usage,
                        metadata={"request_messages": req_messages, "response": content},
                    )
                except Exception as e:
                    logger.exception(f"Translation failed: {model_name}")
                    if attempt == max_retries:
                        return TranslationResult(
                            model_name=model_name,
                            source_text=text,
                            processed_text="",
                            raw_json="",
                            metadata={"error": str(e)},
                        )
            
            return TranslationResult(
                model_name=model_name,
                source_text=text,
                processed_text="",
                raw_json="",
                metadata={"error": "Max retries exceeded"},
            )

        with PriorityThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(translate_with_model, client): client
                for client in self.translator_clients
            }
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _run_polishes(
        self, source_text: str, translations: list[TranslationResult]
    ) -> list[PolishResult]:
        """Run polish on all translations with all polisher models."""
        results: list[PolishResult] = []

        if not self.polisher_clients:
            return results

        tasks: list[tuple[ModelClient, TranslationResult]] = []
        for client in self.polisher_clients:
            for trans in translations:
                if trans.processed_text:
                    tasks.append((client, trans))

        def polish_with_model(
            client: ModelClient, trans: TranslationResult
        ) -> PolishResult:
            # 润色时传入翻译器返回的原始 JSON
            input_json = trans.raw_json if trans.raw_json else trans.processed_text
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translation editor. "
                        f"Polish and improve the following {self.lang_out} translation. "
                        f"Fix any errors, improve fluency, but preserve the original meaning.\n\n"
                        f"## CRITICAL RULES:\n"
                        f"1. Input is a JSON array. Output MUST be a JSON array with the SAME number of elements.\n"
                        f"2. Each element MUST keep the same 'id' value from the input.\n"
                        f"3. Replace 'input' field with 'output' field containing your polished translation.\n"
                        f"4. Keep ALL <style id='N'>...</style> tags exactly as they appear.\n"
                        f"5. Keep ALL {{vN}} placeholders exactly as they appear.\n"
                        f"6. Only modify the actual translated text content, not the structure.\n\n"
                        f"## Example:\n"
                        f"Input: [{{'id': 0, 'output': 'text1'}}, {{'id': 1, 'output': 'text2'}}]\n"
                        f"Output: [{{'id': 0, 'output': 'polished1'}}, {{'id': 1, 'output': 'polished2'}}]"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original ({self.lang_in}):\n{source_text}\n\n"
                        f"Translation to polish (JSON array):\n{input_json}"
                    ),
                },
            ]
            try:
                content, token_usage, req_messages = client.chat(messages)
                raw_json = strip_markdown_code_block(content)
                processed_text = extract_actual_translation_text(raw_json)
                return PolishResult(
                    model_name=client.config.model_name,
                    source_text=source_text,
                    processed_text=processed_text,
                    from_translator=trans.model_name,
                    raw_json=raw_json,
                    token_usage=token_usage,
                    metadata={"request_messages": req_messages, "response": content},
                )
            except Exception as e:
                logger.exception(f"Polish failed: {client.config.model_name}")
                return PolishResult(
                    model_name=client.config.model_name,
                    source_text=source_text,
                    processed_text=trans.processed_text,
                    from_translator=trans.model_name,
                    raw_json=trans.raw_json,
                    metadata={"error": str(e)},
                )

        with PriorityThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(polish_with_model, client, trans): (client, trans)
                for client, trans in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _run_evaluations(
        self,
        source_text: str,
        translations: list[TranslationResult],
        polishes: list[PolishResult],
    ) -> list[EvaluationResult]:
        """Evaluate all translations and polishes."""
        results: list[EvaluationResult] = []

        if not self.evaluator_clients:
            return results

        # Build evaluation tasks: (client, source, translation, raw_json, target_model, target_type)
        tasks: list[tuple[ModelClient, str, str, str, str, str]] = []

        # Evaluate translations
        for client in self.evaluator_clients:
            for trans in translations:
                if trans.processed_text:
                    tasks.append(
                        (
                            client,
                            source_text,
                            trans.processed_text,
                            trans.raw_json,
                            trans.model_name,
                            "translator",
                        )
                    )

        # Evaluate polishes
        for client in self.evaluator_clients:
            for polish in polishes:
                if polish.processed_text:
                    # Use unique identifier for polished result
                    model_id = f"{polish.from_translator}+{polish.model_name}"
                    tasks.append(
                        (
                            client,
                            source_text,
                            polish.processed_text,
                            polish.raw_json,
                            model_id,
                            "polisher",
                        )
                    )

        def evaluate_with_model(
            client: ModelClient,
            source: str,
            translation: str,
            raw_json: str,
            target_model: str,
            target_type: str,
        ) -> EvaluationResult:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Evaluate the translation from {self.lang_in} to {self.lang_out}.\n"
                        "Score each dimension from 1-10:\n"
                        "- Accuracy: Meaning preservation\n"
                        "- Fluency: Natural expression\n"
                        "- Consistency: Terminology consistency\n"
                        "- Terminology: Technical term accuracy\n"
                        "- Completeness: No omissions\n\n"
                        'Output JSON: {"accuracy":N,"fluency":N,"consistency":N,'
                        '"terminology":N,"completeness":N,"reasoning":"..."}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Original:\n{source}\n\nTranslation:\n{translation}",
                },
            ]
            try:
                content, token_usage, req_messages = client.chat(
                    messages, response_format={"type": "json_object"}
                )
                scores, reasoning = self._parse_evaluation(content)

                # Track scores for model ranking
                with self._lock:
                    self._model_scores[target_model].append(scores.average)

                return EvaluationResult(
                    evaluator_model=client.config.model_name,
                    target_model=target_model,
                    target_type=target_type,
                    source_text=source,
                    processed_text=translation,
                    raw_json=raw_json,
                    scores=scores,
                    reasoning=reasoning,
                    token_usage=token_usage,
                    metadata={"request_messages": req_messages, "response": content},
                )
            except Exception as e:
                logger.exception(f"Evaluation failed: {client.config.model_name}")
                return EvaluationResult(
                    evaluator_model=client.config.model_name,
                    target_model=target_model,
                    target_type=target_type,
                    source_text=source,
                    processed_text=translation,
                    raw_json=raw_json,
                    metadata={"error": str(e)},
                )

        with PriorityThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    evaluate_with_model, client, src, trans, raw, model, ttype
                ): model
                for client, src, trans, raw, model, ttype in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _get_polish_score(
        self,
        polish: PolishResult,
        evaluations: list[EvaluationResult],
    ) -> float:
        """Get the evaluation score for a polish result."""
        for ev in evaluations:
            if (
                ev.target_type == "polisher"
                and ev.target_model == polish.model_name
                and ev.scores
            ):
                return ev.scores.average
        return 0.0

    def _parse_evaluation(self, content: str) -> tuple[EvaluationScores, str]:
        """Parse evaluation JSON response."""
        try:
            data = json.loads(content)
            scores = EvaluationScores(
                accuracy=float(data.get("accuracy", 5)),
                fluency=float(data.get("fluency", 5)),
                consistency=float(data.get("consistency", 5)),
                terminology=float(data.get("terminology", 5)),
                completeness=float(data.get("completeness", 5)),
            )
            return scores, data.get("reasoning", "")
        except json.JSONDecodeError:
            # Fallback regex parsing
            scores = EvaluationScores()
            for dim in ["accuracy", "fluency", "consistency", "terminology", "completeness"]:
                match = re.search(rf'"{dim}":\s*(\d+(?:\.\d+)?)', content, re.I)
                if match:
                    setattr(scores, dim, float(match.group(1)))
            return scores, content

    def _select_best(
        self,
        translations: list[TranslationResult],
        polishes: list[PolishResult],
        evaluations: list[EvaluationResult],
    ) -> tuple[str, str, str, str]:
        """Select best result based on evaluation scores.

        不区分 translator 和 polisher，直接根据评估分数选择最高分的结果。

        Returns (display_text, raw_json, model_name, model_type):
        - display_text: 用于报告显示的文本
        - raw_json: 用于更新 IL 文档的原始 JSON
        - model_name: 模型名称
        - model_type: "translator" 或 "polisher"
        """
        if not evaluations:
            # No evaluation, use first successful translation
            for t in translations:
                if t.processed_text:
                    return t.processed_text, t.raw_json, t.model_name, "translator"
            return "", "", "", ""

        # 直接从评估结果中找最高分，不区分 translator/polisher
        # 评估结果中已经包含了 processed_text 和 raw_json
        best_eval: EvaluationResult | None = None
        best_score = -1.0

        for ev in evaluations:
            if ev.scores and ev.processed_text:
                avg_score = ev.scores.average
                if avg_score > best_score:
                    best_score = avg_score
                    best_eval = ev

        if best_eval and best_eval.processed_text:
            best_text = best_eval.processed_text
            best_raw_json = best_eval.raw_json
            best_model = best_eval.target_model
            best_type = best_eval.target_type

            logger.info(
                f"[Pipeline] 选择最佳结果: model={best_model}, type={best_type}, "
                f"score={best_score:.2f}, text={best_text[:50]}..."
            )

            return best_text, best_raw_json, best_model, best_type

        # Fallback to first translator result
        for t in translations:
            if t.processed_text:
                return t.processed_text, t.raw_json, t.model_name, "translator"
        return "", "", "", ""

    def get_process_data(self) -> PipelineProcessData:
        """Get complete process data."""
        # Update final model scores
        for model, scores in self._model_scores.items():
            self._process_data.model_scores[model] = (
                sum(scores) / len(scores) if scores else 0.0
            )
        return self._process_data

    def save_process_data(self, output_dir: Path | str | None = None) -> str | None:
        """Save process data to JSON file."""
        save_dir = Path(output_dir) if output_dir else self.output_dir
        if not save_dir:
            return None

        save_dir.mkdir(parents=True, exist_ok=True)
        json_path = save_dir / "pipeline_process_data.json"
        process_data = self.get_process_data()
        logger.info(f"[Pipeline] save_process_data: total_paragraphs={process_data.total_paragraphs}, "
                    f"paragraphs_count={len(process_data.paragraphs)}, "
                    f"pending_paragraphs={len(self._pending_paragraphs)}")
        process_data.save_to_json(json_path)
        logger.info(f"Process data saved to {json_path}")
        return str(json_path)

    def generate_report(self, output_dir: Path | str | None = None, original_filename: str = "report") -> str | None:
        """Generate evaluation report in HTML format.

        The HTML report contains:
        - Summary table with average scores for each model across all evaluators
        - Detailed evaluation results for each paragraph
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        if not save_dir:
            return None

        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct filename: {original_filename}-{lang_in}-{lang_out}.html
        # Remove extension from original_filename if present
        base_name = Path(original_filename).stem
        report_name = f"{base_name}-{self.pipeline_config.lang_in}-{self.pipeline_config.lang_out}.html"
        report_path = save_dir / report_name

        data = self.get_process_data()
        html_path = generate_evaluation_report_html(data, report_path)

        logger.info(f"HTML report saved to {html_path}")
        return html_path

    def update_docs_with_best_results(self, docs: Any, il_translator: Any = None) -> int:
        """
        更新 IL 文档中的翻译结果为评估后选择的最佳结果。

        当选择的是 polisher 的结果时，使用 il_translator 重新解析润色后的文本，
        正确设置 pdf_paragraph_composition。

        Args:
            docs: IL Document 对象，包含 page 列表，每个 page 包含 pdf_paragraph 列表
            il_translator: ILTranslator 或 ILTranslatorLLMOnly 实例，用于解析翻译输出

        Returns:
            更新的段落数量
        """
        paragraphs_data = self._process_data.paragraphs
        if not paragraphs_data:
            logger.warning("[Pipeline] No paragraphs data found, skipping update")
            return 0

        # 获取 il_translator 的解析方法
        # ILTranslatorLLMOnly 有一个 il_translator 属性指向 ILTranslator 实例
        parser = None
        if il_translator:
            if hasattr(il_translator, 'il_translator'):
                # ILTranslatorLLMOnly
                parser = il_translator.il_translator
            elif hasattr(il_translator, 'parse_translate_output'):
                # ILTranslator
                parser = il_translator

        if not parser:
            logger.warning("[Pipeline] 无法获取 il_translator 解析器，跳过更新")
            return 0

        # 构建 debug_id 到段落对象的映射
        debug_id_to_paragraph: dict[str, Any] = {}
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph.debug_id:
                    debug_id_to_paragraph[paragraph.debug_id] = paragraph

        logger.info(
            f"[Pipeline] update_docs_with_best_results: "
            f"{len(paragraphs_data)} 个批次, {len(debug_id_to_paragraph)} 个 IL 段落"
        )

        updated_count = 0
        skipped_count = 0

        # 遍历每个批次
        for para_data in paragraphs_data:
            # 只有当选择了 polisher 的结果时才更新
            if para_data.selected_type != "polisher":
                continue

            if not para_data.selected_raw_text:
                continue

            if not para_data.batch_id_to_debug_id:
                skipped_count += 1
                continue

            try:
                parsed = json.loads(para_data.selected_raw_text)
                if isinstance(parsed, dict):
                    parsed = [parsed]

                for item in parsed:
                    item_id = int(item.get("id", -1))
                    output_text = item.get("output", item.get("input", ""))

                    if item_id < 0 or not output_text:
                        continue

                    # 通过 batch_id_to_debug_id 找到对应的段落 debug_id
                    debug_id = para_data.batch_id_to_debug_id.get(item_id)
                    if not debug_id:
                        logger.warning(f"[Pipeline] 批次 {para_data.paragraph_id} 中 id={item_id} 没有对应的 debug_id")
                        continue

                    # 找到对应的段落对象
                    paragraph = debug_id_to_paragraph.get(debug_id)
                    if not paragraph:
                        logger.warning(f"[Pipeline] 找不到 debug_id={debug_id} 对应的段落")
                        continue

                    # 获取保存的 translate_input
                    translate_input = self._pipeline_translate_inputs.get(debug_id)
                    if not translate_input:
                        logger.warning(f"[Pipeline] 找不到 debug_id={debug_id} 的 translate_input")
                        continue

                    # 使用 il_translator 重新解析润色后的文本
                    paragraph.unicode = output_text
                    paragraph.pdf_paragraph_composition = parser.parse_translate_output(
                        translate_input,
                        output_text,
                        None,  # tracker
                        None,  # llm_translate_tracker
                    )
                    # 设置默认样式
                    for composition in paragraph.pdf_paragraph_composition:
                        if (
                            composition.pdf_same_style_unicode_characters
                            and composition.pdf_same_style_unicode_characters.pdf_style is None
                        ):
                            composition.pdf_same_style_unicode_characters.pdf_style = (
                                paragraph.pdf_style
                            )
                    updated_count += 1
                    logger.debug(
                        f"[Pipeline] 更新段落 debug_id={debug_id} (model={para_data.selected_model})"
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"[Pipeline] 无法解析 JSON: {e}")
                continue

        if skipped_count > 0:
            logger.warning(f"[Pipeline] {skipped_count} 个批次缺少 batch_id_to_debug_id 映射")

        logger.info(f"[Pipeline] 更新完成: {updated_count} 个段落已更新为润色结果")
        return updated_count

    # Placeholder methods required by BaseTranslator interface
    def get_formular_placeholder(self, placeholder_id: int | str):
        return "{v" + str(placeholder_id) + "}", rf"{{\s*v\s*{placeholder_id}\s*}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int | str):
        return (
            f"<style id='{placeholder_id}'>",
            rf"<\s*style\s*id\s*=\s*'\s*{placeholder_id}\s*'\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int | str):
        return "</style>", r"<\s*\/\s*style\s*>"
