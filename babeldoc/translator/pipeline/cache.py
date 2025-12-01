"""
Pipeline Cache - Text hash based caching for translation and evaluation.

Uses normalized text hash as cache key to avoid redundant API calls.
"""

import hashlib
import json
import logging
import random
import re
import threading

import peewee
from peewee import SQL
from peewee import AutoField
from peewee import CharField
from peewee import Model
from peewee import SqliteDatabase
from peewee import TextField
from peewee import fn

from babeldoc.const import CACHE_FOLDER

logger = logging.getLogger(__name__)

# Database instance - initialized lazily
_db = SqliteDatabase(None)
_db_initialized = False
_db_lock = threading.Lock()

# Cleanup configuration
CLEAN_PROBABILITY = 0.001  # 0.1% chance to trigger cleanup
MAX_CACHE_ROWS = 100_000  # Keep only the latest 100,000 rows
_cleanup_lock = threading.Lock()


def _normalize_text(text: str) -> str:
    """
    Normalize text for hashing by removing extra whitespace and newlines.

    This ensures that texts with different formatting but same content
    produce the same hash.
    """
    if not text:
        return ""
    # Replace multiple whitespace/newlines with single space
    normalized = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return normalized.strip()


def _compute_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text."""
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _PipelineCache(Model):
    """Database model for pipeline cache entries."""

    id = AutoField()
    cache_type = CharField(max_length=20)  # translation, polish, evaluation
    model_name = CharField(max_length=100)
    text_hash = CharField(max_length=64)  # SHA256 hash
    extra_key = TextField(default="")  # Additional key info (e.g., from_translator for polish)
    result_json = TextField()  # JSON serialized result

    class Meta:
        database = _db
        constraints = [
            SQL(
                """
            UNIQUE (
                cache_type,
                model_name,
                text_hash,
                extra_key
                )
            ON CONFLICT REPLACE
            """,
            ),
        ]


def _ensure_db_initialized():
    """Ensure the database is initialized (lazy initialization)."""
    global _db_initialized
    if _db_initialized:
        return

    with _db_lock:
        if _db_initialized:
            return

        CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
        cache_db_path = CACHE_FOLDER / "pipeline_cache.v1.db"
        logger.info(f"Initializing pipeline cache database at {cache_db_path}")

        _db.init(
            cache_db_path,
            pragmas={
                "journal_mode": "wal",
                "busy_timeout": 1000,
            },
        )
        _db.create_tables([_PipelineCache], safe=True)
        _db_initialized = True


def _maybe_cleanup():
    """Trigger cache cleanup with a small probability."""
    if random.random() >= CLEAN_PROBABILITY:  # noqa: S311
        return

    if not _cleanup_lock.acquire(blocking=False):
        return

    try:
        logger.info("Cleaning up pipeline cache...")
        max_id = _PipelineCache.select(fn.MAX(_PipelineCache.id)).scalar()
        if not max_id or max_id <= MAX_CACHE_ROWS:
            return
        threshold = max_id - MAX_CACHE_ROWS
        _PipelineCache.delete().where(_PipelineCache.id <= threshold).execute()
    finally:
        _cleanup_lock.release()


class PipelineCache:
    """
    Cache for pipeline translation, polish, and evaluation results.

    Uses text hash as key to avoid redundant API calls for same content.
    """

    def __init__(self, cache_type: str, model_name: str):
        """
        Initialize cache for a specific operation type and model.

        Args:
            cache_type: One of "translation", "polish", "evaluation"
            model_name: Name of the model being used
        """
        self.cache_type = cache_type
        self.model_name = model_name

    def get(
        self,
        source_text: str,
        extra_key: str = "",
    ) -> dict | None:
        """
        Get cached result for the given source text.

        Args:
            source_text: The source text to look up
            extra_key: Additional key info (e.g., from_translator for polish)

        Returns:
            Cached result dict or None if not found
        """
        _ensure_db_initialized()

        text_hash = _compute_hash(source_text)

        try:
            result = _PipelineCache.get_or_none(
                cache_type=self.cache_type,
                model_name=self.model_name,
                text_hash=text_hash,
                extra_key=extra_key,
            )
            if result:
                _maybe_cleanup()
                return json.loads(result.result_json)
            return None
        except peewee.OperationalError as e:
            if "database is locked" in str(e):
                logger.debug("Pipeline cache is locked")
                return None
            raise

    def set(
        self,
        source_text: str,
        result: dict,
        extra_key: str = "",
    ):
        """
        Cache a result for the given source text.

        Args:
            source_text: The source text as key
            result: Result dict to cache
            extra_key: Additional key info (e.g., from_translator for polish)
        """
        _ensure_db_initialized()

        text_hash = _compute_hash(source_text)

        try:
            _PipelineCache.create(
                cache_type=self.cache_type,
                model_name=self.model_name,
                text_hash=text_hash,
                extra_key=extra_key,
                result_json=json.dumps(result, ensure_ascii=False),
            )
            _maybe_cleanup()
        except peewee.OperationalError as e:
            if "database is locked" in str(e):
                logger.debug("Pipeline cache is locked")
            else:
                raise


class TranslationCache(PipelineCache):
    """Cache for translation results."""

    def __init__(self, model_name: str):
        super().__init__("translation", model_name)

    def get_translation(self, source_text: str) -> tuple[str, str] | None:
        """
        Get cached translation.

        Returns:
            Tuple of (processed_text, raw_json) or None if not found
        """
        result = self.get(source_text)
        if result:
            return result.get("processed_text", ""), result.get("raw_json", "")
        return None

    def set_translation(self, source_text: str, processed_text: str, raw_json: str):
        """Cache a translation result."""
        self.set(source_text, {"processed_text": processed_text, "raw_json": raw_json})


class PolishCache(PipelineCache):
    """Cache for polish results."""

    def __init__(self, model_name: str):
        super().__init__("polish", model_name)

    def get_polish(
        self,
        source_text: str,
        translation_text: str,
        from_translator: str,
    ) -> tuple[str, str] | None:
        """
        Get cached polish result.

        Args:
            source_text: Original source text
            translation_text: Translation to be polished
            from_translator: Name of the translator that produced the translation

        Returns:
            Tuple of (polished_text, raw_json) or None if not found
        """
        # Use combined hash of source + translation as key
        combined_text = f"{source_text}\n---TRANSLATION---\n{translation_text}"
        result = self.get(combined_text, extra_key=from_translator)
        if result:
            return result.get("processed_text", ""), result.get("raw_json", "")
        return None

    def set_polish(
        self,
        source_text: str,
        translation_text: str,
        from_translator: str,
        polished_text: str,
        raw_json: str,
    ):
        """Cache a polish result."""
        combined_text = f"{source_text}\n---TRANSLATION---\n{translation_text}"
        self.set(
            combined_text,
            {"processed_text": polished_text, "raw_json": raw_json},
            extra_key=from_translator,
        )


class EvaluationCache(PipelineCache):
    """Cache for evaluation results."""

    def __init__(self, model_name: str):
        super().__init__("evaluation", model_name)

    def get_evaluation(
        self,
        source_text: str,
        candidates_key: str,
    ) -> dict | None:
        """
        Get cached evaluation result.

        Args:
            source_text: Original source text
            candidates_key: Key representing the set of candidates being evaluated
                           (e.g., sorted list of model IDs)

        Returns:
            Cached evaluation dict or None if not found
        """
        result = self.get(source_text, extra_key=candidates_key)
        return result

    def set_evaluation(
        self,
        source_text: str,
        candidates_key: str,
        evaluation_result: dict,
    ):
        """Cache an evaluation result."""
        self.set(source_text, evaluation_result, extra_key=candidates_key)
