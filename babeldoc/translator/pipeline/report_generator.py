"""
HTML Report Generator for Translation Pipeline Evaluation.

Generates an HTML report with:
- Summary table with average scores for each model across all evaluators
- Detailed evaluation results for each paragraph
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from babeldoc.translator.pipeline.models import EvaluationScores
from babeldoc.translator.pipeline.models import PipelineProcessData

logger = logging.getLogger(__name__)


def strip_markdown_code_block(text: str) -> str:
    """
    Strip markdown code block formatting from text.

    Removes patterns like:
    - ```json ... ```
    - ```\n ... ```
    - ``` ... ```
    """
    if not text:
        return ""

    text = text.strip()

    # Pattern: ```json\n...\n``` or ```\n...\n```
    if text.startswith("```"):
        # Find the end of the first line (might be ```json or just ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            # Remove the opening ``` or ```json line
            text = text[first_newline + 1:]

        # Remove trailing ```
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    return text


def extract_actual_source_text(source_text: str) -> str:
    """
    Extract actual source text from the full prompt format.

    The source_text might be a full prompt in one of these formats:
    1. Contains "## Here is the input:" followed by a JSON array with "input" fields
    2. Contains "Now translate the following text:" followed by the actual text
    3. Starts with "You are a professional" (translation prompt)

    This function extracts and returns the actual source texts.
    """
    if not source_text:
        return ""

    # Format 1: "## Here is the input:" with JSON array
    marker1 = "## Here is the input:"
    if marker1 in source_text:
        try:
            json_start = source_text.find(marker1)
            json_part = source_text[json_start + len(marker1):].strip()

            # Find the JSON array
            array_start = json_part.find("[")
            if array_start != -1:
                # Find the matching closing bracket
                bracket_count = 0
                array_end = -1
                for i, char in enumerate(json_part[array_start:], start=array_start):
                    if char == "[":
                        bracket_count += 1
                    elif char == "]":
                        bracket_count -= 1
                        if bracket_count == 0:
                            array_end = i + 1
                            break

                if array_end != -1:
                    json_array_str = json_part[array_start:array_end]
                    data = json.loads(json_array_str)

                    # Extract "input" fields from each item
                    inputs = []
                    for item in data:
                        if isinstance(item, dict) and "input" in item:
                            input_text = item["input"]
                            # Remove style tags for cleaner display
                            clean_text = re.sub(r"<style[^>]*>|</style>", "", input_text)
                            # Remove placeholders like {v3}
                            clean_text = re.sub(r"\{v\d+\}", "", clean_text)
                            inputs.append(clean_text.strip())

                    if inputs:
                        return " ".join(inputs)

        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            pass

    # Format 2: "Now translate the following text:" marker
    marker2 = "Now translate the following text:"
    if marker2 in source_text:
        idx = source_text.find(marker2)
        actual_text = source_text[idx + len(marker2):].strip()
        if actual_text:
            return actual_text

    # Format 3: Check if it looks like a translation prompt (starts with role description)
    # This catches prompts that start with "You are a professional..."
    if source_text.strip().startswith("You are a professional"):
        # Try to find the text after "## Output" section
        output_marker = "## Output"
        if output_marker in source_text:
            # Look for text after the output instructions
            output_idx = source_text.find(output_marker)
            after_output = source_text[output_idx:]

            # Find "Now translate the following text:" after ## Output
            if marker2 in after_output:
                translate_idx = after_output.find(marker2)
                actual_text = after_output[translate_idx + len(marker2):].strip()
                if actual_text:
                    return actual_text

    return source_text


def extract_actual_translation_text(translated_text: str) -> str:
    """
    Extract actual translation text from JSON output format.

    The translated_text might be a JSON array or object with "output" fields.
    This function extracts and concatenates the actual translation texts.
    """
    if not translated_text:
        return ""

    text = translated_text.strip()

    # Try to parse as JSON
    try:
        # Check if it starts with [ (JSON array) or { (JSON object)
        if not text.startswith("[") and not text.startswith("{"):
            return translated_text

        data = json.loads(text)

        # Handle single object case (e.g., {"id": 0, "output": "..."})
        if isinstance(data, dict):
            if "output" in data:
                output_text = data["output"]
                # Remove style tags for cleaner display
                clean_text = re.sub(r"<style[^>]*>|</style>", "", output_text)
                # Remove placeholders like {v3}
                clean_text = re.sub(r"\{v\d+\}", "", clean_text)
                return clean_text.strip()
            return translated_text

        # Handle array case (e.g., [{"id": 0, "output": "..."}, ...])
        if isinstance(data, list):
            outputs = []
            for item in data:
                if isinstance(item, dict) and "output" in item:
                    output_text = item["output"]
                    # Remove style tags for cleaner display
                    clean_text = re.sub(r"<style[^>]*>|</style>", "", output_text)
                    # Remove placeholders like {v3}
                    clean_text = re.sub(r"\{v\d+\}", "", clean_text)
                    outputs.append(clean_text.strip())

            if outputs:
                return " ".join(outputs)

    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.debug(f"Failed to parse JSON in extract_actual_translation_text: {e}")

    # If JSON parsing succeeded but no output found, or parsing failed,
    # try a regex-based extraction as fallback
    try:
        # Try to extract "output" values using regex
        # This handles cases where json.loads failed or structure was unexpected
        # Pattern matches both "output": "..." and 'output': '...'
        output_pattern = r'["\']output["\']\s*:\s*"((?:[^"\\]|\\.)*)"|["\']output["\']\s*:\s*\'((?:[^\'\\]|\\.)*)\''
        matches = re.findall(output_pattern, text)
        if matches:
            outputs = []
            for match in matches:
                # match is a tuple (double_quoted, single_quoted)
                output_text = match[0] if match[0] else match[1]
                # Unescape JSON string
                try:
                    output_text = json.loads(f'"{output_text}"')
                except json.JSONDecodeError:
                    pass
                # Remove style tags for cleaner display
                clean_text = re.sub(r"<style[^>]*>|</style>", "", output_text)
                # Remove placeholders like {v3}
                clean_text = re.sub(r"\{v\d+\}", "", clean_text)
                if clean_text.strip():
                    outputs.append(clean_text.strip())
            if outputs:
                return " ".join(outputs)
    except Exception as e:
        logger.debug(f"Regex fallback failed in extract_actual_translation_text: {e}")

    return translated_text


def extract_raw_translation_output(translated_text: str) -> str:
    """
    Extract raw output from JSON format WITHOUT cleaning style tags or placeholders.

    This is used to match what ILTranslatorLLMOnly stores in paragraph.unicode.
    The only transformation applied is extracting from JSON and removing excessive punctuation.
    """
    if not translated_text:
        return ""

    # Try to parse as JSON
    try:
        text = translated_text.strip()
        # Check if it starts with [ (JSON array) or { (JSON object)
        if not text.startswith("[") and not text.startswith("{"):
            return translated_text

        data = json.loads(text)

        # Handle single object case (e.g., {"id": 0, "output": "..."})
        if isinstance(data, dict):
            if "output" in data:
                output_text = data["output"]
                # Only remove excessive punctuation
                output_text = re.sub(r"[. 。…，]{20,}", ".", output_text)
                return output_text
            return translated_text

        # Handle array case (e.g., [{"id": 0, "output": "..."}, ...])
        if isinstance(data, list):
            outputs = []
            for item in data:
                if isinstance(item, dict) and "output" in item:
                    output_text = item["output"]
                    # Only remove excessive punctuation (matching il_translator_llm_only.py)
                    output_text = re.sub(r"[. 。…，]{20,}", ".", output_text)
                    outputs.append(output_text)

            if outputs:
                # For single item, return as-is; for multiple items, join with newline
                # This matches how ILTranslatorLLMOnly handles batch translations
                return outputs[0] if len(outputs) == 1 else "\n".join(outputs)

    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    return translated_text


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _score_color(score: float) -> str:
    """Get color based on score (green=good, red=bad)."""
    if score >= 8.0:
        return "#28a745"  # Green
    elif score >= 6.0:
        return "#ffc107"  # Yellow/Orange
    else:
        return "#dc3545"  # Red


def _aggregate_model_scores(
    data: PipelineProcessData,
    model_mapping: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Aggregate evaluation scores by target model.

    Args:
        data: Pipeline process data
        model_mapping: Optional mapping from model names to aliases for anonymization

    Returns:
        {display_name: {'type': 'Translator'|'Polisher', 'scores': EvaluationScores}}
    """
    # Structure: {display_name: {'type': str, 'dims': {dim: [scores]}}}
    raw_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"type": "", "dims": defaultdict(list)}
    )

    for para in data.paragraphs:
        for ev in para.evaluations:
            if ev.scores:
                target = ev.target_model
                # Determine type from target_type
                model_type = "Translator" if ev.target_type == "translator" else "Polisher"

                # Convert unique ID to display name
                # "translator:gpt-4o-mini" -> "gpt-4o-mini"
                # "polisher:deepseek-chat+gpt-4o-mini" -> "deepseek-chat→gpt-4o-mini"
                if target.startswith("translator:"):
                    display_name = target[len("translator:"):]
                elif target.startswith("polisher:"):
                    # Remove prefix and convert + to →
                    display_name = target[len("polisher:"):].replace("+", "→")
                else:
                    # Fallback for old format
                    display_name = target

                # Anonymize if mapping provided
                if model_mapping:
                    display_name = _anonymize_model_name(display_name, model_mapping)

                raw_data[display_name]["type"] = model_type
                raw_data[display_name]["dims"]["accuracy"].append(ev.scores.accuracy)
                raw_data[display_name]["dims"]["fluency"].append(ev.scores.fluency)
                raw_data[display_name]["dims"]["consistency"].append(ev.scores.consistency)
                raw_data[display_name]["dims"]["terminology"].append(ev.scores.terminology)
                raw_data[display_name]["dims"]["completeness"].append(ev.scores.completeness)

    # Calculate averages
    result: dict[str, dict[str, Any]] = {}
    for target, model_data in raw_data.items():
        dims = model_data["dims"]
        scores = EvaluationScores(
            accuracy=sum(dims["accuracy"]) / len(dims["accuracy"]) if dims["accuracy"] else 0,
            fluency=sum(dims["fluency"]) / len(dims["fluency"]) if dims["fluency"] else 0,
            consistency=sum(dims["consistency"]) / len(dims["consistency"]) if dims["consistency"] else 0,
            terminology=sum(dims["terminology"]) / len(dims["terminology"]) if dims["terminology"] else 0,
            completeness=sum(dims["completeness"]) / len(dims["completeness"]) if dims["completeness"] else 0,
        )
        result[target] = {"type": model_data["type"], "scores": scores}

    return result


def _ensure_clean_translation(text: str) -> str:
    """
    Ensure translation text is clean for display.

    If the text still looks like JSON (starts with [ or {), try to extract
    the actual translation again. This is a safeguard for cases where
    the initial extraction failed.
    """
    if not text:
        return ""

    text = text.strip()

    # If it looks like JSON, try to extract again
    if text.startswith("[") or text.startswith("{"):
        extracted = extract_actual_translation_text(text)
        # Only use extracted if it's different and not still JSON
        if extracted != text and not extracted.strip().startswith("[") and not extracted.strip().startswith("{"):
            return extracted

    return text


def _collect_model_results(
    para,
    model_mapping: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Collect translation/polish results with their scores for a paragraph."""
    results = []

    # Get evaluation scores for each model
    # The key is the unique ID used in evaluation (e.g., "translator:gpt-4o-mini")
    model_scores: dict[str, float] = {}
    for ev in para.evaluations:
        if ev.scores:
            model_scores[ev.target_model] = ev.scores.average

    # Add translations
    for trans in para.translations:
        if trans.processed_text:
            # Match the unique ID format used in evaluation
            unique_id = f"translator:{trans.model_name}"
            # Ensure clean translation text for display
            clean_translation = _ensure_clean_translation(trans.processed_text)
            # Anonymize model name if mapping provided
            display_name = trans.model_name
            if model_mapping:
                display_name = _anonymize_model_name(display_name, model_mapping)
            results.append({
                "model": display_name,
                "role": "Translator",
                "translation": clean_translation,
                "score": model_scores.get(unique_id, model_scores.get(trans.model_name, 0)),
            })

    # Add polishes - each polish comes from a specific translator
    for polish in para.polishes:
        if polish.processed_text:
            # Match the unique ID format used in evaluation
            unique_id = f"polisher:{polish.from_translator}+{polish.model_name}"
            # Display name shows the translation source
            display_name = f"{polish.from_translator}→{polish.model_name}"
            # Anonymize if mapping provided
            if model_mapping:
                display_name = _anonymize_model_name(display_name, model_mapping)
            # Ensure clean translation text for display
            clean_translation = _ensure_clean_translation(polish.processed_text)
            results.append({
                "model": display_name,
                "role": "Polisher",
                "translation": clean_translation,
                "score": model_scores.get(unique_id, model_scores.get(polish.model_name, 0)),
            })

    return results


def _create_model_alias_mapping(process_data: PipelineProcessData) -> dict[str, str]:
    """
    Create a mapping from model names to anonymous aliases (A, B, C, D, ...).

    Returns:
        Dict mapping original model names to aliases (e.g., {"gpt-4o-mini": "A", "deepseek-chat": "B"})
    """
    # Collect all unique model names from translations, polishes, and evaluations
    model_names: set[str] = set()

    for para in process_data.paragraphs:
        # Translator models
        for trans in para.translations:
            if trans.model_name:
                model_names.add(trans.model_name)
        # Polisher models
        for polish in para.polishes:
            if polish.model_name:
                model_names.add(polish.model_name)
            if polish.from_translator:
                model_names.add(polish.from_translator)
        # Evaluator models
        for ev in para.evaluations:
            if ev.evaluator_model:
                model_names.add(ev.evaluator_model)

    # Sort for consistent ordering
    sorted_names = sorted(model_names)

    # Create mapping: A, B, C, ..., Z, AA, AB, ...
    mapping: dict[str, str] = {}
    for i, name in enumerate(sorted_names):
        if i < 26:
            alias = chr(ord('A') + i)
        else:
            # For more than 26 models, use AA, AB, etc.
            alias = chr(ord('A') + (i // 26) - 1) + chr(ord('A') + (i % 26))
        mapping[name] = alias

    return mapping


def _anonymize_model_name(name: str, mapping: dict[str, str]) -> str:
    """
    Convert a model name or compound name to its anonymous form.

    Handles:
    - Simple names: "gpt-4o-mini" -> "A"
    - Arrow names: "gpt-4o-mini→deepseek-chat" -> "A→B"
    - Plus names: "gpt-4o-mini+deepseek-chat" -> "A+B"
    """
    if not name:
        return name

    # Handle arrow notation (polisher display names)
    if "→" in name:
        parts = name.split("→")
        return "→".join(mapping.get(p, p) for p in parts)

    # Handle plus notation (internal polisher IDs)
    if "+" in name:
        parts = name.split("+")
        return "+".join(mapping.get(p, p) for p in parts)

    return mapping.get(name, name)


def generate_evaluation_report_html(
    process_data: PipelineProcessData,
    output_path: Path | str,
) -> str:
    """
    Generate HTML evaluation report with anonymized model names.

    Model names are replaced with aliases (A, B, C, ...) for blind evaluation.
    A separate JSON file records the mapping from aliases to actual model names.

    Args:
        process_data: Pipeline process data with evaluation results
        output_path: Path to save the HTML report

    Returns:
        Path to the generated HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create model alias mapping for anonymization
    model_mapping = _create_model_alias_mapping(process_data)

    # Save mapping to JSON file
    mapping_path = output_path.parent / "model_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        # Save as {alias: model_name} for easier lookup
        reverse_mapping = {v: k for k, v in model_mapping.items()}
        json.dump(reverse_mapping, f, ensure_ascii=False, indent=2)
    logger.info(f"Model mapping saved to {mapping_path}")

    # Build aggregated scores with anonymized names
    model_summary = _aggregate_model_scores(process_data, model_mapping)

    # Generate HTML with anonymized names
    html_content = _generate_html(process_data, model_summary, model_mapping)

    # Write to file
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"HTML report saved to {output_path}")
    return str(output_path)


def _generate_html(
    process_data: PipelineProcessData,
    model_summary: dict[str, dict[str, Any]],
    model_mapping: dict[str, str] | None = None,
) -> str:
    """Generate the complete HTML content with anonymized model names."""

    # CSS styles
    css = """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c5282;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #2c5282;
        }
        h2 {
            color: #2c5282;
            margin: 30px 0 15px 0;
            font-size: 1.3em;
        }
        h3 {
            color: #4a5568;
            margin: 20px 0 10px 0;
            font-size: 1.1em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border: 1px solid #e2e8f0;
        }
        th {
            background-color: #2c5282;
            color: white;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f7fafc;
        }
        tr:hover {
            background-color: #edf2f7;
        }
        .summary-table th {
            text-align: center;
        }
        .summary-table td {
            text-align: center;
        }
        .summary-table td:first-child {
            text-align: left;
            font-weight: 500;
        }
        .overall-row {
            background-color: #ebf8ff !important;
            font-weight: bold;
        }
        .model-header {
            font-size: 13px;
        }
        .model-type {
            font-size: 11px;
            color: #a0aec0;
            font-weight: normal;
        }
        .segment {
            margin: 25px 0;
            padding: 20px;
            background-color: #f7fafc;
            border-radius: 6px;
            border-left: 4px solid #2c5282;
        }
        .segment-header {
            color: #2c5282;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .source-text {
            background-color: #fff;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            border: 1px solid #e2e8f0;
            color: #4a5568;
            font-size: 14px;
        }
        .source-label {
            font-weight: 600;
            color: #718096;
            margin-bottom: 5px;
            font-size: 13px;
        }
        .translation-cell {
            max-width: 400px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        .score {
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            display: inline-block;
        }
        .score-good {
            background-color: #c6f6d5;
            color: #22543d;
        }
        .score-medium {
            background-color: #fefcbf;
            color: #744210;
        }
        .score-bad {
            background-color: #fed7d7;
            color: #822727;
        }
        .evaluation-details {
            margin-top: 15px;
            padding: 10px;
            background-color: #fff;
            border-radius: 4px;
            font-size: 13px;
            color: #718096;
        }
        .evaluation-item {
            margin: 5px 0;
            padding: 5px 0;
            border-bottom: 1px solid #e2e8f0;
        }
        .evaluation-item:last-child {
            border-bottom: none;
        }
        .role-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .role-translator {
            background-color: #bee3f8;
            color: #2a4365;
        }
        .role-polisher {
            background-color: #c6f6d5;
            color: #22543d;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .stat-item {
            background-color: #ebf8ff;
            padding: 15px 20px;
            border-radius: 6px;
            min-width: 150px;
        }
        .stat-label {
            font-size: 12px;
            color: #4a5568;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c5282;
        }
        .cover-header {
            text-align: center;
            padding: 20px 0 30px 0;
            margin-bottom: 20px;
            border-bottom: 1px solid #e2e8f0;
        }
        .cover-header .brand {
            font-size: 16px;
            color: #333;
            margin-bottom: 8px;
        }
        .cover-header .brand strong {
            font-weight: 700;
        }
        .cover-header .website {
            font-size: 14px;
            color: #4a5568;
            margin-bottom: 8px;
        }
        .cover-header .website a {
            color: #2c5282;
            text-decoration: none;
        }
        .cover-header .website a:hover {
            text-decoration: underline;
        }
        .cover-header .disclaimer {
            font-size: 13px;
            color: #718096;
            font-style: italic;
        }
    </style>
    """

    # Build summary table
    summary_html = _build_summary_table(model_summary)

    # Build statistics (with anonymized best model name)
    stats_html = _build_stats(process_data, model_mapping)

    # Build detail sections (with anonymized model names)
    details_html = _build_detail_sections(process_data, model_mapping)

    # Complete HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Translation Evaluation Report</title>
    {css}
</head>
<body>
    <div class="container">
        <div class="cover-header">
            <div class="brand">此文件由<strong>译曲同工</strong>提供翻译服务</div>
            <div class="website">更多信息请访问 <a href="https://aitranspro.com" target="_blank">aitranspro.com</a></div>
            <div class="disclaimer">——内容仅供内部评估与试阅——</div>
        </div>

        <h1>Translation Evaluation Report</h1>

        {stats_html}

        <h2>Summary: Average Scores by Model</h2>
        {summary_html}

        <h2>Detailed Results</h2>
        {details_html}
    </div>
</body>
</html>
"""
    return html


def _build_stats(
    process_data: PipelineProcessData,
    model_mapping: dict[str, str] | None = None,
) -> str:
    """Build statistics section with anonymized model names."""
    total_paragraphs = len(process_data.paragraphs)
    total_tokens = process_data.total_token_usage.total_tokens

    # Find best model and anonymize if mapping provided
    best_model = process_data.get_best_model() or "N/A"
    if model_mapping and best_model != "N/A":
        # Handle prefixed model IDs like "translator:xxx" or "polisher:xxx+yyy"
        if best_model.startswith("translator:"):
            display_name = best_model[len("translator:"):]
            best_model = _anonymize_model_name(display_name, model_mapping)
        elif best_model.startswith("polisher:"):
            display_name = best_model[len("polisher:"):].replace("+", "→")
            best_model = _anonymize_model_name(display_name, model_mapping)
        else:
            best_model = _anonymize_model_name(best_model, model_mapping)
    best_score = max(process_data.model_scores.values()) if process_data.model_scores else 0

    return f"""
    <div class="stats">
        <div class="stat-item">
            <div class="stat-label">Total Paragraphs</div>
            <div class="stat-value">{total_paragraphs}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Total Tokens</div>
            <div class="stat-value">{total_tokens:,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Best Model</div>
            <div class="stat-value" style="font-size: 16px;">{_escape_html(best_model)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Best Score</div>
            <div class="stat-value">{best_score:.2f}</div>
        </div>
    </div>
    """


def _build_summary_table(model_summary: dict[str, dict[str, Any]]) -> str:
    """Build the summary scores table."""
    if not model_summary:
        return "<p>No evaluation data available.</p>"

    models = list(model_summary.keys())
    dimensions = ["Accuracy", "Fluency", "Consistency", "Terminology", "Completeness"]

    # Header row
    header_cells = ["<th>Metric / Model</th>"]
    for model in models:
        model_type = model_summary[model]["type"]
        header_cells.append(
            f'<th class="model-header">{_escape_html(model)}<br>'
            f'<span class="model-type">({model_type})</span></th>'
        )
    header_row = "<tr>" + "".join(header_cells) + "</tr>"

    # Data rows
    data_rows = []
    for dim in dimensions:
        cells = [f"<td>{dim}</td>"]
        for model in models:
            scores = model_summary[model]["scores"]
            value = getattr(scores, dim.lower())
            cells.append(f"<td>{value:.2f}</td>")
        data_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Overall row
    overall_cells = ["<td><strong>Overall</strong></td>"]
    for model in models:
        scores = model_summary[model]["scores"]
        color = _score_color(scores.average)
        overall_cells.append(
            f'<td><span class="score" style="background-color: {color}20; color: {color};">'
            f'{scores.average:.2f}</span></td>'
        )
    overall_row = '<tr class="overall-row">' + "".join(overall_cells) + "</tr>"

    return f"""
    <table class="summary-table">
        <thead>{header_row}</thead>
        <tbody>
            {"".join(data_rows)}
            {overall_row}
        </tbody>
    </table>
    """


def _build_detail_sections(
    process_data: PipelineProcessData,
    model_mapping: dict[str, str] | None = None,
) -> str:
    """Build detailed paragraph sections with anonymized model names."""
    sections = []

    for i, para in enumerate(process_data.paragraphs):
        sections.append(_build_segment_detail(i + 1, para, model_mapping))

    return "".join(sections)


def _build_segment_detail(
    segment_num: int,
    para,
    model_mapping: dict[str, str] | None = None,
) -> str:
    """Build detail view for one segment/paragraph with anonymized model names."""
    # Source text (already extracted in translator.py) - no truncation
    source_text = para.source_text

    # Model results table
    model_results = _collect_model_results(para, model_mapping)
    results_table = _build_model_results_table(model_results)

    # Evaluation details
    eval_details = ""
    if para.evaluations:
        eval_items = []
        for ev in para.evaluations:
            if ev.reasoning:
                # Anonymize evaluator and target model names
                evaluator_name = ev.evaluator_model
                target_name = ev.target_model
                if model_mapping:
                    evaluator_name = _anonymize_model_name(evaluator_name, model_mapping)
                    # target_model might have prefix like "translator:" or "polisher:"
                    if target_name.startswith("translator:"):
                        target_display = target_name[len("translator:"):]
                        target_name = _anonymize_model_name(target_display, model_mapping)
                    elif target_name.startswith("polisher:"):
                        target_display = target_name[len("polisher:"):].replace("+", "→")
                        target_name = _anonymize_model_name(target_display, model_mapping)
                    else:
                        target_name = _anonymize_model_name(target_name, model_mapping)
                # Build evaluation item with reasoning and suggestions
                suggestion_html = ""
                if ev.suggestions:
                    suggestion_html = (
                        f'<div class="suggestion" style="margin-top: 4px; padding-left: 12px; '
                        f'color: #0066cc; font-style: italic;">'
                        f'💡 优化建议: {_escape_html(ev.suggestions)}</div>'
                    )
                eval_items.append(
                    f'<div class="evaluation-item" style="margin-bottom: 8px;">'
                    f'<strong>{_escape_html(evaluator_name)}</strong> → '
                    f'<strong>{_escape_html(target_name)}</strong>: '
                    f'<span class="score" style="background-color: {_score_color(ev.scores.average)}20; '
                    f'color: {_score_color(ev.scores.average)};">{ev.scores.average:.1f}/10</span> '
                    f'- {_escape_html(ev.reasoning)}'
                    f'{suggestion_html}</div>'
                )
        if eval_items:
            eval_details = f"""
            <div class="evaluation-details">
                <strong>Evaluation Details:</strong>
                {"".join(eval_items)}
            </div>
            """

    return f"""
    <div class="segment">
        <div class="segment-header">Segment {segment_num}</div>
        <div class="source-label">Source:</div>
        <div class="source-text">{_escape_html(source_text)}</div>
        {results_table}
        {eval_details}
    </div>
    """


def _build_model_results_table(model_results: list[dict[str, Any]]) -> str:
    """Build the model results table for a segment."""
    if not model_results:
        return ""

    rows = []
    for result in model_results:
        role_class = "role-translator" if result["role"] == "Translator" else "role-polisher"
        score = result["score"]
        score_class = "score-good" if score >= 8 else ("score-medium" if score >= 6 else "score-bad")

        # Translation text (already extracted) - no truncation
        trans_text = result["translation"]

        rows.append(f"""
        <tr>
            <td>{_escape_html(result["model"])}</td>
            <td><span class="role-tag {role_class}">{result["role"]}</span></td>
            <td class="translation-cell">{_escape_html(trans_text)}</td>
            <td><span class="score {score_class}">{score:.2f}</span></td>
        </tr>
        """)

    return f"""
    <table>
        <thead>
            <tr>
                <th style="width: 120px;">Model</th>
                <th style="width: 100px;">Role</th>
                <th>Translation</th>
                <th style="width: 80px;">Score</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """
