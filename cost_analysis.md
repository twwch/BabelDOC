# Pipeline Cost Analysis

Based on the data from `output/pipeline_test/pipeline_process_data.json` and current model pricing.

## Summary

| Stage | Model | Cost (USD) | Cost per 1k Chars (USD) |
| :--- | :--- | :--- | :--- |
| **Translation** | DeepSeek-V3.2-Exp | $0.0485 | **$0.00022** |
| **Polishing** | Gemini-2.5-Flash | $1.1623 | **$0.00528** |
| **Evaluation** | GPT-5.1 | $1.3223 | **$0.00600** |
| **Total** | | **$2.5331** | **$0.01150** |

**Total Source Characters:** 220,227

---

## Detailed Breakdown

### 1. Translation (DeepSeek-V3.2-Exp)
*   **Input Tokens:** 115,691
*   **Output Tokens:** 38,243
*   **Pricing:**
    *   Input: $0.28 / 1M tokens
    *   Output: $0.42 / 1M tokens
*   **Cost Calculation:**
    *   Input: $0.0324
    *   Output: $0.0161
    *   **Total:** $0.0485

### 2. Polishing (Gemini-2.5-Flash)
*   **Input Tokens:** 146,998
*   **Output Tokens:** 447,292
*   **Pricing:**
    *   Input: $0.30 / 1M tokens
    *   Output: $2.50 / 1M tokens
*   **Cost Calculation:**
    *   Input: $0.0441
    *   Output: $1.1182
    *   **Total:** $1.1623

### 3. Evaluation (GPT-5.1)
*   **Input Tokens:** 183,218
*   **Output Tokens:** 109,325
*   **Pricing:**
    *   Input: $1.25 / 1M tokens
    *   Output: $10.00 / 1M tokens
*   **Cost Calculation:**
    *   Input: $0.2290
    *   Output: $1.0933
    *   **Total:** $1.3223
