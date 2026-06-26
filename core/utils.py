"""
MedBuddy — Utility Functions
Robust JSON extraction from LLM responses (handles think tags, markdown code blocks, etc.)
"""
import re
import json


def extract_json_from_response(content: str) -> str:
    """
    Extract clean JSON from an LLM response that may contain:
    - <think>...</think> reasoning blocks (Qwen models)
    - ```json ... ``` markdown code blocks
    - Preamble text before the JSON
    - Trailing text after the JSON

    Args:
        content: Raw LLM response string

    Returns:
        Cleaned string containing only the JSON content
    """
    if not content:
        return ""

    text = content.strip()

    # Step 1: Remove <think>...</think> blocks (Qwen models)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Step 2: Remove markdown code block wrappers
    # Handle ```json ... ``` and ``` ... ```
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Step 3: Try to find JSON object { ... }
    obj_match = re.search(r'\{.*\}', text, re.DOTALL)
    arr_match = re.search(r'\[.*\]', text, re.DOTALL)

    # Return the first valid JSON structure found
    if obj_match:
        candidate = obj_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    if arr_match:
        candidate = arr_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Step 4: If no valid JSON found, return the cleaned text as-is
    return text


def parse_json_response(content: str) -> dict | list | None:
    """
    Parse JSON from an LLM response, handling think tags and markdown.

    Args:
        content: Raw LLM response string

    Returns:
        Parsed JSON object (dict or list), or None if parsing fails
    """
    cleaned = extract_json_from_response(content)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
