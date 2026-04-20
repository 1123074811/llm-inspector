"""
runner/token_counter.py — Token counting with tiktoken when available.

Uses tiktoken for accurate OpenAI-compatible token counting.
Falls back to character/4 approximation when tiktoken is not installed.

Reference for fallback heuristic:
    OpenAI Cookbook: "How to count tokens with tiktoken"
    URL: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken

Reference for per-message overhead:
    OpenAI API documentation, "Counting tokens for chat completions API calls"
    URL: https://platform.openai.com/docs/guides/text-generation
    Note: ~3 tokens per message for role/separator overhead (empirical).
"""
from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)

# Try to import tiktoken at module load time.
# Optional dependency — graceful fallback if absent.
try:
    import tiktoken as _tiktoken  # type: ignore
    HAS_TIKTOKEN: bool = True
except ImportError:
    _tiktoken = None  # type: ignore
    HAS_TIKTOKEN: bool = False

# Per-message overhead (role + separator tokens in chat format).
# Source: OpenAI API docs — approximately 3 tokens per message.
_PER_MESSAGE_OVERHEAD: int = 3


def count_tokens(text: str, model: str = "gpt-4o") -> tuple[int, str]:
    """
    Count tokens in a text string.

    Tries tiktoken (accurate) first, falls back to len(text)//4 heuristic.

    Args:
        text:  input string
        model: model name used to select the encoding (default: gpt-4o)

    Returns:
        (count, method) where method is one of:
          - "tiktoken-o200k"     — counted via tiktoken o200k_base encoding
          - "tiktoken-cl100k"    — counted via tiktoken cl100k_base encoding
          - "fallback-estimate"  — character/4 approximation

    Reference:
        OpenAI Cookbook: How to count tokens with tiktoken
        URL: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    """
    if not text:
        return 0, "fallback-estimate"

    if HAS_TIKTOKEN and _tiktoken is not None:
        # Try model-specific encoding first
        try:
            enc = _tiktoken.encoding_for_model(model)
            enc_name = enc.name
            # Normalise encoding name → method label
            if "o200k" in enc_name:
                method = "tiktoken-o200k"
            else:
                method = "tiktoken-cl100k"
            count = len(enc.encode(text))
            return count, method
        except KeyError:
            pass  # Model not recognised → try fallback encoding
        except Exception as exc:
            logger.debug("tiktoken encoding_for_model failed", model=model, error=str(exc))

        # Fallback: use o200k_base (GPT-4o series)
        try:
            enc = _tiktoken.get_encoding("o200k_base")
            count = len(enc.encode(text))
            return count, "tiktoken-o200k"
        except Exception as exc:
            logger.debug("tiktoken get_encoding o200k_base failed", error=str(exc))

        # Last resort tiktoken fallback: cl100k_base
        try:
            enc = _tiktoken.get_encoding("cl100k_base")
            count = len(enc.encode(text))
            return count, "tiktoken-cl100k"
        except Exception as exc:
            logger.debug("tiktoken get_encoding cl100k_base failed", error=str(exc))

    # Character/4 heuristic (no tiktoken)
    count = max(1, len(text) // 4)
    return count, "fallback-estimate"


def count_messages_tokens(
    messages: list[dict],
    model: str = "gpt-4o",
) -> tuple[int, str]:
    """
    Count total tokens across a list of chat messages.

    Adds per-message overhead (~3 tokens each) matching OpenAI's chat
    format encoding (role token + separator tokens).

    Args:
        messages: list of dicts with at least a "content" key
        model:    model name for encoding selection

    Returns:
        (total_count, method) same method labels as count_tokens()

    Reference:
        OpenAI API: Counting tokens for chat completions API calls
        URL: https://platform.openai.com/docs/guides/text-generation
    """
    if not messages:
        return 0, "fallback-estimate"

    total_tokens = 0
    method_used = "fallback-estimate"

    for msg in messages:
        content = msg.get("content") or ""
        if not isinstance(content, str):
            # Handle structured content blocks (list of dicts)
            try:
                content = " ".join(
                    str(part.get("text", "")) if isinstance(part, dict) else str(part)
                    for part in content
                )
            except Exception:
                content = str(content)

        n, method = count_tokens(content, model=model)
        total_tokens += n + _PER_MESSAGE_OVERHEAD
        # Use the most precise method seen across messages
        if method != "fallback-estimate":
            method_used = method

    return total_tokens, method_used
