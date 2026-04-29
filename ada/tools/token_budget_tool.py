"""Token budget allocation tool for Ada."""

from typing import Dict, Any, Optional
from ada.core.token_budget_manager import TokenBudgetManager


def allocate_token_budget(
    total_budget: int,
    reserved_for_output: int = 1000,
    model_max_tokens: Optional[int] = None,
    initial_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Allocate token budget for a prompt and return allocation summary and final prompt.

    Args:
        total_budget: Total token budget available.
        reserved_for_output: Tokens to reserve for model output.
        model_max_tokens: Model's maximum context length (if None, no hard cap).
        initial_prompt: Initial user/assistant prompt to process.

    Returns:
        Dictionary containing:
            - allocation_summary: Budget breakdown (input budget, output budget, etc.)
            - final_prompt: Processed prompt respecting token limits
    """
    manager = TokenBudgetManager(
        total_budget=total_budget,
        reserved_for_output=reserved_for_output,
        model_max_tokens=model_max_tokens,
    )

    allocation_summary = manager.get_budget_allocation()

    final_prompt = ""
    if initial_prompt:
        final_prompt = manager.truncate_prompt_to_budget(initial_prompt)
    else:
        final_prompt = manager.build_default_prompt()

    return {
        "allocation_summary": allocation_summary,
        "final_prompt": final_prompt,
    }
