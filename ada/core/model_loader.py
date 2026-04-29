# ada/core/model_loader.py
"""
مدیر بارگذاری مدل با استفاده از Nexus Router.
بهترین مدل Ollama را با توجه به وظیفه، پیچیدگی و سلامت سیستم انتخاب می‌کند.
"""

import logging
from typing import Optional

from nexus_router import ModelRouter, ResourceGuard

logger = logging.getLogger(__name__)

# نمونه‌های سراسری (lazy-loaded برای جلوگیری از مصرف RAM در صورت عدم نیاز)
_router: Optional[ModelRouter] = None
_guard: Optional[ResourceGuard] = None

def _get_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router

def _get_guard() -> ResourceGuard:
    global _guard
    if _guard is None:
        _guard = ResourceGuard()
    return _guard

def select_model_for_task(task: str, complexity: str = "medium") -> str:
    """
    بهترین مدل Ollama را برای وظیفۀ داده‌شده برمی‌گرداند.
    اگر سیستم در وضعیت بحرانی باشد، کوچک‌ترین مدل ممکن را انتخاب می‌کند.
    در صورت یافت نشدن هیچ مدل، یک خطای واضح صادر می‌شود.
    """
    router = _get_router()
    guard = _get_guard()
    health = guard.check_health()

    # اگر سیستم بحرانی است، مدل سبک انتخاب کن
    if not health["safe_to_proceed"]:
        logger.warning(
            "System resources critical: %s. Falling back to smallest compatible model.",
            health["reasons"]
        )
        # سعی می‌کنیم کوچک‌ترین مدل متناسب با وظیفه را پیدا کنیم
        model_name = router.select_model(task=task, complexity="simple", prefer_small=True)
        if model_name:
            logger.info("Fallback model selected: %s", model_name)
            return model_name
        # اگر پیدا نشد، هر مدل LLM عمومی را برگردان
        model_name = router.select_model(task="general", complexity="simple", prefer_small=True)
        if model_name:
            logger.info("Fallback general model selected: %s", model_name)
            return model_name
        raise RuntimeError("No model available (even fallback) – check Ollama installation.")

    # حالت عادی: بهترین مدل متناسب با وظیفه را انتخاب کن
    model_name = router.select_model(task=task, complexity=complexity)
    if model_name:
        logger.info("Model selected for task '%s' [%s]: %s", task, complexity, model_name)
        return model_name

    # اگر هیچ مدل تخصصی پیدا نشد، به دنبال مدل عمومی بگرد
    logger.warning("No specialized model found for task '%s', trying general LLM.", task)
    model_name = router.select_model(task="general", complexity=complexity)
    if model_name:
        logger.info("Falling back to general model: %s", model_name)
        return model_name

    # اگر حتی مدل عمومی هم نبود، خطای صریح بده
    raise RuntimeError(
        f"No Ollama model could be selected for task='{task}', complexity='{complexity}'. "
        "Please pull at least one model."
    )
