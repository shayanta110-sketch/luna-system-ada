# nexus_router/core.py
"""
Intelligent model router and hardware guardian for local LLMs (Ollama).
Combines ResourceGuard (RAM/VRAM monitoring) and ModelRouter (task-based model selection).
"""

import logging
import time
import psutil
from typing import Dict, Any, Optional, Tuple, List

try:
    import GPUtil
except ImportError:
    GPUtil = None

from .ollama_adapter import OllamaAdapter

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# آستانه‌های سلامت سیستم
# ------------------------------------------------------------------
RAM_THRESHOLD_PERCENT = 90        # درصد مصرف RAM که سیستم ناامن می‌شود
VRAM_THRESHOLD_PERCENT = 95       # درصد مصرف VRAM ناامن
MIN_FREE_RAM_GB = 2.0             # حداقل RAM آزاد مجاز به گیگابایت
MIN_FREE_VRAM_MB = 500            # حداقل VRAM آزاد مجاز به مگابایت
HEALTH_CACHE_SECONDS = 2.0        # مدت زمان کش کردن وضعیت سلامت (ثانیه)


class ResourceGuard:
    """
    نگهبان سخت‌افزار – بررسی RAM/VRAM قبل از اجرای مدل.
    """

    def __init__(self):
        self._last_check = 0.0
        self._cached_health: Optional[Dict[str, Any]] = None

    def check_health(self, force: bool = False) -> Dict[str, Any]:
        """
        وضعیت فعلی سیستم را برمی‌گرداند. نتیجه تا HEALTH_CACHE_SECONDS ثانیه کش می‌شود
        مگر آنکه force=True باشد.
        """
        now = time.time()
        if not force and (now - self._last_check) < HEALTH_CACHE_SECONDS:
            return self._cached_health or {}

        ram = psutil.virtual_memory()
        health: Dict[str, Any] = {
            "timestamp": now,
            "ram": {
                "total_gb": ram.total / (1024 ** 3),
                "available_gb": ram.available / (1024 ** 3),
                "percent": ram.percent,
            },
            "gpu": {
                "available": False,
                "vram_free_mb": 0,
                "vram_total_mb": 0,
                "vram_percent": 0,
            },
            "safe_to_proceed": True,
            "reasons": [],
        }

        # بررسی RAM
        if ram.percent > RAM_THRESHOLD_PERCENT:
            health["safe_to_proceed"] = False
            health["reasons"].append(
                f"RAM usage {ram.percent}% > {RAM_THRESHOLD_PERCENT}%"
            )
        if ram.available < MIN_FREE_RAM_GB * (1024 ** 3):
            health["safe_to_proceed"] = False
            health["reasons"].append(
                f"Free RAM {ram.available/(1024**3):.1f} GB < {MIN_FREE_RAM_GB} GB"
            )

        # بررسی GPU (در صورت وجود GPUtil)
        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    health["gpu"] = {
                        "available": True,
                        "vram_free_mb": gpu.memoryFree,
                        "vram_total_mb": gpu.memoryTotal,
                        "vram_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100
                        if gpu.memoryTotal > 0
                        else 0,
                    }
                    if gpu.memoryTotal > 0:
                        vram_pct = health["gpu"]["vram_percent"]
                        if vram_pct > VRAM_THRESHOLD_PERCENT:
                            health["safe_to_proceed"] = False
                            health["reasons"].append(
                                f"VRAM usage {vram_pct:.1f}% > {VRAM_THRESHOLD_PERCENT}%"
                            )
                        if gpu.memoryFree < MIN_FREE_VRAM_MB:
                            health["safe_to_proceed"] = False
                            health["reasons"].append(
                                f"Free VRAM {gpu.memoryFree} MB < {MIN_FREE_VRAM_MB} MB"
                            )
            except Exception as e:
                logger.debug(f"GPU health check failed: {e}")

        self._last_check = now
        self._cached_health = health
        return health


class ModelRouter:
    """
    مسیریاب هوشمند مدل: بهترین مدل موجود در Ollama را برای یک وظیفۀ خاص انتخاب می‌کند.
    """

    def __init__(self, adapter: Optional[OllamaAdapter] = None):
        self.adapter = adapter or OllamaAdapter()
        self.guard = ResourceGuard()

    def select_model(
        self,
        task: str,
        complexity: str = "medium",
        required_capabilities: Optional[List[str]] = None,
        prefer_small: bool = True,
    ) -> Optional[str]:
        """
        انتخاب بهترین مدل با توجه به نوع وظیفه، پیچیدگی و منابع موجود.

        Args:
            task: نوع وظیفه (code, reasoning, vision, general, ...)
            complexity: simple, medium, complex
            required_capabilities: لیست قابلیت‌های ضروری (مثلاً ["vision"])
            prefer_small: اولویت‌دهی به مدل‌های کوچک

        Returns:
            نام مدل Ollama (مثلاً "phi3:mini") یا None
        """
        models = self.adapter.get_models()
        if not models:
            logger.error("No Ollama models found.")
            return None

        health = self.guard.check_health()
        free_ram_gb = health["ram"]["available_gb"]

        candidates: List[Tuple[str, float, float, float]] = []  # (name, score, params, est_ram_gb)

        for name, info in models.items():
            # ۱. فیلتر نوع مدل بر اساس وظیفه
            m_type = info.get("type", "llm")
            if task == "code" and m_type != "code":
                continue
            if task == "vision" and m_type != "vision":
                continue
            if task == "reasoning" and m_type not in ("llm", "code"):
                continue
            if task == "general" and m_type not in ("llm", "code"):
                continue

            params = info.get("params_b", 1.0)

            # ۲. فیلتر پیچیدگی
            if complexity == "simple" and params > 3:
                continue
            if complexity == "medium" and (params < 1.5 or params > 7):
                continue
            if complexity == "complex" and params < 4:
                continue

            # ۳. فیلتر قابلیت‌های خاص
            caps = info.get("capabilities", {})
            if required_capabilities:
                if not all(caps.get(cap, False) for cap in required_capabilities):
                    continue

            # ۴. بررسی RAM
            est_ram_gb = info.get("est_ram_mb", 0) / 1024
            if est_ram_gb > free_ram_gb - 1.0:      # حاشیه امن ۱ گیگابایت
                continue

            # ۵. محاسبه امتیاز اولیه
            score = info.get("quality_score", 50)

            # ۶. اولویت مدل‌های کوچک
            if prefer_small and params <= 3.0:
                score += 30
                if params <= 1.5:
                    score += 20

            # ۷. جریمۀ مدل‌های خیلی بزرگ
            if params >= 7.0:
                score -= 40

            candidates.append((name, score, params, est_ram_gb))

        if not candidates:
            logger.warning("No suitable model found. Falling back to smallest LLM.")
            llms = [(n, i) for n, i in models.items() if i.get("type") == "llm"]
            if llms:
                llms.sort(key=lambda x: x[1].get("params_b", 999))
                return llms[0][0]
            return None

        # مرتب‌سازی بر اساس امتیاز (نزولی)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        logger.info(
            "ModelRouter selected: %s (score=%.1f, params=%.1f B, RAM≈%.1f GB)",
            best[0], best[1], best[2], best[3]
        )
        return best[0]

    def can_load_model(self, model_name: str) -> Tuple[bool, str]:
        """
        بررسی می‌کند که آیا RAM کافی برای بارگذاری مدل داده‌شده وجود دارد یا خیر.
        Returns: (می‌توان بارگذاری کرد؟, پیام توضیحی)
        """
        model_info = self.adapter.get_model(model_name)
        if not model_info:
            return False, f"Model '{model_name}' not found in Ollama."

        health = self.guard.check_health()
        required_ram_gb = model_info.get("est_ram_mb", 0) / 1024
        available_ram_gb = health["ram"]["available_gb"]

        if available_ram_gb < required_ram_gb + 1.0:
            return (
                False,
                f"Insufficient RAM: need ≈{required_ram_gb:.1f} GB, have {available_ram_gb:.1f} GB free",
            )
        return True, "OK"

    def get_resource_status(self) -> Dict[str, Any]:
        """
        گزارش کامل از وضعیت منابع و مدل‌های موجود.
        """
        health = self.guard.check_health()
        models = self.adapter.get_models()
        return {
            "system_health": health,
            "available_models": list(models.keys()),
            "model_count": len(models),
        }
