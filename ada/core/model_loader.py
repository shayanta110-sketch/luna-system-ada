# core/model_loader.py
"""
مدل لودر برای بارگذاری و مدیریت مدل‌های زبان بزرگ (LLM) در Ada.
این ماژول می‌تواند با nexus-router (اختیاری) یکپارچه شود یا به صورت مستقل کار کند.
"""

import logging
from typing import Optional, Dict, Any

# تلاش برای ایمپورت nexus-router (اختیاری)
try:
    from nexus_router import NexusRouter
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    NexusRouter = None

logger = logging.getLogger(__name__)

# لیست مدل‌های پیش‌فرض (مناسب برای سخت‌افزار محدود)
DEFAULT_MODELS = {
    "cpu": "llama2:7b",      # مدل سبک برای CPU
    "gpu": "llama2:7b",      # در GPU 2GB ممکن است جا نشود، اما به عنوان placeholder
    "tiny": "tinyllama:1.1b" # خیلی سبک
}

class ModelLoader:
    """
    لودر مدل با قابلیت انتخاب خودکار بر اساس منابع موجود و یکپارچگی اختیاری با nexus-router.
    """

    @staticmethod
    def load_model(
        model_name: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ) -> str:
        """
        یک مدل را انتخاب کرده و نام آن را برمی‌گرداند.
        
        Args:
            model_name: نام مدل مورد نظر (اختیاری)
            device: "cpu", "gpu" یا "auto"
            **kwargs: پارامترهای اضافی (ممکن است به nexus-router ارسال شود)
        
        Returns:
            نام مدل انتخاب شده به صورت رشته
        """
        # اگر nexus-router در دسترس باشد و مدل خاصی مشخص نشده، از آن استفاده کن
        if NEXUS_AVAILABLE and model_name is None:
            try:
                # در nexus-router واقعی، متدی برای انتخاب خودکار مدل وجود ندارد.
                # در عوض ما از یک روش ساده استفاده می‌کنیم.
                # اگر nexus-router قابلیت推薦 مدل دارد، اینجا می‌توانید آن را صدا بزنید.
                # اما فعلاً از منطق ساده زیر استفاده می‌کنیم:
                router = NexusRouter()
                # فرض می‌کنیم router یک attribute به نام `default_model` دارد
                if hasattr(router, 'default_model'):
                    selected = router.default_model
                    logger.info(f"nexus-router recommended model: {selected}")
                    return selected
                else:
                    logger.warning("nexus-router available but no default_model attribute")
            except Exception as e:
                logger.warning(f"nexus-router error: {e}")
        
        # اگر مدل به طور مستقیم مشخص شده، از آن استفاده کن
        if model_name:
            logger.info(f"Using explicitly specified model: {model_name}")
            return model_name
        
        # Fallback: انتخاب بر اساس device
        if device == "gpu":
            selected = DEFAULT_MODELS["gpu"]
        elif device == "cpu":
            selected = DEFAULT_MODELS["cpu"]
        else:  # auto
            # سعی می‌کنیم GPU را تشخیص دهیم (ساده)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus and gpus[0].memoryFree > 1024:  # حداقل 1GB VRAM آزاد
                    selected = DEFAULT_MODELS["gpu"]
                else:
                    selected = DEFAULT_MODELS["cpu"]
            except ImportError:
                selected = DEFAULT_MODELS["cpu"]
        
        logger.info(f"Fallback model selected: {selected}")
        return selected

    @staticmethod
    def get_capabilities(model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        دریافت قابلیت‌های یک مدل (طول context, توابع پشتیبانی شده و غیره).
        
        Args:
            model_id: شناسه مدل (اگر None باشد، مدل پیش‌فرض را بار می‌کند)
        
        Returns:
            دیکشنری شامل اطلاعاتی مثل max_context_length, supports_functions و غیره
        """
        if model_id is None:
            model_id = ModelLoader.load_model()
        
        # اگر nexus-router متد get_model_capabilities داشته باشد، از آن استفاده کن
        if NEXUS_AVAILABLE:
            try:
                from nexus_router import get_model_capabilities
                return get_model_capabilities(model_id)
            except (ImportError, AttributeError):
                logger.debug("get_model_capabilities not available in nexus-router")
        
        # اطلاعات پیش‌فرض برای مدل‌های شناخته شده
        capabilities = {
            "max_context_length": 2048,
            "supports_functions": False,
            "supports_streaming": True,
            "model_type": "llama",
        }
        
        # تنظیمات خاص برای مدل‌های معروف
        if "llama2" in model_id:
            capabilities["max_context_length"] = 4096
        elif "tinyllama" in model_id:
            capabilities["max_context_length"] = 2048
        
        logger.info(f"Capabilities for {model_id}: {capabilities}")
        return capabilities

    @staticmethod
    def unload_model(model_id: str) -> bool:
        """
        تلاش برای تخلیه مدل از حافظه (در صورت پشتیبانی nexus-router یا Ollama)
        
        Returns:
            True اگر موفقیت‌آمیز بود، در غیر این صورت False
        """
        if NEXUS_AVAILABLE:
            try:
                # فرض می‌کنیم nexus-router متد unload دارد
                router = NexusRouter()
                if hasattr(router, 'unload_model'):
                    router.unload_model(model_id)
                    logger.info(f"Model {model_id} unloaded via nexus-router")
                    return True
            except Exception as e:
                logger.warning(f"Failed to unload via nexus-router: {e}")
        
        # راهکار ساده: فراخوانی API Ollama برای تخلیه (اختیاری)
        try:
            import requests
            response = requests.post("http://localhost:11434/api/generate", json={"model": model_id, "keep_alive": 0})
            if response.status_code == 200:
                logger.info(f"Model {model_id} unloaded via Ollama keep_alive=0")
                return True
        except Exception:
            pass
        
        logger.warning(f"Could not unload model {model_id} – manual cleanup may be needed")
        return False
