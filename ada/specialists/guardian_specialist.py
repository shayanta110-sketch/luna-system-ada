# ada/specialists/guardian_specialist.py
"""
متخصص Guardian برای Ada.
این متخصص قبل از اجرای مدل اصلی، وضعیت منابع سیستم (رم، حافظه ویدئویی و ...)
را از طریق سرویس ada-guardian بررسی کرده و در صورت ناکافی بودن، از اجرا جلوگیری می‌کند.
"""

import logging
import requests
from typing import Dict, Any, Optional

from ada.specialists.base_specialist import BaseSpecialist, SpecialistContext

logger = logging.getLogger(__name__)


class GuardianSpecialist(BaseSpecialist):
    """
    متخصص Guardian – بررسی منابع سیستم قبل از اجرای مدل.
    
    این متخصص با ada-guardian (سرویس پایش منابع) ارتباط برقرار کرده و
    در صورت کمبود منابع، اجرای مدل اصلی را متوقف می‌کند.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه متخصص Guardian.
        
        Args:
            config: تنظیمات شامل guardian_url، timeout و required_resources
        """
        super().__init__(config)
        # آدرس سرویس ada-guardian – پورت پیش‌فرض 8001 مطابق با docker-compose شما
        self.guardian_url = self.config.get("guardian_url", "http://localhost:8001")
        # تایم‌اوت درخواست به guardian (ثانیه)
        self.timeout = self.config.get("timeout", 5)
        # منابع مورد نیاز (مقدار پیش‌فرض برای سخت‌افزار شما)
        self.required_resources = self.config.get("required_resources", {
            "ram_available_mb": 1024,   # حداقل 1 گیگابایت رم آزاد
            "vram_available_mb": 256,   # حداقل 256 مگابایت حافظه ویدئویی آزاد
        })

    def get_specialty_name(self) -> str:
        """نام متخصص – 'guardian'"""
        return "guardian"

    def check_resources(self) -> Dict[str, Any]:
        """
        دریافت وضعیت منابع از سرویس ada-guardian.
        
        Returns:
            دیکشنری وضعیت شامل کلیدهای ram_available_mb، vram_available_mb و ...
            در صورت خطا، دیکشنری شامل {"error": ..., "available": False}
        """
        try:
            response = requests.get(
                f"{self.guardian_url}/api/v1/resources",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to communicate with ada-guardian at {self.guardian_url}: {e}")
            return {"error": str(e), "available": False}

    def is_resources_sufficient(self, resources: Dict[str, Any]) -> bool:
        """
        بررسی کافی بودن منابع بر اساس required_resources.
        
        Args:
            resources: دیکشنری منابع دریافتی از ada-guardian
            
        Returns:
            True اگر تمام منابع مورد نیاز به اندازه کافی موجود باشند.
        """
        if "error" in resources:
            logger.warning("Guardian reported error, assuming insufficient resources")
            return False

        # اگر required_resources خالی باشد، همیشه کافی در نظر گرفته می‌شود
        if not self.required_resources:
            return True

        for resource, required in self.required_resources.items():
            available = resources.get(resource)
            if available is None:
                logger.warning(f"Resource {resource} not reported by guardian – assuming insufficient")
                return False
            if available < required:
                logger.warning(
                    f"Insufficient {resource}: available={available} MB, required={required} MB"
                )
                return False
        return True

    def before_model_invoke(self, context: SpecialistContext) -> SpecialistContext:
        """
        هوک قبل از اجرای مدل – بررسی منابع و در صورت لزوم مسدود کردن اجرا.
        
        اطلاعات وضعیت در context.custom_data ذخیره می‌شوند.
        
        Args:
            context: زمینه درخواست
            
        Returns:
            context به‌روز شده با اطلاعات guardian
        """
        logger.info("Checking system resources via ada-guardian...")
        resources = self.check_resources()
        
        # ذخیره نتیجه در custom_data برای استفاده در سایر بخش‌ها
        context.set_custom("guardian_resources", resources)
        
        if not self.is_resources_sufficient(resources):
            logger.error("Insufficient resources for model invocation – blocking execution")
            context.set_custom("guardian_blocked", True)
            context.set_custom(
                "guardian_message",
                f"Resource constraints prevent model invocation. Required: {self.required_resources}"
            )
        else:
            logger.info("Resources sufficient, proceeding with model invocation")
            context.set_custom("guardian_blocked", False)
            context.set_custom("guardian_message", "Resources OK")
        
        return context

    def can_proceed(self, context: SpecialistContext) -> bool:
        """
        تصمیم‌گیری برای ادامه یا توقف اجرای مدل اصلی.
        
        Returns:
            False اگر guardian_blocked=True باشد، در غیر این صورت True
        """
        blocked = context.get_custom("guardian_blocked", False)
        if blocked:
            logger.debug("Guardian specialist blocks execution")
        return not blocked

    def after_model_invoke(self, context: SpecialistContext) -> SpecialistContext:
        """
        هوک بعد از اجرای مدل (اختیاری) – می‌توان برای آزادسازی منابع استفاده کرد.
        فعلاً عملیات خاصی انجام نمی‌دهد.
        """
        # در صورت نیاز می‌توان پس از اجرای موفق، guardian را مطلع کرد
        logger.debug("Guardian after_model_invoke – no action needed")
        return context

    def on_error(self, context: SpecialistContext, error: Exception) -> SpecialistContext:
        """
        مدیریت خطاهای رخ داده در حین اجرای متخصص Guardian.
        """
        logger.error(f"Guardian specialist encountered error: {error}")
        context.set_custom("guardian_error", str(error))
        context.set_custom("guardian_blocked", True)  # در صورت خطا، مسدود کن
        return context

    def get_required_resources(self) -> Dict[str, Any]:
        """
        منابع مورد نیاز خود متخصص Guardian.
        (برای هماهنگی با زنجیره متخصصان)
        """
        return {
            "ram_mb": 50,      # خود guardian حافظه کمی مصرف می‌کند
            "network": True    # نیاز به اتصال شبکه
        }

    def validate_config(self) -> bool:
        """
        اعتبارسنجی تنظیمات متخصص Guardian.
        """
        # بررسی وجود guardian_url و timeout معتبر
        if not isinstance(self.guardian_url, str) or not self.guardian_url.startswith(("http://", "https://")):
            logger.error("guardian_url must be a valid HTTP/HTTPS URL")
            return False
        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            logger.error("timeout must be a positive number")
            return False
        return True
