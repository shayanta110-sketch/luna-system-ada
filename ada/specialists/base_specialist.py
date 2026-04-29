# ada/specialists/base_specialist.py
"""
ماژول کلاس پایه برای تمام متخصصان (Specialists) سیستم Ada.
این کلاس یک قرارداد استاندارد برای پیاده‌سازی پلاگین‌های تخصصی
مانند Guardian، Optimizer، Validator و غیره تعریف می‌کند.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpecialistContext:
    """
    زمینه اجرای متخصص - حاوی اطلاعات مربوط به درخواست جاری.
    
    Attributes:
        request_id: شناسه یکتای درخواست
        model_name: نام مدل در حال استفاده
        prompt: متن ورودی کاربر
        response: پاسخ تولید شده (در صورت وجود)
        metadata: اطلاعات اضافی و دلخواه
        custom_data: دیکشنری برای ذخیره داده‌های سفارشی توسط متخصصان
    """
    request_id: str
    model_name: str
    prompt: str
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """دسترسی آسان به metadata"""
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """تنظیم مقدار در metadata"""
        self.metadata[key] = value
    
    def set_custom(self, key: str, value: Any) -> None:
        """تنظیم مقدار در داده‌های سفارشی متخصص"""
        self.custom_data[key] = value
    
    def get_custom(self, key: str, default: Any = None) -> Any:
        """دسترسی به داده‌های سفارشی متخصص"""
        return self.custom_data.get(key, default)


class BaseSpecialist(ABC):
    """
    کلاس پایه انتزاعی برای تمام متخصصان Ada.
    
    هر متخصص می‌تواند قبل از اجرای مدل (pre-process)، بعد از اجرای مدل
    (post-process) و یا برای تصمیم‌گیری در مورد ادامه روند اجرا، هوک‌های
    خود را پیاده‌سازی کند.
    
    Example:
        class MySpecialist(BaseSpecialist):
            def get_specialty_name(self) -> str:
                return "my_specialist"
            
            def before_model_invoke(self, context: SpecialistContext) -> SpecialistContext:
                # منطق قبل از اجرای مدل
                context.set("my_flag", True)
                return context
            
            def after_model_invoke(self, context: SpecialistContext) -> SpecialistContext:
                # منطق بعد از اجرای مدل
                result = context.response
                # پردازش نتیجه
                return context
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه متخصص با تنظیمات اختیاری.
        
        Args:
            config: دیکشنری حاوی تنظیمات پیکربندی مخصوص متخصص
        """
        self.config = config or {}
        self.name = self.get_specialty_name()
        self.enabled = self.config.get("enabled", True)
        self.priority = self.config.get("priority", 100)  # اولویت کمتر = اجرای زودتر
        logger.info(f"Specialist '{self.name}' initialized with priority {self.priority}")
    
    @abstractmethod
    def get_specialty_name(self) -> str:
        """
        برگرداندن نام یکتای متخصص.
        
        Returns:
            نام متخصص به صورت رشته (مثال: "guardian", "optimizer", "validator")
        """
        pass
    
    def before_model_invoke(self, context: SpecialistContext) -> SpecialistContext:
        """
        هوکی که قبل از فراخوانی مدل اصلی اجرا می‌شود.
        
        این متد می‌تواند:
        - context را برای افزودن اطلاعات تغییر دهد
        - بررسی‌های پیش از اجرا انجام دهد
        - منابع را آماده کند
        
        Args:
            context: زمینه جاری درخواست
            
        Returns:
            context اصلاح شده (یا همان context اصلی)
        """
        # پیاده‌سازی پیش‌فرض: بدون تغییر
        return context
    
    def after_model_invoke(self, context: SpecialistContext) -> SpecialistContext:
        """
        هوکی که بعد از فراخوانی مدل اصلی اجرا می‌شود.
        
        این متد می‌تواند:
        - پاسخ مدل را پردازش کند
        - فیلتر یا اصلاح روی خروجی اعمال کند
        - منابع را آزاد کند
        
        Args:
            context: زمینه جاری درخواست (شامل response تولید شده)
            
        Returns:
            context اصلاح شده (یا همان context اصلی)
        """
        # پیاده‌سازی پیش‌فرض: بدون تغییر
        return context
    
    def on_error(self, context: SpecialistContext, error: Exception) -> SpecialistContext:
        """
        هوکی که در صورت بروز خطا در حین اجرای مدل فراخوانی می‌شود.
        
        Args:
            context: زمینه جاری درخواست
            error: خطای رخ داده
            
        Returns:
            context اصلاح شده (می‌تواند شامل پیام خطای سفارشی باشد)
        """
        logger.error(f"Specialist '{self.name}' caught error: {error}")
        # پیاده‌سازی پیش‌فرض: خطا را در context ذخیره می‌کند
        context.set("error", str(error))
        context.set("error_specialist", self.name)
        return context
    
    def can_proceed(self, context: SpecialistContext) -> bool:
        """
        تصمیم‌گیری در مورد ادامه روند اجرا یا توقف آن.
        
        اگر این متد False برگرداند، اجرای مدل اصلی انجام نخواهد شد.
        
        Args:
            context: زمینه جاری درخواست
            
        Returns:
            True اگر اجرا ادامه یابد، False در غیر این صورت
        """
        # پیاده‌سازی پیش‌فرض: همیشه True برگردان
        return True
    
    def cleanup(self, context: Optional[SpecialistContext] = None) -> None:
        """
        پاکسازی منابع پس از پایان اجرا.
        
        این متد در پایان چرخه حیات درخواست فراخوانی می‌شود.
        
        Args:
            context: زمینه درخواست (در صورت وجود)
        """
        logger.debug(f"Specialist '{self.name}' cleanup completed")
    
    def get_required_resources(self) -> Dict[str, Any]:
        """
        برگرداندن منابع مورد نیاز این متخصص.
        
        Returns:
            دیکشنری شامل منابع مورد نیاز (مثال: {"ram_mb": 512, "vram_mb": 128})
        """
        # پیاده‌سازی پیش‌فرض: بدون نیاز خاص
        return {}
    
    def validate_config(self) -> bool:
        """
        اعتبارسنجی تنظیمات متخصص.
        
        Returns:
            True اگر تنظیمات معتبر باشند، False در غیر این صورت
        """
        # پیاده‌سازی پیش‌فرض: تنظیمات خالی معتبر است
        return True
    
    def __repr__(self) -> str:
        """نمایش رشته‌ای از متخصص برای دیباگ"""
        return f"<{self.__class__.__name__} name='{self.name}' enabled={self.enabled} priority={self.priority}>"


class SpecialistChain:
    """
    زنجیره‌ای از متخصصان که به ترتیب اولویت اجرا می‌شوند.
    
    این کلاس مسئول مدیریت و هماهنگی چندین متخصص به صورت زنجیره‌ای است.
    """
    
    def __init__(self, specialists: Optional[list[BaseSpecialist]] = None):
        """
        مقداردهی اولیه زنجیره متخصصان.
        
        Args:
            specialists: لیست اولیه متخصصان (اختیاری)
        """
        self._specialists: Dict[str, BaseSpecialist] = {}
        if specialists:
            for specialist in specialists:
                self.register(specialist)
    
    def register(self, specialist: BaseSpecialist) -> None:
        """
        ثبت یک متخصص جدید در زنجیره.
        
        Args:
            specialist: نمونه‌ای از یک کلاس مشتق شده از BaseSpecialist
        """
        if not isinstance(specialist, BaseSpecialist):
            raise TypeError(f"Expected BaseSpecialist, got {type(specialist)}")
        
        if not specialist.validate_config():
            logger.warning(f"Specialist '{specialist.name}' has invalid config, skipping registration")
            return
        
        self._specialists[specialist.name] = specialist
        logger.info(f"Registered specialist: {specialist}")
    
    def unregister(self, name: str) -> bool:
        """
        حذف یک متخصص از زنجیره.
        
        Args:
            name: نام متخصص مورد نظر
            
        Returns:
            True اگر متخصص وجود داشت و حذف شد، False در غیر این صورت
        """
        if name in self._specialists:
            del self._specialists[name]
            logger.info(f"Unregistered specialist: {name}")
            return True
        return False
    
    def get_specialist(self, name: str) -> Optional[BaseSpecialist]:
        """دریافت متخصص بر اساس نام"""
        return self._specialists.get(name)
    
    def get_all_specialists(self) -> list[BaseSpecialist]:
        """دریافت لیست تمام متخصصان (مرتب شده بر اساس اولویت)"""
        return sorted(self._specialists.values(), key=lambda s: s.priority)
    
    def run_before_hooks(self, context: SpecialistContext) -> SpecialistContext:
        """
        اجرای هوک before_model_invoke برای تمام متخصصان فعال.
        
        متخصصان بر اساس اولویت (کمترین عدد = اولویت بالاتر) اجرا می‌شوند.
        
        Args:
            context: زمینه جاری درخواست
            
        Returns:
            context اصلاح شده پس از عبور از تمام هوک‌ها
        """
        for specialist in self.get_all_specialists():
            if not specialist.enabled:
                continue
            
            try:
                logger.debug(f"Running before hook for {specialist.name}")
                context = specialist.before_model_invoke(context)
                
                # اگر متخصص اجازه ادامه نداد، زنجیره را قطع کن
                if not specialist.can_proceed(context):
                    logger.warning(f"Specialist {specialist.name} blocked execution")
                    context.set("blocked_by", specialist.name)
                    break
                    
            except Exception as e:
                logger.exception(f"Error in before hook of {specialist.name}: {e}")
                context = specialist.on_error(context, e)
        
        return context
    
    def run_after_hooks(self, context: SpecialistContext) -> SpecialistContext:
        """
        اجرای هوک after_model_invoke برای تمام متخصصان فعال.
        
        متخصصان به ترتیب معکوس اولویت اجرا می‌شوند (آخرین متخصص اول).
        
        Args:
            context: زمینه جاری درخواست
            
        Returns:
            context اصلاح شده پس از عبور از تمام هوک‌ها
        """
        for specialist in reversed(self.get_all_specialists()):
            if not specialist.enabled:
                continue
            
            try:
                logger.debug(f"Running after hook for {specialist.name}")
                context = specialist.after_model_invoke(context)
            except Exception as e:
                logger.exception(f"Error in after hook of {specialist.name}: {e}")
                context = specialist.on_error(context, e)
        
        return context
    
    def cleanup_all(self, context: Optional[SpecialistContext] = None) -> None:
        """
        فراخوانی cleanup برای تمام متخصصان.
        
        Args:
            context: زمینه درخواست (اختیاری)
        """
        for specialist in self.get_all_specialists():
            try:
                specialist.cleanup(context)
            except Exception as e:
                logger.exception(f"Error in cleanup of {specialist.name}: {e}")
    
    def __len__(self) -> int:
        """تعداد متخصصان ثبت شده"""
        return len(self._specialists)
    
    def __repr__(self) -> str:
        """نمایش رشته‌ای از زنجیره"""
        return f"<SpecialistChain specialists={list(self._specialists.keys())}>"


# ______________________________________________________________________
# مثال استفاده (در صورت اجرای مستقیم فایل)
# ______________________________________________________________________
if __name__ == "__main__":
    # تنظیم لاگینگ برای مشاهده خروجی
    logging.basicConfig(level=logging.INFO)
    
    # مثال: پیاده‌سازی یک متخصص ساده برای تست
    class ExampleSpecialist(BaseSpecialist):
        def get_specialty_name(self) -> str:
            return "example"
        
        def before_model_invoke(self, context: SpecialistContext) -> SpecialistContext:
            print(f"[Example] Before model: {context.prompt[:50]}...")
            context.set("example_timestamp", "2024-01-01")
            return context
        
        def after_model_invoke(self, context: SpecialistContext) -> SpecialistContext:
            print(f"[Example] After model, response length: {len(context.response or '')}")
            return context
    
    # ایجاد زنجیره و ثبت متخصص
    chain = SpecialistChain()
    specialist = ExampleSpecialist({"enabled": True, "priority": 10})
    chain.register(specialist)
    
    # شبیه‌سازی یک درخواست
    ctx = SpecialistContext(
        request_id="test-123",
        model_name="llama2",
        prompt="Hello, how are you?"
    )
    
    # اجرای هوک‌ها
    ctx = chain.run_before_hooks(ctx)
    print(f"Context after before hooks: {ctx.metadata}")
    
    # شبیه‌سازی اجرای مدل
    ctx.response = "I'm fine, thank you!"
    
    ctx = chain.run_after_hooks(ctx)
    chain.cleanup_all(ctx)
    
    print("Test completed successfully!")
