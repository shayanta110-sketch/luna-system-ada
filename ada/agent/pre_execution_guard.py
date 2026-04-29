# pre_execution_guard.py – نسخه نهایی یکپارچه با nexus_router
"""
 این ماژول یک سیستم مانیتورینگ و مدیریت پیشگیرانه منابع سخت‌افزاری برای Ada فراهم می‌کنه.
 کارش اینه که قبل از اجرای هر عملیات سنگین، وضعیت حافظه رم (RAM) و حافظه ویدئویی (VRAM)
 رو چک می‌کنه و اگه منابع کافی نباشه، از اجرا جلوگیری می‌کنه.
 """

import asyncio
import logging
from typing import Callable, Optional, Tuple, Dict, Any

# وابستگی‌های اختیاری: برای قدرتمندتر شدن ماژول، از nexus_router و کتابخونه‌های سیستمی استفاده می‌کنیم
try:
    # nexus_router رو که خودت ساختی، اینجا ایمپورت می‌کنیم
    from nexus_router import NexusRouter  
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False

try:
    import psutil  # برای چک کردن حافظه رم سیستم
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil  # برای چک کردن حافظه کارت گرافیک (NVIDIA)
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# راه‌اندازی logger برای ثبت رویدادهای مهم
logger = logging.getLogger(__name__)

class ResourceUnavailableError(RuntimeError):
    """Exception سفارشی برای وقتی که منابع کافی برای اجرا وجود نداره."""
    pass

class PreExecutionGuard:
    """
    کلاس اصلی برای مدیریت و بررسی منابع قبل از اجرا.
    این کلاس می‌تونه با nexus_router یکپارچه بشه و از قابلیت‌های پیشرفته‌اش مثل timeout و retry استفاده کنه.
    """

    def __init__(self, router: Optional[NexusRouter] = None):
        """
        مقداردهی اولیه.
        اگر router اوکی باشه، از قابلیت‌هاش برای چک کردن سلامت سیستم هم استفاده می‌کنیم.
        """
        self.router = router
        self._enabled = NEXUS_AVAILABLE
        # آستانه‌های پیش‌فرض برای سخت‌افزار مقرون‌به‌صرفه (مثل سیستم خودت)
        self._ram_threshold_mb: int = 512   # 512 مگابایت رم آزاد باید باشه
        self._vram_threshold_mb: int = 256  # 256 مگابایت حافظه کارت گرافیک آزاد

    def set_thresholds(self, ram_mb: Optional[int] = None, vram_mb: Optional[int] = None) -> None:
        """برای تنظیم دستی آستانه‌های حافظه."""
        if ram_mb is not None:
            self._ram_threshold_mb = ram_mb
        if vram_mb is not None:
            self._vram_threshold_mb = vram_mb
        logger.debug(f"Thresholds updated: RAM={self._ram_threshold_mb}MB, VRAM={self._vram_threshold_mb}MB")

    async def check_system_health(self) -> Tuple[bool, str]:
        """
        بررسی می‌کنه که آیا منابع سیستم (رم و حافظه گرافیک) کافی هستند یا نه.
        خروجی: (آیا سلامت هست؟, پیام متناسب)
        """
        if not self._enabled and self.router is None:
            return True, "nexus-router not configured, skipping system checks"
        if not NEXUS_AVAILABLE:
            return True, "nexus-router not available, skipping system checks"

        # پرچم‌های اولیه، فرض می‌کنیم همه چی اوکی هست
        ram_ok = True
        vram_ok = True
        messages = []

        # --- بررسی رم سیستم با psutil ---
        if PSUTIL_AVAILABLE:
            try:
                # از اجرای حلقه‌ای که ممکنه مسدود بشه جلوگیری می‌کنیم
                mem = await asyncio.to_thread(psutil.virtual_memory)
                available_ram_mb = mem.available / (1024 * 1024)
                if available_ram_mb < self._ram_threshold_mb:
                    ram_ok = False
                    messages.append(f"RAM low: {available_ram_mb:.0f}MB < {self._ram_threshold_mb}MB threshold")
                else:
                    messages.append(f"RAM ok: {available_ram_mb:.0f}MB found")
            except Exception as e:
                logger.warning(f"Failed to get RAM stats: {e}")
                messages.append(f"RAM check failed: {e}")
        else:
            logger.debug("psutil not available, skipping RAM health check")
            messages.append("psutil not installed, skipping RAM check")

        # --- بررسی حافظه‌ی گرافیک NVIDIA با GPUtil ---
        if GPUTIL_AVAILABLE:
            try:
                gpus = await asyncio.to_thread(GPUtil.getGPUs)
                if gpus:
                    # از اولین کارت گرافیک (معمولاً اصلی) استفاده می‌کنیم
                    gpu = gpus[0]
                    free_vram_mb = gpu.memoryFree
                    if free_vram_mb < self._vram_threshold_mb:
                        vram_ok = False
                        messages.append(f"GPU memory low: {free_vram_mb}MB < {self._vram_threshold_mb}MB threshold")
                    else:
                        messages.append(f"GPU memory ok: {free_vram_mb}MB found")
                else:
                    messages.append("No NVIDIA GPU detected, skipping VRAM check")
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")
                messages.append(f"GPU check failed: {e}")
        else:
            logger.debug("GPUtil not available, skipping VRAM health check")
            messages.append("GPUtil not installed, skipping VRAM check")

        # همه چی اوکی بود پس True برمی‌گردونیم
        return (ram_ok and vram_ok), "; ".join(messages)

    async def execute_with_guard(self, node_func: Callable, *args, **kwargs):
        """
        تابع اصلی که قبل از اجرا سلامت سیستم رو چک می‌کنه و اگر منابع کافی نبود، خطا میده.
        این تابع می‌تونه هم تابع sync و هم async رو هندل کنه.
        """
        is_healthy, message = await self.check_system_health()
        if not is_healthy:
            error_msg = f"Pre-execution guard blocked node: {message}"
            logger.error(error_msg)
            raise ResourceUnavailableError(error_msg)

        logger.info(f"Pre-execution check passed: {message}")

        # انتخاب درست نوع تابع (sync یا async) برای جلوگیری از بلاک شدن حلقه رویداد
        if asyncio.iscoroutinefunction(node_func):
            return await node_func(*args, **kwargs)
        else:
            # اگر تابع sync هست، توی یک thread دیگه اجراش کن تا حلقه asyncio مسدود نشه
            return await asyncio.to_thread(node_func, *args, **kwargs)

# ------------------------------
# (اختیاری) نمونه Singleton و Hook برای یکپارچگی با NexusRouter
# ------------------------------
_global_guard: Optional[PreExecutionGuard] = None

def get_guard() -> PreExecutionGuard:
    """Singleton برای دسترسی آسان به PreExecutionGuard."""
    global _global_guard
    if _global_guard is None:
        if NEXUS_AVAILABLE:
            router = NexusRouter()
            _global_guard = PreExecutionGuard(router)
        else:
            _global_guard = PreExecutionGuard()
    return _global_guard

def global_pre_execution_hook(*args, **kwargs):
    """
    تابع Hook نهایی برای استفاده در nexus_router به عنوان pre-execution middleware.
    این تابع مستقیماً execute_with_guard رو صدا می‌زنه تا قبل از پردازش درخواست،
    وضعیت سیستم رو بررسی کنه.
    """
    guard = get_guard()
    # برای راحتی کار، execute_with_guard رو با دریافت تابع بعدی (مثل next middleware) صدا می‌زنیم
    # اما اینجا فرض می‌کنیم که هدف، اجرای مستقیم یه تابع هست.
    return guard.execute_with_guard(*args, **kwargs)

# ------------------------------
# برای تست کردن ماژول به صورت جداگانه
# ------------------------------
async def dummy_compute():
    print("Doing some work...")
    return "Work done"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    guard = get_guard()
    
    # تست 1: بررسی سلامت سیستم
    healthy, msg = asyncio.run(guard.check_system_health())
    print(f"Health check: {healthy}, message: {msg}")
    
    # تست 2: اجرای یه تابع با ضمانت
    try:
        result = asyncio.run(guard.execute_with_guard(dummy_compute))
        print(f"Result: {result}")
    except ResourceUnavailableError as e:
        print(f"Failed: {e}")
