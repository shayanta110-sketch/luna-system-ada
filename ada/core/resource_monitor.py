# core/resource_monitor.py
"""
پایشگر منابع سیستم برای Ada.
این ماژول به صورت دوره‌ای مصرف حافظه رم را بررسی کرده و در صورت تجاوز از آستانه تعیین شده،
یک عملیات پاکسازی (مثل تخلیه مدل‌های Ollama) را فراخوانی می‌کند.
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Callable, Union, Awaitable

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    پایشگر منابع سیستم (رم) با قابلیت اجرا هم در محیط sync و هم async.
    """

    def __init__(
        self,
        threshold_percent: float = 80.0,
        check_interval_seconds: float = 5.0,
        cleanup_callback: Optional[Union[Callable[[], None], Callable[[], Awaitable[None]]]] = None,
        auto_start: bool = False
    ):
        """
        Args:
            threshold_percent: درصد مصرف رم که در صورت تجاوز، پاکسازی انجام شود (مثال: 80.0)
            check_interval_seconds: فاصله زمانی بین هر بررسی (ثانیه)
            cleanup_callback: تابعی که در زمان فشار حافظه فراخوانی می‌شود (می‌تواند sync یا async باشد)
            auto_start: اگر True باشد، پایشگر بلافاصله شروع به کار می‌کند
        """
        if not PSUTIL_AVAILABLE:
            raise RuntimeError("psutil is required for ResourceMonitor. Please install: pip install psutil")

        self.threshold = threshold_percent
        self.interval = check_interval_seconds
        self.callback = cleanup_callback

        # کنترل برای نسخه threading
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # کنترل برای نسخه asyncio
        self._async_task: Optional[asyncio.Task] = None
        self._running = False

        if auto_start:
            self.start()

    # ----------------------------------------------------------------------
    # بخش threading (برای برنامه‌های sync)
    # ----------------------------------------------------------------------
    def start(self) -> None:
        """شروع پایشگر در یک نخ جداگانه."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("ResourceMonitor already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=False, name="ResourceMonitor")
        self._thread.start()
        logger.info(f"ResourceMonitor started (threshold={self.threshold}%, interval={self.interval}s)")

    def stop(self, timeout: float = 3.0) -> None:
        """
        توقف graceful پایشگر و انتظار برای پایان نخ.
        
        Args:
            timeout: حداکثر زمان انتظار برای پایان نخ (ثانیه)
        """
        if self._thread is None or not self._thread.is_alive():
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("ResourceMonitor thread did not terminate within timeout")
        else:
            logger.info("ResourceMonitor stopped")
        self._thread = None

    def _monitor_loop(self) -> None:
        """حلقه اصلی در نخ (sync)"""
        while not self._stop_event.is_set():
            try:
                mem_percent = psutil.virtual_memory().percent
                if mem_percent >= self.threshold:
                    logger.warning(f"High memory pressure: {mem_percent:.1f}% ≥ {self.threshold}%")
                    self._trigger_cleanup()
                else:
                    logger.debug(f"Memory OK: {mem_percent:.1f}%")
            except Exception as e:
                logger.exception(f"Error in resource monitor loop: {e}")
            
            # انتظار برای بازه زمانی بعدی یا رویداد توقف
            self._stop_event.wait(timeout=self.interval)

    def _trigger_cleanup(self) -> None:
        """فراخوانی تابع پاکسازی در حالت sync"""
        if not self.callback:
            logger.warning("No cleanup callback provided – memory pressure unhandled")
            return
        
        try:
            # اگر callback یک coroutine باشد، نمی‌توانیم در اینجا await کنیم (چون در نخ معمولی هستیم)
            # پس در این حالت فقط اجرا نمی‌شود. کاربر باید از نسخه async استفاده کند.
            if asyncio.iscoroutinefunction(self.callback):
                logger.error("Cleanup callback is async but called from sync monitor – skipping. Use AsyncResourceMonitor instead.")
            else:
                self.callback()
        except Exception as e:
            logger.exception(f"Cleanup callback failed: {e}")

    # ----------------------------------------------------------------------
    # بخش asyncio (برای برنامه‌های async مانند Ada)
    # ----------------------------------------------------------------------
    async def start_async(self) -> None:
        """شروع پایشگر به صورت async task"""
        if self._running:
            logger.warning("AsyncResourceMonitor already running")
            return
        self._running = True
        self._async_task = asyncio.create_task(self._async_monitor_loop())
        logger.info(f"AsyncResourceMonitor started (threshold={self.threshold}%, interval={self.interval}s)")

    async def stop_async(self) -> None:
        """توقف پایشگر async"""
        if not self._running:
            return
        self._running = False
        if self._async_task:
            self._async_task.cancel()
            try:
                await self._async_task
            except asyncio.CancelledError:
                pass
            self._async_task = None
        logger.info("AsyncResourceMonitor stopped")

    async def _async_monitor_loop(self) -> None:
        """حلقه اصلی async"""
        while self._running:
            try:
                # اجرای psutil.virtual_memory در thread جدا تا مسدود نشود
                mem = await asyncio.to_thread(psutil.virtual_memory)
                mem_percent = mem.percent
                if mem_percent >= self.threshold:
                    logger.warning(f"High memory pressure: {mem_percent:.1f}% ≥ {self.threshold}%")
                    await self._trigger_cleanup_async()
                else:
                    logger.debug(f"Memory OK: {mem_percent:.1f}%")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in async resource monitor: {e}")
            
            await asyncio.sleep(self.interval)

    async def _trigger_cleanup_async(self) -> None:
        """فراخوانی تابع پاکسازی در حالت async"""
        if not self.callback:
            logger.warning("No cleanup callback provided – memory pressure unhandled")
            return
        
        try:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback()
            else:
                # اگر callback sync است، در thread جدا اجرا کن تا event loop مسدود نشود
                await asyncio.to_thread(self.callback)
        except Exception as e:
            logger.exception(f"Cleanup callback failed: {e}")

    # ----------------------------------------------------------------------
    # متدهای کمکی ایستا
    # ----------------------------------------------------------------------
    @staticmethod
    def get_memory_usage_percent() -> float:
        """درصد مصرف رم فعلی"""
        return psutil.virtual_memory().percent

    @staticmethod
    def get_memory_available_mb() -> float:
        """مقدار رم آزاد به مگابایت"""
        return psutil.virtual_memory().available / (1024 * 1024)

    @staticmethod
    def get_cpu_usage_percent(interval: float = 0.5) -> float:
        """درصد مصرف CPU در بازه زمانی کوتاه"""
        return psutil.cpu_percent(interval=interval)

    @staticmethod
    def get_gpu_memory_info() -> Optional[dict]:
        """
        دریافت اطلاعات حافظه کارت گرافیک (در صورت وجود NVIDIA و نصب GPUtil)
        بازگشت: دیکشنری شامل total, used, free (به مگابایت) یا None
        """
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    "total_mb": gpu.memoryTotal,
                    "used_mb": gpu.memoryUsed,
                    "free_mb": gpu.memoryFree,
                    "utilization_percent": gpu.memoryUtil * 100
                }
        except ImportError:
            logger.debug("GPUtil not installed for GPU monitoring")
        except Exception as e:
            logger.debug(f"GPU info error: {e}")
        return None


# ______________________________________________________________________
# مثال استفاده (در صورت اجرای مستقیم)
# ______________________________________________________________________
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    # مثال: تابع پاکسازی ساده
    def clean():
        print("Cleaning up models...")

    # استفاده همزمان (threading)
    print("=== Sync mode ===")
    monitor = ResourceMonitor(threshold_percent=50.0, cleanup_callback=clean, auto_start=True)
    try:
        time.sleep(10)
    finally:
        monitor.stop()

    # استفاده async
    async def main():
        print("\n=== Async mode ===")
        async_monitor = ResourceMonitor(threshold_percent=50.0, cleanup_callback=clean)
        await async_monitor.start_async()
        try:
            await asyncio.sleep(10)
        finally:
            await async_monitor.stop_async()

    asyncio.run(main())
