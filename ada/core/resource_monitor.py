import psutil
import threading
import time
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Periodically monitors RAM usage and triggers cleanup of Ollama models
    when memory pressure is detected during long-running executions.
    """

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        memory_threshold_percent: float = 85.0,
        ollama_cleanup_callback: Optional[Callable[[], None]] = None,
        auto_start: bool = False
    ):
        """
        Initialize the resource monitor.

        Args:
            check_interval_seconds: How often to check RAM usage (seconds)
            memory_threshold_percent: RAM usage percentage triggering cleanup
            ollama_cleanup_callback: Function to call for Ollama model cleanup
            auto_start: Whether to begin monitoring immediately
        """
        self.check_interval = check_interval_seconds
        self.threshold = memory_threshold_percent
        self.cleanup_callback = ollama_cleanup_callback
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

        if auto_start:
            self.start()

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._running:
            logger.warning("ResourceMonitor already running")
            return

        self._stop_event.clear()
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"ResourceMonitor started (interval={self.check_interval}s, threshold={self.threshold}%)")

    def stop(self) -> None:
        """Stop the monitoring thread."""
        if not self._running:
            return

        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        self._running = False
        logger.info("ResourceMonitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in separate thread."""
        while not self._stop_event.is_set():
            try:
                self._check_memory_and_cleanup()
            except Exception as e:
                logger.exception(f"Error in memory check: {e}")

            # Sleep with periodic stop checks
            for _ in range(int(self.check_interval / 0.5)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.5)

    def _check_memory_and_cleanup(self) -> None:
        """Check current RAM usage and trigger cleanup if threshold exceeded."""
        memory_percent = self.get_memory_usage_percent()

        if memory_percent >= self.threshold:
            logger.warning(
                f"High memory usage detected: {memory_percent:.1f}% (threshold={self.threshold}%)"
            )
            if self.cleanup_callback:
                logger.info("Triggering Ollama model cleanup...")
                try:
                    self.cleanup_callback()
                except Exception as e:
                    logger.exception(f"Ollama cleanup callback failed: {e}")
            else:
                logger.warning("No cleanup callback provided - memory pressure unhandled")
        else:
            logger.debug(f"Memory usage: {memory_percent:.1f}% (below threshold)")

    @staticmethod
    def get_memory_usage_percent() -> float:
        """
        Get current RAM usage percentage.

        Returns:
            Memory usage percentage (0-100)
        """
        return psutil.virtual_memory().percent

    @staticmethod
    def get_memory_usage_gb() -> tuple[float, float]:
        """
        Get current RAM usage in GB.

        Returns:
            Tuple of (used_gb, total_gb)
        """
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return used_gb, total_gb

    @property
    def is_running(self) -> bool:
        """Check if monitor is actively running."""
        return self._running
