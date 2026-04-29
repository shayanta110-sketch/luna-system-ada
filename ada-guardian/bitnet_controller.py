"""
BitNet Controller for ADA Guardian
Manages real-time RAM/VRAM monitoring, temperature checks, and throttling logic.
"""

import psutil
import GPUtil
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    vram_percent: Optional[float] = None
    gpu_temp_c: Optional[float] = None
    cpu_temp_c: Optional[float] = None

class BitNetController:
    """
    Controller for monitoring and throttling BitNet operations based on system resources.
    """
    
    def __init__(
        self,
        ram_threshold: float = 85.0,
        vram_threshold: float = 90.0,
        temp_threshold: float = 85.0,
        check_interval: float = 2.0,
        throttle_callback: Optional[Callable[[bool], None]] = None
    ):
        """
        Initialize the BitNet Controller.
        
        Args:
            ram_threshold: RAM usage percentage to trigger throttling
            vram_threshold: VRAM usage percentage to trigger throttling
            temp_threshold: GPU temperature (Celsius) to trigger throttling
            check_interval: Seconds between monitoring checks
            throttle_callback: Callback function when throttling state changes
        """
        self.ram_threshold = ram_threshold
        self.vram_threshold = vram_threshold
        self.temp_threshold = temp_threshold
        self.check_interval = check_interval
        self.throttle_callback = throttle_callback
        self.is_throttled = False
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self._monitoring:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("BitNet Controller monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        logger.info("BitNet Controller monitoring stopped")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            metrics = self.get_system_metrics()
            should_throttle = self._check_throttle_conditions(metrics)
            
            if should_throttle != self.is_throttled:
                self.is_throttled = should_throttle
                if self.throttle_callback:
                    self.throttle_callback(self.is_throttled)
                status = "throttled" if self.is_throttled else "resumed"
                logger.info(f"BitNet operation {status}")
                
            time.sleep(self.check_interval)
            
    def _check_throttle_conditions(self, metrics: SystemMetrics) -> bool:
        """
        Evaluate if throttling should be active based on system metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            True if throttling should be active, False otherwise
        """
        if metrics.ram_percent >= self.ram_threshold:
            logger.warning(f"RAM usage high: {metrics.ram_percent:.1f}%")
            return True
            
        if metrics.vram_percent is not None and metrics.vram_percent >= self.vram_threshold:
            logger.warning(f"VRAM usage high: {metrics.vram_percent:.1f}%")
            return True
            
        if metrics.gpu_temp_c is not None and metrics.gpu_temp_c >= self.temp_threshold:
            logger.warning(f"GPU temperature high: {metrics.gpu_temp_c:.1f}°C")
            return True
            
        return False
        
    @staticmethod
    def get_system_metrics() -> SystemMetrics:
        """
        Collect current system metrics including RAM, VRAM, and temperatures.
        
        Returns:
            SystemMetrics object with current resource usage
        """
        # RAM metrics
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        ram_percent = ram.percent
        
        # GPU metrics (if available)
        vram_used_gb = None
        vram_total_gb = None
        vram_percent = None
        gpu_temp_c = None
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                vram_used_gb = gpu.memoryUsed / 1024.0
                vram_total_gb = gpu.memoryTotal / 1024.0
                vram_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else None
                gpu_temp_c = gpu.temperature
        except Exception as e:
            logger.debug(f"Could not retrieve GPU metrics: {e}")
            
        # CPU temperature (optional, not always available)
        cpu_temp_c = None
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp_c = temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                cpu_temp_c = temps['cpu_thermal'][0].current
        except Exception:
            pass
            
        return SystemMetrics(
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            ram_percent=ram_percent,
            vram_used_gb=vram_used_gb,
            vram_total_gb=vram_total_gb,
            vram_percent=vram_percent,
            gpu_temp_c=gpu_temp_c,
            cpu_temp_c=cpu_temp_c
        )
        
    def get_current_status(self) -> dict:
        """
        Get current monitoring status and metrics.
        
        Returns:
            Dictionary with throttling state and current metrics
        """
        metrics = self.get_system_metrics()
        return {
            'throttled': self.is_throttled,
            'metrics': {
                'ram_percent': metrics.ram_percent,
                'vram_percent': metrics.vram_percent,
                'gpu_temp_c': metrics.gpu_temp_c,
                'cpu_temp_c': metrics.cpu_temp_c
            }
        }
        
    def set_thresholds(self, ram: Optional[float] = None, vram: Optional[float] = None, temp: Optional[float] = None) -> None:
        """
        Dynamically update throttling thresholds.
        
        Args:
            ram: New RAM threshold percentage
            vram: New VRAM threshold percentage
            temp: New GPU temperature threshold in Celsius
        """
        if ram is not None:
            self.ram_threshold = ram
            logger.info(f"RAM threshold updated to {ram}%")
        if vram is not None:
            self.vram_threshold = vram
            logger.info(f"VRAM threshold updated to {vram}%")
        if temp is not None:
            self.temp_threshold = temp
            logger.info(f"Temperature threshold updated to {temp}°C")
