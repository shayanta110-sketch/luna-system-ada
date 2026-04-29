"""Pre-execution guard module for checking system health before node execution."""

import asyncio
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Attempt to import nexus-router for resource checks
try:
    from nexus_router import NexusRouter
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    logger.warning("nexus-router not available. System health checks will be disabled.")


class PreExecutionGuard:
    """Global hook to check system health before running any node."""

    def __init__(self, router: Optional[NexusRouter] = None):
        self.router = router
        self._enabled = NEXUS_AVAILABLE
        self._ram_threshold_mb: int = 512   # Minimum free RAM in MB
        self._vram_threshold_mb: int = 256  # Minimum free VRAM in MB

    def set_thresholds(self, ram_mb: Optional[int] = None, vram_mb: Optional[int] = None) -> None:
        """Set minimum free memory thresholds."""
        if ram_mb is not None:
            self._ram_threshold_mb = ram_mb
        if vram_mb is not None:
            self._vram_threshold_mb = vram_mb

    async def check_system_health(self) -> tuple[bool, str]:
        """
        Check RAM and VRAM availability using nexus-router.
        Returns (is_healthy, message).
        """
        if not self._enabled or not self.router:
            return True, "nexus-router unavailable, skipping system checks"

        try:
            # Get system stats from nexus-router
            stats = await self.router.get_system_stats()

            # Check RAM
            free_ram_mb = stats.get("free_ram_mb", 0)
            if free_ram_mb < self._ram_threshold_mb:
                return False, f"Low RAM: {free_ram_mb}MB free (threshold: {self._ram_threshold_mb}MB)"

            # Check VRAM (if available)
            free_vram_mb = stats.get("free_vram_mb", None)
            if free_vram_mb is not None and free_vram_mb < self._vram_threshold_mb:
                return False, f"Low VRAM: {free_vram_mb}MB free (threshold: {self._vram_threshold_mb}MB)"

            return True, f"System healthy: RAM={free_ram_mb}MB free, VRAM={free_vram_mb if free_vram_mb else 'N/A'}MB free"

        except Exception as e:
            logger.error(f"Failed to check system health: {e}")
            return False, f"Health check failed: {e}"

    async def execute_with_guard(self, node_func: Callable, *args, **kwargs):
        """
        Pre-execution hook: check health, then run node if healthy.
        Raises RuntimeError if health check fails.
        """
        is_healthy, message = await self.check_system_health()
        if not is_healthy:
            raise RuntimeError(f"Pre-execution guard blocked node: {message}")

        logger.info(f"Pre-execution check passed: {message}")
        return await node_func(*args, **kwargs) if asyncio.iscoroutinefunction(node_func) else node_func(*args, **kwargs)


# Global singleton instance
_global_guard: Optional[PreExecutionGuard] = None


def get_guard() -> PreExecutionGuard:
    """Get or initialize the global pre-execution guard."""
    global _global_guard
    if _global_guard is None:
        if NEXUS_AVAILABLE:
            router = NexusRouter()
            _global_guard = PreExecutionGuard(router)
        else:
            _global_guard = PreExecutionGuard()
    return _global_guard


async def global_pre_execution_hook(node_func: Callable, *args, **kwargs):
    """
    Global entry point for pre-execution checks.
    Usage: result = await global_pre_execution_hook(my_node, arg1, arg2)
    """
    guard = get_guard()
    return await guard.execute_with_guard(node_func, *args, **kwargs)
