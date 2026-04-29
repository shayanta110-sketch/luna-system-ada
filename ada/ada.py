"""ADA - AI Development Assistant with Nexus Router integration."""

import asyncio
import json
from typing import Any, Callable, Dict, Optional


class NexusRouterHook:
    """Pre-execution hook for Nexus Router resource checking."""
    
    def __init__(self, router_endpoint: str = "http://localhost:8080/check"):
        self.router_endpoint = router_endpoint
    
    async def check_resources(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check resource availability via Nexus Router before LLM request."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.router_endpoint,
                    json=request_context,
                    timeout=5.0
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"allowed": result.get("allowed", True), "data": result}
                    else:
                        return {"allowed": False, "error": f"Router check failed with status {response.status}"}
            except Exception as e:
                return {"allowed": False, "error": str(e)}
    
    def create_pre_execution_wrapper(self, llm_request_func: Callable) -> Callable:
        """Wrap LLM request function with pre-execution resource check."""
        async def wrapper(*args, **kwargs):
            # Extract request context from args/kwargs
            context = {
                "args": str(args),
                "kwargs_keys": list(kwargs.keys()),
                "model": kwargs.get("model", "unknown"),
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Check resources via Nexus Router
            check_result = await self.check_resources(context)
            
            if not check_result.get("allowed", False):
                error_msg = check_result.get("error", "Resource check denied by Nexus Router")
                raise RuntimeError(f"Nexus Router blocked request: {error_msg}")
            
            # Proceed with actual LLM request
            return await llm_request_func(*args, **kwargs)
        
        return wrapper


# Global hook instance
_nexus_hook = None


def initialize_nexus_hook(router_endpoint: str = "http://localhost:8080/check"):
    """Initialize the Nexus Router pre-execution hook."""
    global _nexus_hook
    _nexus_hook = NexusRouterHook(router_endpoint)
    return _nexus_hook


def get_nexus_hook() -> Optional[NexusRouterHook]:
    """Get the initialized Nexus Router hook."""
    return _nexus_hook


# Resource monitor for periodic checks
_resource_monitor_task = None


async def resource_monitor_loop(router_endpoint: str = "http://localhost:8080/check", interval: float = 30.0):
    """Periodically check resource health via Nexus Router."""
    import aiohttp
    
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    router_endpoint,
                    json={"type": "health_check"},
                    timeout=5.0
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if not result.get("healthy", True):
                            print(f"[ADA] Resource monitor warning: unhealthy state - {result.get('message', '')}")
                    else:
                        print(f"[ADA] Resource monitor error: HTTP {response.status}")
        except Exception as e:
            print(f"[ADA] Resource monitor exception: {e}")
        
        await asyncio.sleep(interval)


def start_resource_monitor(router_endpoint: str = "http://localhost:8080/check", interval: float = 30.0):
    """Start the background resource monitor task."""
    global _resource_monitor_task
    if _resource_monitor_task is None or _resource_monitor_task.done():
        loop = asyncio.get_event_loop()
        _resource_monitor_task = loop.create_task(resource_monitor_loop(router_endpoint, interval))
        return _resource_monitor_task
    return _resource_monitor_task


def stop_resource_monitor():
    """Stop the background resource monitor task."""
    global _resource_monitor_task
    if _resource_monitor_task and not _resource_monitor_task.done():
        _resource_monitor_task.cancel()
        _resource_monitor_task = None


def setup_shutdown_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print("\n[ADA] Shutting down...")
        stop_resource_monitor()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main execution entry point."""
    setup_shutdown_handlers()
    initialize_nexus_hook()
    start_resource_monitor()
    
    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        stop_resource_monitor()


# Example integration function
def integrate_with_llm_client(llm_client: Any, method_name: str = "generate"):
    """Integrate Nexus Router hook with an LLM client instance."""
    if _nexus_hook is None:
        raise RuntimeError("Nexus hook not initialized. Call initialize_nexus_hook() first.")
    
    original_method = getattr(llm_client, method_name)
    wrapped_method = _nexus_hook.create_pre_execution_wrapper(original_method)
    setattr(llm_client, method_name, wrapped_method)
    return llm_client


if __name__ == "__main__":
    asyncio.run(main())
