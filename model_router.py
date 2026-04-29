class DynamicModelOrchestrator:
    def _should_use_gpu(self, free_vram_mb):
        if free_vram_mb < 2500:
            return 0
        return 1
