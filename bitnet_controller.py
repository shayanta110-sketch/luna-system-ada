class BitNetController:
    def __init__(self):
        self.ram_threshold = 14 * 1024  # 14GB in MB (85% of 16GB)
        self.vram_threshold = 1800  # 1800MB
        self.vram_critical = 500  # 500MB

    def ensure_vram_available(self, free_vram_mb):
        if free_vram_mb < self.vram_critical:
            self.unload_all_gpu_models()
            return True
        return False

    def unload_all_gpu_models(self):
        # Implementation to unload all GPU models
        pass