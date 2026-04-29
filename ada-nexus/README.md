# Ada-Nexus

A combined system with hardware optimization features for high-performance computing.

## Overview

Ada-Nexus integrates advanced scheduling and resource management to maximize throughput on modern hardware. It automatically detects and tunes to CPU, GPU, and memory subsystems.

## Hardware Optimization Features

- **CPU Affinity & NUMA Awareness** – Bind processes to specific cores and optimize for non-uniform memory access.
- **GPU Acceleration** – Leverages CUDA, OpenCL, or Vulkan for parallel workloads.
- **Memory Pooling & Huge Pages** – Reduces latency and improves cache efficiency.
- **Adaptive Frequency Scaling** – Dynamically adjusts CPU/GPU clocks based on workload.
- **I/O Prefetch & Vectorized Reads** – Optimizes storage access patterns.

## System Requirements

- **OS:** Linux (recommended Ubuntu 22.04+ or RHEL 9+)
- **CPU:** x86_64 or ARM64 with at least 4 cores
- **RAM:** 8 GB minimum (16 GB recommended)
- **GPU:** NVIDIA (CUDA 11.7+), AMD (ROCm 5.0+), or Intel Arc
- **Storage:** 2 GB free space

## Installation

```bash
git clone https://github.com/your-repo/ada-nexus.git
cd ada-nexus
pip install -r requirements.txt  # if Python-based
make install                      # if build from source
```

## Running the Combined System

### Basic Run
```bash
./start_nexus --mode combined --config configs/default.yaml
```

### With Full Hardware Optimization
```bash
./start_nexus --mode combined \
  --enable-cpu-affinity \
  --gpu-id 0 \
  --huge-pages 2048 \
  --adaptive-scaling
```

### Docker Deployment (Recommended for Production)
```bash
docker build -t ada-nexus .
docker run --rm --privileged \
  --cpuset-cpus=0-7 \
  --gpus all \
  --memory=16g \
  --shm-size=8g \
  ada-nexus
```

## Configuration

Edit `configs/hardware_optimized.yaml`:
```yaml
hardware:
  cpu:
    affinity: [0,2,4,6]
    governor: performance
  gpu:
    backend: cuda
    device_id: 0
    memory_fraction: 0.8
  memory:
    huge_pages: true
    page_size_kb: 2048
```

## Performance Tuning

- **CPU:** Run `scripts/tune_cpu.sh` to set scaling governors and IRQ affinity.
- **GPU:** Use `nvidia-smi -pl [power_limit]` to cap power for stable frequency.
- **Network:** Enable jumbo frames if using InfiniBand or 10GbE.
- **Monitoring:** Launch `./monitor.py --metrics all` to view real-time usage.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Huge pages not available | `sudo sysctl -w vm.nr_hugepages=1024` |
| GPU not detected | Run `lspci \| grep -i vga`; update drivers |
| CPU affinity failed | Start process with `sudo` or set `CAP_SYS_NICE` |

## License

MIT / Apache 2.0 (see LICENSE file)

## References

- [Performance Optimization Guide](docs/optimization.md)
- [Hardware Tuning Scripts](scripts/)
- [Benchmarking Results](benchmarks/)
