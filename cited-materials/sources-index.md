# Sources Index

本目录保存 `agentic-ai-head-cpu-insight-2025h2plus.md` 使用到的本地引用材料。

## Index

| Local file | Original URL | Used for |
| --- | --- | --- |
| `nvidia-cpu-gpu-memory-sharing-2025-09-05.pdf` | https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/ | Chrome 打印版网页 PDF；支撑 CPU 内存成为 warm KV 层、统一内存地址空间、NVLink-C2C 对 KV offload 的意义 |
| `nvidia-kv-bottlenecks-dynamo-2025-09-18.pdf` | https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/ | Chrome 打印版网页 PDF；支撑 KV cache 可卸到 CPU RAM / SSD / network storage，说明 host CPU 已成为分层状态存储中枢 |
| `nvidia-wide-expert-parallelism-2025-12-18.pdf` | https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/ | Chrome 打印版网页 PDF；支撑 MoE expert routing、placement 和跨 GPU 通信对 host orchestration 的要求 |
| `nvidia-inference-transfer-library-2026-03-09.pdf` | https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/ | Chrome 打印版网页 PDF；支撑 NIXL、RDMA / NVLink / NVMe-oF 等数据搬运能力，以及 host 进入轻数据面 |
| `nvidia-disaggregated-llm-k8s-2026-03-23.pdf` | https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/ | Chrome 打印版网页 PDF；支撑 disaggregated inference、ingress-router / prefill / decode 解耦部署，对“算子下发”章节最关键 |
| `nvidia-agentic-inference-dynamo-2026-04-17.pdf` | https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/ | Chrome 打印版网页 PDF；支撑 agentic inference 的 WORM 式 KV 访问、85%-97% hit、97.2% aggregate hit、11.7x read/write ratio |
| `nvidia-vera-cpu-ai-factories-2026-03.pdf` | https://developer.nvidia.com/blog/nvidia-vera-cpu-delivers-high-performance-bandwidth-and-efficiency-for-ai-factories/ | Chrome 打印版网页 PDF；支撑 Vera CPU 的高带宽内存、NVLink-C2C 与“AI factory 控制平面”定位 |
| `storagereview-vera-rubin-ces-2026.pdf` | https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack | Chrome 打印版网页 PDF；支撑 Vera/Rubin/BlueField-4 一体化平台拓扑，以及相关图示 |
| `astera-cxl-kv-cache-2025-11.pdf` | https://www.asteralabs.com/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance/ | Chrome 打印版网页 PDF；支撑 CXL memory 扩展对 KV warm tier 的容量/经济性信号 |
| `kimi-agent-swarm-2026-04-11.pdf` | https://www.kimi.com/blog/agent-swarm.html | Chrome 打印版网页 PDF；支撑极宽 fan-out/fan-in、多 sub-agents 并行对 host CPU 调度宽度的压力 |
| `volcengine-mobile-use-agent-2026-04-29.pdf` | https://developer.volcengine.com/articles/7628489608359395369 | Chrome 打印版网页 PDF；支撑 Mobile Use Agent / 手机 GUI agent 的产品形态；报告中用于推断多模态 prefill 与高频状态切换 |
| `openclaw-readme-2026-04-14-window.md` | https://github.com/openclaw/openclaw | 支撑 OpenClaw 的 always-on、多入口、Android node / camera / canvas 等产品形态；报告中用于 workload 形态推断 |
| `anthropic-claude-code-subagents-current.pdf` | https://docs.anthropic.com/en/docs/claude-code/sub-agents | Chrome 打印版网页 PDF；作为补充证据，支撑 Claude Code subagents 的 separate context window 与额外上下文收集延迟 |

## Notes

- Anthropic 文档未暴露明确发布日期，在报告中只作为补充证据使用，不作为 `2025-07-01` 日期边界内的主证据。
- OpenClaw 与 Mobile Use Agent 两项主要用于 workload 形态推断，不单独作为底层 CPU 负载机制的直接证据。
