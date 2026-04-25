## 1. 引言：Agentic AI 如何重新定义系统瓶颈

近两年，大模型推理系统的优化重点经历了显著迁移。早期工作主要关注 GPU 侧的算力利用率、注意力算子实现和显存容量边界；而在 agentic AI 兴起之后，系统行为从"单次请求、连续 decode"转向"多阶段推理、状态保留、外部中断、上下文复用与多代理并发"的复合执行模式。这一转变并非渐进式改良，而是从根本上改变了系统瓶颈的空间分布——**瓶颈正在从 GPU 内部外溢到 host 侧编排链路**。

### 1.1 为什么瓶颈会外溢：GPU 效率提升的副作用

一个反直觉的事实是：GPU 越高效，CPU 越容易成为瓶颈。Georgia Tech 与 Intel 的联合研究（2025-11）表明，典型 agentic 工作负载中工具处理占端到端延迟的 **50%–90.6%**；GPU 升级越快，瓶颈越迅速向 CPU 侧转移 [1]。这不是因为 CPU 变弱了，而是因为 GPU 的计算速度提升超过了 host 侧编排速度的提升，导致两者之间出现了越来越大的"能力鸿沟"。

DeepSeek V4（2026-04-24 发布）的架构设计从另一个角度验证了这一趋势：其 Engram 架构明确将**静态知识检索模块放在 CPU RAM**，动态推理模块放在 GPU [30]。这一设计并非权宜之计，而是基于一个经济学事实——CPU RAM 每 GB 成本比 GPU HBM 低 **10–20 倍**。当模型规模达到 1.6T 参数时，将所有知识放在 GPU 上既不经济也不必要。Engram 的提出标志着工业界已经默认：**CPU 侧不再是 GPU 的被动跟随者，而是与 GPU 并列的、承担不同职责的计算层**。

### 1.2 四条技术主线的同时收敛

现有材料可以归纳为四条同时收敛的技术主线，任何一条都足以单独抬高 CPU 地位，而 agentic workload 让它们同时出现：

**算子下发从"发命令"变成"编排状态机"**。传统 serving 假设请求是"单上下文、长 decode、稳定批次"，但 agentic inference 表现为 prefill → decode → 暂停 → 恢复 → 分叉 → 合并的复合执行模式。每个阶段切换都需要 host CPU 做 request state transition、worker affinity 决策、KV object 生命周期跟踪。权重量化越激进（IQ4/FP4），模型越小，Batch 内可容纳的请求越多，Kernel 发射频率越高，CPU 调度负载反而越重。

**KV 卸载从"容量兜底"变成"生命周期管理"**。NVIDIA Dynamo 数据显示 agentic inference cache hit 可达 **85%–97%**，read/write ratio 高达 **11.7x** [9]。系统价值重心从"写新 KV"转向"保留、路由、预取和恢复旧 KV"，CPU 内存已从 spill 层升级为 warm tier。DeepSeek V4 的 CSA（Compressed Sparse Attention）4x 压缩 + HCA（Heavily Compressed Attention）128x 压缩进一步将 KV cache 降至 V3.2 的 **10%** [30]，但这一压缩收益的实现本身依赖 CPU 侧的索引构建和检索调度。

**MoE 从"稀疏计算优势"变成"host-side orchestration 压力"**。DeepSeek V4（1.6T 总参 / 49B 激活参）单节点无法容纳全部专家，冷专家命中会触发同步 CPU→GPU 拷贝。CPU 承担权重搬运、路由协调和拓扑感知负载均衡三重职责。2026 年的 FineMoE [31]、SpecMoEOff [32] 等系统进一步证明，专家预取的准确率直接决定系统吞吐。

**PD 分离把 CPU 从"单节点调度器"升级为"跨池编排中枢"**。Hao AI Lab 2025-11 回顾确认 PD 分离已成为"几乎每个主要 LLM 服务栈的默认手册"。机头 CPU 需要管理跨节点 KV Cache 传输（同节点 <0.1% 开销，跨节点需 **90 Gbps+**）、序列化/反序列化以及预填充池与解码池的动态负载均衡。Agentic batch inference 还引入了 **middle-phase thrashing** 问题——异步推进的 agent 其暂时不活跃的 KV 被 LRU 驱逐，恢复时需反复重算或传输 [33]。

### 1.3 核心数据信号

NVIDIA 2026 年 4 月的 Dynamo agentic inference 数据显示，在 agentic workload 中，后续调用的 cache hit 可达 **85%–97%**，4 个 teammate agent 聚合后可到 **97.2%**，累计 **read/write ratio 为 11.7x** [9]。这意味着系统的价值重心从"多写一点新 KV"转到"把旧状态留住、路由对、提前取回、避免重算"。

![图1 Agentic inference KV读写比](assets/nvidia-dynamo-agentic-kv-readwrite-2026.webp)

**图1** Agentic inference 的 KV 读写关系。累计读取明显快于累计写入，说明 agentic workload 的核心压力正从"持续写入新状态"转向"保留、路由、预取与恢复既有状态"。来源：NVIDIA, 2026-04-17 [9]。

本文基于 现有报告、图表与引用材料，结合 DeepSeek V4 技术报告和 2026 年最新研究成果，将现有研究归纳为上述四条相互耦合的技术主线，并结合真实产品工作负载与平台演化信号，对机头 CPU 的角色、瓶颈与选型做出系统性判断。

