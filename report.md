# Agentic AI 推理机头 CPU 洞察：算子下发、KV 卸载与 MoE 场景

> **更新日期：** 2026-04-24  
> **资料时间边界：** 2025-07-01 及之后公开发表的论文、专利、产品发布与产业分析  
> **范围说明：** 本洞察聚焦推理（inference）阶段的主机 CPU（Host CPU / Head Node CPU）负载，而非训练或纯边缘端侧推理。

---

## 执行摘要

- **Agentic AI 正将系统瓶颈从 GPU 推向 CPU。** Georgia Tech 与 Intel 的联合研究（2025-11）表明，典型 Agentic 工作负载中工具处理占端到端延迟的 **50%–90.6%**；GPU 升级越快，瓶颈越迅速向 CPU 侧转移。
- **算子下发（Kernel Launch / Dispatch）已成为新的"调度墙"，且权重量化越激进，调度墙越明显。** 2026 年多项独立研究显示，在量化小模型或高并发服务场景中，单次前向传播可能发射数百个微秒级 Kernel，每个 Launch 约 2.5 μs 的 CPU 侧驱动开销成为主导延迟；CPU 核心竞争可将该开销从微秒级放大到毫秒级。vLLM V1 通过 Persistent Batch、Numpy 替代 Python Native 等重构，将吞吐提升 **1.7×**。
- **KV Cache 卸载到主机内存是长上下文推理的必选项，但 CPU-GPU 传输开销决定收益上限。** 2025 下半年至 2026 年的 NOSA、ScoutAttention、CXL 内存扩展方案显示，卸载系统需同时约束跨设备传输量并隐藏延迟；CXL 内存层可在生产建模中将 GPU 需求降低 **87%**。
- **MoE 推理在内存受限场景下触发专家权重卸载，CPU 成为专家调度的 Orchestrator。** 2026 年 Speculating Experts 与 FluxMoE 等研究指出，专家权重从 CPU DRAM 按需加载会制造严重的 CPU-GPU 传输瓶颈；基于内部表示的推测预取可将 TPOT 降低 **14%**。
- **PD 分离（Prefill-Decode Disaggregation）使机头 CPU 从"单节点调度器"升级为"跨池编排中枢"。** 2025 下半年至 2026 年，PD 分离已成为生产默认架构。机头 CPU 需要管理跨节点 KV Cache 传输（同节点 <0.1% 开销，跨节点需 **90 Gbps+**）、序列化/反序列化以及预填充池与解码池的动态负载均衡。
- **CPU:GPU 配比正发生结构性翻转。** 产业共识（NVIDIA GTC 2026、TrendForce、Arm）认为，传统 AI 数据中心 1:4–1:8 的 CPU:GPU 比例将向 **1:1–1:2** 演进；NVIDIA Vera CPU、AMD EPYC Turin、Intel Xeon 6/7 均针对 Agentic 编排与 RL 后训练场景强化单核性能与内存带宽。

---

## 概念边界

| 术语 | 定义 | 与本洞察的关联 |
|---|---|---|
| **算子下发 / Operator Dispatch** | CPU  host 进程通过 CUDA Runtime / Driver 向 GPU 提交 Kernel 并触发 MMIO Doorbell 写入的过程。在 LLM 推理中常被称为 Kernel Launch。 | CPU 单核性能、进程调度与 PCIe 延迟直接决定下发速度；高并发下 CPU Oversubscription 会导致 GPU 空闲等待。 |
| **KV 卸载 / KV Cache Offloading** | 将 Transformer 推理中产生的 Key-Value Cache 从 GPU HBM 转移到容量更大但带宽更低的主机 DRAM（或 SSD / CXL 内存扩展层），以支持更长上下文或更大 Batch。 | 卸载策略需要在"容量扩展"与"CPU-GPU 传输延迟"之间权衡；CPU 负责管理页表、异步调度和数据搬运。 |
| **MoE（Mixture of Experts）** | 稀疏激活架构，每 Token 仅激活少量专家（Expert）子网络，可在总参数量巨大的同时保持较低的单 Token 计算量。 | 当专家总数超过 GPU 显存容量时，未命中专家需从 CPU 内存加载；CPU 侧的路由预测、权重调度和 All-to-All 协调成为性能关键。 |
| **Agentic AI / 智能体 AI** | 以大模型为大脑、通过工具调用（Tool Use）、多步规划与反射迭代完成复杂任务的系统，区别于单轮问答式 LLM。 | 其执行模式为频繁的"GPU 推理 → CPU 工具执行/编排 → GPU 推理"交替，天然放大主机 CPU 的负载与延迟敏感性。 |

---

## 当前状态：厂商、产品与架构

### 1. NVIDIA Vera CPU — 专为 Agentic 推理设计的机头处理器

2026 年 3 月 GTC 上，NVIDIA 将 Vera CPU 从"GPU 附属品"重新定位为可独立部署的 Agentic 编排核心。这是本次洞察最具标志性的产品信号。

- **核心规格：** 88 颗定制 Olympus Armv9.2 核心，支持 NVIDIA Spatial Multithreading（SMT），单芯片 2270 亿晶体管；LPDDR5X 内存带宽达 **1.2 TB/s**；NVLink-C2C 与 GPU 互联带宽 **1.8 TB/s**。
- **Agentic 定位：** NVIDIA 官方将 Vera 定义为"AI Factories 的控制平面"，强调其在沙箱执行、RL 后训练反馈循环、代码编译与数据预处理中的低尾延迟表现，相比竞品沙箱性能提升 **50%**。
- **独立商业模式：** Meta 已签署大规模 Grace-only 部署协议并计划 2027 年引入 Vera；CoreWeave、Oracle、Alibaba、ByteDance 等云厂商将在 2026 下半年提供 standalone Vera CPU 实例。NVIDIA 预计该业务将成为"数十亿美元级别"收入线。

<img src="assets/nvidia-vera-cpu-architecture.png" alt="NVIDIA Vera CPU 架构概览" width="760">

> **图：** NVIDIA Vera CPU 架构与关键指标。88 颗 Olympus 核心与 1.2 TB/s LPDDR5X 内存带宽使其成为当前面向 Agentic AI 编排密度最高的机头 CPU 之一。来源：NVIDIA GTC 2026 / StorageReview。

### 2. AMD EPYC Turin & Intel Xeon 6/7 — x86 阵营的应对

- **AMD：** 2025 年 Q4 数据中心收入创纪录达 54 亿美元（同比 +39%），Lisa Su 明确将 EPYC CPU 增长归因于 Agentic 与新兴 AI 负载对"head nodes 和并行任务"的需求。第五代 EPYC Turin 在 2025 年底已占服务器 CPU 收入过半。
- **Intel：** Xeon 6+（Clearwater Forest，288 核）与 Xeon 7（Diamond Rapids，256 核）计划基于 Intel 18A 工艺发布，但受良率影响，大规模量产可能延迟至 2027 年。Intel 已承认"严重误判"数据中心 CPU 需求增速，正将晶圆产能从客户端转向服务器端。

### 3. 机头 CPU 产品横向对比：Vera vs EPYC Turin vs Xeon 6

截至 2026 年 Q2，三大厂商均发布了面向 Agentic AI 推理的机头 CPU 方案，但设计哲学差异显著：

| 指标 | NVIDIA Vera | AMD EPYC Turin | Intel Xeon 6 Granite Rapids |
|---|---|---|---|
| **核心架构** | 88 核 Olympus (Armv9.2)，单芯片单片设计 | 最高 192 核 Zen 5，Chiplet 设计 | 最高 128 核 P-core，Chiplet 设计 |
| **线程/并发** | 176 线程 (SMT，物理分区) | 384 线程 (SMT) | 256 线程 (HT) |
| **内存带宽** | **1.2 TB/s** LPDDR5X (~14 GB/s 每核) | ~614 GB/s DDR5 (~3.2 GB/s 每核) | ~307 GB/s DDR5 (~2.4 GB/s 每核) |
| **GPU 互联** | NVLink-C2C 1.8 TB/s | PCIe Gen5 x128 | PCIe Gen5 |
| **Agentic 相关实测** | 沙箱性能 **1.5×** 于 x86 竞品；Redpanda cross-core 吞吐 **+73%**；64 核后仍持续扩展 | 多线程吞吐领先，但 32 核后带宽饱和导致扩展平坦 | 单核频率高（5.0–5.7 GHz），延迟敏感型负载占优 |
| **独立部署** | 已确认 standalone 商业模式（Meta、CoreWeave 等） | 传统服务器市场主导 | 受 18A 良率影响，量产可能延迟至 2027 |
| **关键限制** | PCIe 控制器 reportedly 与第三方 GPU 存在兼容性问题 | 无 | 功耗与 TCO 略高于 Turin |

**关键洞察：**
- **Vera 的优势在于单芯片统一内存域 + 极高每核带宽**，对 Kernel Launch 密集、沙箱执行密集、KV Cache 调度的 Agentic 负载极为适配；但 88 核上限意味着纯核心数密集型负载（如大规模并行数据处理）仍可能落后于 192 核的 Turin。
- **AMD Turin 仍是核心密度与 TCO 冠军**，每美元吞吐量最高，但 Chiplet 架构在跨 CCD 通信时存在 NUMA 延迟，对需要频繁跨核同步的调度任务不利。
- **Intel Granite Rapids 单核频率最高**，在需要极低尾延迟的 tokenization、JSON 解析、API 序列化等串行任务上仍有优势，但内存带宽和核心密度双双落后。
- **机头 CPU 选型正在分层：** GPU 伴随型（co-located）推理节点可能优先 Vera（C2C 带宽 + 统一内存）；通用推理网关与纯 CPU 编排节点可能继续沿用 EPYC Turin（成本 + 软件生态）；对延迟极度敏感的边缘推理节点可能选择 Intel（高主频）。

### 4. CXL 内存扩展层 — KV Cache Offloading 的新介质

Astera Labs 的 Leo CXL Smart Memory Controller（2025-11 实测数据）显示，在生产级 LLM 推理负载中：
- KV Cache 存储于 CXL 内存可减少 GPU 需求达 **87%**；
- Prefill 阶段 GPU 利用率提升 **75%**；
- 每查询 CPU 利用率降低 **40%**；
- 系统可支持 **2 倍** 并发 LLM 实例。

这意味着 CXL 正在创造一种介于 GPU HBM 与主机 DRAM 之间的"内存带宽缓冲层"，对 KV 卸载场景具有直接的经济学意义。

---

## 技术评估：四大 CPU 瓶颈场景

### 场景一：算子下发（Operator Dispatch / Kernel Launch）

#### 发现 0：权重量化越激进，越容易撞上"调度墙"——一个反直觉的因果链
2026 年 3 月的一篇深度工程实测揭示了一个被忽视的新范式：当模型通过 IQ4/FP4 等激进量化手段被压缩到可完全驻留 GPU L2 Cache 时，内存带宽瓶颈消失，但**算子下发（Dispatch）瓶颈凸显**。一个 135M 参数的量化模型单次前向传播发射 301 个 Kernel，每个 Launch 约 2.5 μs，总计 **750 μs** 的纯下发税，几乎等于单 Token 总时间（792 μs）。Kernel Fusion 将发射次数降至 181 次后，吞吐提升 **20%**（1255 → 1508 tok/s）。

这一因果链对机头 CPU 的选型有直接影响：
- **量化降低显存压力 → 模型更小 → Batch 内可容纳更多请求 → Kernel 发射频率更高 → CPU 调度负载更重。**
- 2026 年 1 月的 LongCat-Flash-Lite 论文同样观察到，在轻量模型+大有效 Batch Size 场景下，瓶颈从内存带宽转向 Kernel Launch Overhead，需通过 extensive kernel fusion 与 NVIDIA PDL（Programmatic Dependent Launch）缓解。
- FlashNorm（2026-04）的微观分析指出，单次 Kernel Launch 在 A100 上约 **10–15 μs**，加上中间张量分配（~5 μs）和 HBM 往返，每次融合可节省 **15–25 μs** 固定开销——这部分开销与模型规模无关，纯粹由 CPU 驱动栈决定。

#### 发现 1：小模型和短序列下，"调度墙"已取代"内存墙"
2026 年 3 月的一篇深度工程博客（基于 RTX 5090 实测）指出，当量化模型可被 L2 Cache 容纳时，推理不再受限于内存带宽，而是受限于 Kernel Launch 开销。一个 135M 参数模型单次前向传播发射 301 个 Kernel，每个 Launch 约 2.5 μs，总计 **750 μs** 的纯下发税，占单 Token 时间预算（792 μs）的绝大部分。通过激进 Kernel Fusion 将发射次数降至 181 次，吞吐从 1255 tok/s 提升至 1508 tok/s。

#### 发现 2：CPU 竞争将微秒级开销放大为毫秒级集群停滞
2026 年 3 月论文《Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference》系统量化了该问题：
- vLLM 在 H100 上运行 Llama 3 8B 时，HTTP 服务占 **33%** 执行时间，调度+输入准备占 **29%**，GPU 实际计算仅 **38%**。
- 当 CPU 进程数超过可用核心数时，Kernel Launch 延迟从 μs 级恶化到 ms 级；在 NCCL 集合通信中，若某一 Rank 的 CPU 被抢占 1 ms，所有 GPU 忙等放大为集群级停滞。
- vLLM 的 `shm_broadcast.py` 广播队列在 5 req/s、100k Token 输入的 TP=4 场景下，dequeue 延迟从 12 ms 恶化到 **228 ms**（19 倍），是 GPU 单步解码时间（44 ms）的 5 倍以上。

#### 发现 3：推理引擎层面的 CPU 优化——以 vLLM V1 为例
2025 年 1 月发布的 vLLM V1（于 2025 下半年成为默认引擎）是一次针对机头 CPU 开销的系统性重构：
- **Persistent Batch：** 缓存输入张量，每步仅应用增量 diffs，避免每步重建张量的 Python 开销。
- **Numpy 替代 Python Native：** 在调度器与数据准备路径上用 Numpy 操作替代原生 Python，显著降低 CPU 占用。
- **Zero-Overhead Prefix Caching：** 即使 Cache 命中率为 0%，吞吐损失也 < 1%，消除了 V0 中因前缀缓存数据结构导致的 CPU 瓶颈。
- **多模态输入预处理 Offload：** 将图像解码与预处理放到独立进程，避免阻塞 GPU Worker。
- **Piecewise CUDA Graphs：** 在保持动态调度能力的同时，尽可能捕获静态子图的 CUDA Graph，减少重复 Kernel Launch。

实测显示，V1 在文本模型上吞吐比 V0 提升最高 **1.7×**；在视觉语言模型（Qwen2-VL）上提升更为显著。vLLM 2026 Q1 Roadmap 进一步将 "Python overhead reduction"、"CPU KV cache production ready" 与 "disaggregated prefilling" 列为重点，表明社区已明确意识到机头 CPU 是下一阶段的优化主战场。

#### 发现 4：缓解路径汇总
- **Kernel Fusion：** 将残差相加、RMSNorm、RoPE、KV Cache 写入、通信操作（AllReduce + Residual Add + RMSNorm）融合为单 Kernel，减少 Launch 次数。
- **Persistent Kernel / Megakernel：** Event Tensor（2026-04）将动态控制流编码为 Tile 级依赖图，生成跨算子的持久化 Kernel，消除跨 Kernel 边界同步。
- **PDL（Programmatic Dependent Launch）：** NVIDIA 2026 年技术允许有依赖关系的 Kernel 提前触发、重叠执行间隙，提升 SM 利用率。
- **推理引擎重构：** vLLM V1 的 Persistent Batch、Zero-Copy DMA、前缀缓存优化与进程结构扁平化。
- **CPU 核心扩容与隔离：** 确保每 GPU 配有足够且隔离的 Host 核心，避免 OS 调度器介入关键路径；多模态预处理等任务应 offload 到独立进程。

### 场景二：KV 卸载（KV Cache Offloading）

#### 发现 1：长上下文必然导致 KV Cache 超出 GPU 显存
以 Qwen3-32B FP16、64K 序列为例，KV Cache 占用约 **16 GB**。在多轮 Agentic 交互或长文档推理中，Cache 规模随 Batch Size 与序列长度线性增长，显存溢出成为常态。将 KV Cache 卸载到主机 DRAM 或 CXL 内存是标准解法，但引入了每解码步的 CPU-GPU 传输开销。

#### 发现 2：稀疏化 + 卸载是 2025 下半年以来的主攻方向
- **NOSA（2025-10，arXiv）：** 首个"原生为 KV Cache Offloading 设计"的可训练稀疏注意力机制。它显式约束 CPU-GPU KV 传输量，在 1B/3B/8B 模型上相比全注意力实现最高 **5.04×** 解码吞吐提升，相比 InfLLMv2 和 ShadowKV 分别提升 **1.92×** 和 **1.83×**。
- **ScoutAttention（2026-03，arXiv）：** 提出 Layer-Ahead CPU Pre-computation 算法，让 CPU 提前一层启动 Attention 计算，并通过异步周期性召回机制保持极低 CPU 负载。在保持精度损失 < 2.4% 的前提下，相比现有卸载方法实现 **2.1×** 加速。
- **CoMEM（2025，OpenReview）：** 针对 Agentic 长上下文，将历史压缩任务卸载到轻量级异步记忆模型，通过 k-step-off Pipeline 重叠记忆摘要与 Agent 执行，解码开销降低 **1.4×**。

#### 发现 3：CPU 在卸载系统中的角色转变
CPU 不再只是"数据搬运工"，而是 KV Cache 的**分级内存管理者**：
- **页级调度：** 类似操作系统 Swap，决定哪些 KV 页驻留 GPU、哪些降级到主机/CXL/SSD。
- **检索与预取：** 在稀疏注意力场景中，CPU 需动态估计 Token 重要性并预取相关 KV 块；若每步都触发检索，累积开销会抵消卸载收益。
- **协同计算：** ScoutAttention 等方案让 CPU 直接参与部分 Attention 计算，而非单纯传输数据，这要求机头 CPU 具备更强的向量/矩阵运算能力。

### 场景三：PD 分离（Prefill-Decode Disaggregation）与机头 CPU 调度负载

#### 发现 1：PD 分离已从研究概念变为 2025–2026 年生产默认架构
2024 年的 DistServe 与 Splitwise 首次系统论证了 PD 分离的收益，而到 2025 年底，Hao AI Lab 的回顾性分析确认该架构已成为"几乎每个主要 LLM 服务栈的默认手册"。vLLM、SGLang、NVIDIA Dynamo、TensorRT-LLM 与 llm-d 均已原生支持 PD 分离。

对机头 CPU 而言，PD 分离意味着**调度器不再只管理单节点 GPU，而是需要跨节点协调 KV Cache 的序列化、传输与反序列化**。vLLM 的 `vllm/distributed/kv_transfer` 模块通过 Connector + LookupBuffer + Pipe 三层抽象实现跨实例 KV 搬运；TensorRT-LLM 则在预填充节点与解码节点之间通过网络层传输 KV Cache Block。

#### 发现 2：KV Cache 传输开销高度依赖 CPU 侧网络栈与调度效率
- **同节点 NVLink：** DistServe 报告传输开销 < 总服务时间的 **0.1%**，可忽略。
- **跨节点网络：** Splitwise 计算表明，OPT-66B 在 512 Token 输入下产生约 **1.13 GB** KV Cache；若请求率达到 10 req/s，需约 **90 Gbps** 带宽才能避免瓶颈。
- **llm-d 0.5（2026-02）的 UCCL Backend：** 采用 host-resident software transport stack，由 CPU 管理传输逻辑而非完全依赖硬件卸载，在网络拥塞下尾延迟恶化仅 **7.1%**（对比 UCX 的 17.1%），验证了机头 CPU 在拥塞控制中的关键作用。

#### 发现 3：Agentic 长交互进一步放大 CPU 调度压力
Agentic 工作负载通常表现为**短输入 + 极长输出**（多轮工具调用后的推理链），这意味着 decode 阶段持续时间远超 prefill。PD 分离后，decode 池需要长时间维持大量并发流的 KV Cache 状态，而 prefill 池则需快速处理频繁到达的新工具调用结果。机头 CPU 的调度器必须在两个池之间做动态负载均衡，并处理 KV Cache 的跨池预热、迁移与回收。vLLM 2026 Q1 Roadmap 明确将"CPU KV cache production ready"和"disaggregated prefilling & KV transfer support"列为核心目标，侧面反映了 CPU 侧调度复杂度正在快速上升。

---

### 场景四：MoE（Mixture of Experts）推理

#### 发现 1：专家权重卸载是内存受限部署的必然选择
以 DeepSeek-R1（671B 总参 / 37B 激活参）为例，单节点 GPU 无法容纳全部专家权重。当专家权重被卸载到 CPU 内存时，每次 Token 路由命中冷专家都会触发同步 CPU→GPU 拷贝，成为解码阶段的决定性瓶颈。

#### 发现 2：推测预取与异步流水线是 2026 年的主要突破
- **Speculating Experts（2026-03，arXiv）：** 利用当前层已计算的内部表示（归一化残差流 + 默认向量）推测下一层将激活的专家，实现权重预取与 GPU 计算的重叠。在 Qwen-30B-A3B 等模型上，相比按需加载实现 **14%** 的 TPOT 降低。若推测执行精度不足，还可叠加轻量级估计器提升命中率。
- **FluxMoE（2026-04，arXiv）：** 解耦"逻辑专家身份"与"物理驻留位置"，通过带宽均衡的存储层次（压缩 GPU 内存 + 主机 DRAM）动态流式化参数，摆脱对路由预测准确率的依赖。
- **中国科学技术大学专利（2025，CN）：** 提出异步并行推理方法，将 GPU 计算与 Expert Parallelism 固有的 All-to-All 通信解耦，允许 Token 数据通信与模型计算异步并行；同时策略性将热点专家常驻 GPU、冷点专家卸载 CPU。

#### 发现 3：CPU 在 MoE 中的三重负载
1. **权重搬运：** PCIe / C2C 带宽有限，CPU 负责将专家权重从主机内存拷贝到 GPU。
2. **路由协调：** All-to-All 集合通信的同步信号由 CPU 侧进程驱动；若任一 Rank 的 CPU 延迟，全网 GPU 等待。
3. **负载均衡与调度：** 动态专家剪枝、容量因子调整、冷热专家分级策略均需在 CPU 侧实时决策。

---

## 证据表

| 时间 | 标题 | 类型 | 主题 | 主要发现 | 关联场景 |
|---|---|---|---|---|---|
| 2025-11 | Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective | 论文 (arXiv) | Agentic AI CPU 瓶颈量化 | 工具处理占 E2E 延迟 50%–90.6%；CPU 并行效率远低于 GPU；提出 COMB/MAS 调度优化。 | Agentic 编排 |
| 2026-03 | Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference | 论文 (arXiv) | CPU 竞争导致 Kernel Launch 延迟 | CPU Oversubscription 使 dequeue 延迟放大 19×；GPU 利用率可降至 70% 以下。 | 算子下发 |
| 2026-03 | What Actually Bottlenecks LLM Inference on Modern GPUs | 工程博客 | Kernel Launch 税（量化小模型） | 301 次 Kernel Launch 中纯驱动开销占 750 μs；IQ4 量化+L2 Pinning 后内存墙消失，调度墙凸显。 | 算子下发 |
| 2026-01 | LongCat-Flash-Lite: Scaling Embeddings | 论文 (arXiv) | 轻量模型 Kernel Launch 瓶颈 | Extreme sparsity + large batch 使瓶颈从内存带宽转向 Kernel Launch Overhead。 | 算子下发 |
| 2025-01 | vLLM V1 Alpha Release | 引擎发布 | CPU 开销重构 | Persistent Batch、Zero-Overhead Prefix Caching、Numpy 替代 Python Native；吞吐提升 1.7×。 | 算子下发 |
| 2026-02 | llm-d v0.5: UCCL Backend for PD Transfer | 工程博客 | PD 分离网络传输 | Host-resident software transport 由 CPU 管理拥塞控制，网络压力下尾延迟恶化仅 7.1%（vs UCX 17.1%）。 | PD 分离 |
| 2026-03 | Disaggregated Serving in TensorRT-LLM | 厂商技术博客 | PD 分离生产实践 | 分离上下文与生成阶段以消除干扰；KV Cache Block 跨节点传输成为关键路径。 | PD 分离 |
| 2025-11 | Disaggregated Inference: 18 Months Later | 技术回顾 | PD 分离 Adoption | PD 分离已成为"几乎每个主要 LLM 服务栈的默认手册"。 | PD 分离 |
| 2026-04 | Event Tensor: Dynamic Megakernels for LLM Serving | 论文 (MLSys 审稿) | 持久化 Kernel 消除 Launch 开销 | 将动态形状与数据依赖编码为 Tile 依赖图，生成 Persistent Kernel，显著降低系统 Warmup 与 Launch 开销。 | 算子下发 |
| 2025-10 | NOSA: Native and Offloadable Sparse Attention | 论文 (arXiv) | 稀疏注意力 + KV 卸载 | 显式约束 CPU-GPU KV 传输量，解码吞吐最高提升 5.04×。 | KV 卸载 |
| 2026-03 | ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation | 论文 (arXiv) | 协同式 GPU-CPU Attention | CPU 提前一层预计算 Attention，异步召回，精度损失 < 2.4%，加速 2.1×。 | KV 卸载 |
| 2025-11 | How CXL Transforms RAG and KV Cache Performance | 厂商技术博客 (Astera Labs) | CXL 内存扩展实测 | CXL 内存使 GPU 需求降低 87%，GPU 利用率提升 75%，每查询 CPU 利用率降低 40%。 | KV 卸载 |
| 2026-03 | Speculating Experts Accelerates Inference for Mixture-of-Experts | 论文 (arXiv) | MoE 专家权重预取 | 利用内部表示推测未来专家，重叠 CPU-GPU 传输与计算，TPOT 降低 14%。 | MoE |
| 2026-04 | FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving | 论文 (arXiv) | MoE 专家动态驻留 | 解耦逻辑专家身份与物理驻留，动态流式化参数，无需依赖路由预测。 | MoE |
| 2025 | 异步并行推理方法专利 (CN) | 专利 | MoE 分布式通信优化 | GPU 计算与 All-to-All 通信异步并行；热点专家常驻 GPU，冷点卸载 CPU。 | MoE |
| 2026-03 | NVIDIA GTC 2026 — Vera CPU / Rubin Platform | 产品发布 | 机头 CPU 架构 | Vera 88 核 Olympus，1.2 TB/s 内存带宽，独立部署商业模式确立。 | 机头 CPU |
| 2026-03 | NVIDIA Vera CPU: Performance compared to AMD and Intel x86 chips | 产业评测 | 机头 CPU 横向对比 | Vera sandbox 性能 1.5× 于 x86；Redpanda cross-core 吞吐 +73%；64 核后持续扩展。 | 机头 CPU |
| 2026-02 | CPUs the New Bottleneck of the Agentic AI Era | 产业分析 | CPU 供需格局 | CPU:GPU 比例预计从 1:4–1:8 转向 1:1–1:2；Intel/AMD 已涨价 10%+，交货周期拉长至 6–10 周。 | 产业格局 |
| 2026-01 | AI Inference Optimization Techniques (Zylos Research) | 技术综述 | MoE 效率与前沿模型 | DeepSeek-R1 671B 总参仅 37B 激活；MoE 使每推理计算量减少 90–95%。 | MoE |

---

## 图表附录

### 图 1：NVIDIA Vera Rubin 六芯片协同架构

<img src="assets/nvidia-vera-rubin-6chips.png" alt="NVIDIA Vera Rubin 六芯片架构" width="760">

> **图：** Vera Rubin 平台采用"极端协同设计"，将 Vera CPU、Rubin GPU、NVLink 6 Switch、ConnectX-9、BlueField-4 DPU 与 Spectrum-6 以太网交换机构建为统一系统。Vera CPU 作为编排与内存中枢，直接决定 Agentic 工作流的延迟与 GPU 利用率。来源：NVIDIA GTC 2026 / StorageReview。

### 图 2：BlueField-4 DPU — 卸载网络、存储与安全以释放 Vera CPU

<img src="assets/nvidia-bluefield4.png" alt="BlueField-4 DPU 架构" width="760">

> **图：** BlueField-4 集成 64 核心 CPU 与 ConnectX-9 SuperNIC，将网络、存储和安全处理从 Vera CPU 与 Rubin GPU 上卸载，使机头 CPU 能专注于 Agentic 编排与 Kernel 调度。来源：NVIDIA GTC 2026 / StorageReview。

---

## 来源列表

### 学术论文与预印本
1. Raj, R., et al. "Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective." arXiv:2511.00739, v3 (Apr 2026). https://arxiv.org/abs/2511.00739
2. "Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference." arXiv:2603.22774 (Mar 2026). https://arxiv.org/abs/2603.22774
3. "What Actually Bottlenecks LLM Inference on Modern GPUs." AI.rs (Mar 2026). https://ai.rs/ai-developer/memory-wall-disappears-llm-inference-bottlenecks
4. "Event Tensor: Dynamic Megakernels for LLM Serving." Under review at MLSys (Apr 2026). https://arxiv.org/abs/2604.13327
5. Huang, Y., et al. "NOSA: Native and Offloadable Sparse Attention." arXiv:2510.13602 (Oct 2025). https://arxiv.org/abs/2510.13602
6. Zhang, Q., et al. "ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation." arXiv:2603.27138 (Mar 2026). https://arxiv.org/abs/2603.27138
7. Madan, V., et al. "Speculating Experts Accelerates Inference for Mixture-of-Experts." arXiv:2603.19289 (Mar 2026). https://arxiv.org/abs/2603.19289
8. "FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving." arXiv:2604.02715 (Apr 2026). https://arxiv.org/abs/2604.02715
9. "CoMEM: A Decoupled Agent Framework with Asynchronous Memory Compression." OpenReview (2025). https://openreview.net/pdf?id=c7c6541f58ddaf647289d2523a9587312294301a

### 厂商官方资料
10. "NVIDIA Vera CPU Delivers High Performance, Bandwidth, and Efficiency for AI Factories." NVIDIA Developer Blog (Mar 2026). https://developer.nvidia.com/blog/nvidia-vera-cpu-delivers-high-performance-bandwidth-and-efficiency-for-ai-factories/
11. "NVIDIA Vera CPU Enters Full Production." Data Center Dynamics (Mar 2026). https://www.datacenterdynamics.com/en/news/nvidia-vera-cpu-enters-full-production-pitched-at-agentic-ai-workloads/
12. "NVIDIA Launches Vera Rubin Architecture at CES 2026." StorageReview (Jan 2026). https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack

### 产业与市场分析
13. "How Agentic AI Is Reshaping the CPU:GPU Ratio." TrendForce Insights (Apr 2026). https://insights.trendforce.com/p/agentic-ai-cpu-gpu
14. "The Forgotten Chip: CPUs the New Bottleneck of the Agentic AI Era." Uncover Alpha (Feb 2026). https://www.uncoveralpha.com/p/the-forgotten-chip-cpus-the-new-bottleneck
15. "Secret Agent CPU." The Diligence Stack / Ben Bajarin (Mar 2026). https://thediligencestack.com/p/secret-agent-cpu
16. "In the Age of Agentic, the CPU is the New Bottleneck." rmmod.com (Mar 2026). https://rmmod.com/posts/agent/agentic-cpu-bottleneck/

### 技术方案与专利
17. "How CXL Transforms RAG and KV Cache Performance." Astera Labs (Nov 2025). https://www.asteralabs.com/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance/
18. "KV cache offloading — CPU RAM vs. storage." NetApp Community (Nov 2025). https://community.netapp.com/t5/Tech-ONTAP-Blogs/KV-cache-offloading-CPU-RAM-vs-storage/ba-p/464463
19. "MoE inference cost cuts: 30+ patents analyzed." PatSnap (Apr 2026). https://www.patsnap.com/resources/blog/articles/moe-inference-cost-cuts-30-patents-analyzed/
20. "AI Inference Optimization Techniques (2025-2026)." Zylos Research (Jan 2026). https://zylos.ai/research/2026-01-11-ai-inference-optimization

---

> **免责声明：** 本洞察基于 2025-07-01 至 2026-04-24 期间公开发表的技术论文、厂商公告与产业分析整理而成。涉及尚未量产的产品（如 NVIDIA Vera Rubin 大规模部署、Intel Xeon 7 等）时间表存在延期风险；性能数据来源于论文或厂商受控测试，实际部署收益取决于具体工作负载与系统配置。
