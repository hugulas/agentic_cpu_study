# Agentic AI 推理机头 CPU 洞察：算子下发、KV 卸载、MoE 与真实工作负载反推

> **更新日期：** 2026-04-24  
> **资料时间边界：** 2025-07-01 及之后公开发布的论文、专利、产品发布与产业分析  
> **范围说明：** 本文聚焦 GPU 推理节点上的 host CPU / control-plane CPU（"机头 CPU"），重点关注三类场景：`算子下发（route / prefill / decode / transfer dispatch）`、`KV 卸载`、`MoE expert dispatch`。不讨论 2025 年 7 月之前的旧论文结论，除非仅用于术语解释。

---

## 执行摘要

- **Agentic AI 正将系统瓶颈从 GPU 推向机头 CPU。** Georgia Tech 与 Intel 的联合研究（2025-11）表明，典型 Agentic 工作负载中工具处理占端到端延迟的 **50%–90.6%**；GPU 升级越快，瓶颈越迅速向 CPU 侧转移。
- **算子下发（Operator Dispatch）已从"单机 kernel launch"演化为"分离式推理编排"，且权重量化越激进，"调度墙"越明显。** 2026 年多项独立研究显示，在量化小模型或高并发服务场景中，单次前向传播可能发射数百个微秒级 Kernel，每个 Launch 约 2.5 μs 的 CPU 侧驱动开销成为主导延迟；CPU 核心竞争可将该开销从微秒级放大到毫秒级，导致多 GPU 集群产生级联等待。vLLM V1 通过 Persistent Batch、Numpy 替代 Python Native 等重构，将吞吐提升 **1.7×**。
- **KV Cache 卸载到主机内存是长上下文推理的必选项，但 CPU-GPU 传输开销决定收益上限。** 2025 下半年至 2026 年的 NOSA、ScoutAttention、CXL 内存扩展方案显示，卸载系统需同时约束跨设备传输量并隐藏延迟；NVIDIA Dynamo 给出的 Agentic 推理 cache hit 可达 **85%–97%**，read/write ratio 高达 **11.7x**，使 KV 管理从"容量问题"升级为"生命周期问题"。CXL 内存层可在生产建模中将 GPU 需求降低 **87%**。
- **MoE 推理在内存受限场景下触发专家权重卸载，CPU 成为专家调度的 Orchestrator。** 2026 年 Speculating Experts 与 FluxMoE 等研究指出，专家权重从 CPU DRAM 按需加载会制造严重的 CPU-GPU 传输瓶颈；基于内部表示的推测预取可将 TPOT 降低 **14%**。NVIDIA Wide EP（2025-12）进一步将 host 侧压力从单请求驱动推向批级路由、拓扑感知放置和跨 GPU/跨节点数据流协同。
- **PD 分离（Prefill-Decode Disaggregation）使机头 CPU 从"单节点调度器"升级为"跨池编排中枢"。** 2025 下半年至 2026 年，PD 分离已成为生产默认架构。机头 CPU 需要管理跨节点 KV Cache 传输（同节点 <0.1% 开销，跨节点需 **90 Gbps+**）、序列化/反序列化以及预填充池与解码池的动态负载均衡。llm-d 0.5 的 UCCL Backend 验证了 CPU 管理 host-resident 传输栈可将网络拥塞下尾延迟恶化控制在 **7.1%**（对比 UCX 的 17.1%）。
- **真实 Agentic 产品形态正在暴露三项传统 serving 论文容易忽略的 CPU 需求：**
  - **高频 prefill-first 调度**（OpenClaw / 手机 GUI Agent 的多模态截图输入 + 短回合切换）
  - **多上下文并存管理**（Claude Code subagents 的独立 context window）
  - **极宽 fan-out/fan-in 瞬时并发**（Kimi Agent Swarm 的 100 sub-agents）
- **CPU:GPU 配比正发生结构性翻转。** 产业共识（NVIDIA GTC 2026、TrendForce、Arm）认为，传统 AI 数据中心 1:4–1:8 的 CPU:GPU 比例将向 **1:1–1:2** 演进；NVIDIA Vera CPU（88 核 Olympus / 1.2 TB/s LPDDR5X）、AMD EPYC Turin（192 核 / 614 GB/s）、Intel Xeon 6/7 均针对 Agentic 编排场景强化内存带宽与一致性互连。Vera 的 Agentic sandbox 性能达到 x86 竞品的 **1.5×**，Redpanda cross-core 吞吐领先 **73%**。

---

## 概念边界

### 1. 什么叫"算子下发"

本文中的"算子下发"不是编译器语境里的 kernel fusion，而是更接近推理系统运行期的 host 侧下发与编排，包括：

- request 到达后的 `router / scheduler` 决策
- `prefill` 与 `decode` 的解耦调度（PD 分离）
- KV block 的 `offload / reload / prefetch`
- MoE token/expert 的分发和跨设备搬运
- RDMA / NVLink / NIXL / storage 传输的触发与管线化
- 底层 CUDA Kernel 从 CPU host 进程向 GPU 的 launch 与同步

### 2. 为什么 agentic AI 会抬高 host CPU 重要性

2026-04-17 的 NVIDIA Dynamo agentic inference 文章给出一组很关键的数据：在 agentic workload 中，后续调用的 cache hit 可达 **85%-97%**，4 个 teammate agent 聚合后可到 **97.2%**，累计 **read/write ratio 为 11.7x**。这意味着系统的价值重心从"多写一点新 KV"转到"把旧状态留住、路由对、提前取回、避免重算"。

<img src="assets/nvidia-dynamo-agentic-kv-readwrite-2026.webp" alt="Agentic inference 中 KV 累积读取显著高于写入" width="760">

> **图：** NVIDIA 给出的 agentic inference KV 读写曲线，说明 agentic 工作负载已经明显呈现 `write-once-read-many`。这直接抬高了 host 侧在保留、路由、预取和恢复上的权重。  
> 来源：NVIDIA, *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo*, 2026-04-17

### 3. 相邻概念辨析

| 术语 | 定义 | 与本洞察的关联 |
|---|---|---|
| **Kernel Launch / Dispatch** | CPU host 进程通过 CUDA Runtime / Driver 向 GPU 提交 Kernel 并触发 MMIO Doorbell 写入的过程。 | CPU 单核性能、进程调度与 PCIe 延迟直接决定下发速度；高并发下 CPU Oversubscription 会导致 GPU 空闲等待。 |
| **KV 卸载 / KV Cache Offloading** | 将 Transformer 推理中产生的 Key-Value Cache 从 GPU HBM 转移到容量更大但带宽更低的主机 DRAM（或 SSD / CXL 内存扩展层），以支持更长上下文或更大 Batch。 | 卸载策略需要在"容量扩展"与"CPU-GPU 传输延迟"之间权衡；CPU 负责管理页表、异步调度和数据搬运。 |
| **MoE（Mixture of Experts）** | 稀疏激活架构，每 Token 仅激活少量专家（Expert）子网络，可在总参数量巨大的同时保持较低的单 Token 计算量。 | 当专家总数超过 GPU 显存容量时，未命中专家需从 CPU 内存加载；CPU 侧的路由预测、权重调度和 All-to-All 协调成为性能关键。 |
| **PD 分离 / Prefill-Decode Disaggregation** | 将计算密集的 prefill 阶段与内存带宽密集的 decode 阶段物理分离到不同 GPU 池，各自独立优化与扩展。 | 分离后机头 CPU 需承担跨节点 KV Cache 传输协调、序列化/反序列化、池间负载均衡等新增职责。 |

---

## 现状判断：机头 CPU 已进入推理关键路径

### 1. 算子下发正在从"单机 kernel launch"演化为"分离式推理编排"

2026-03-23 的 NVIDIA Kubernetes 文章把 `disaggregated LLM inference` 明确拆成 `ingress-router`、`prefill worker`、`decode worker`，并用 NIXL 负责节点间高吞吐数据传输。这个拆法本身就说明，host 侧需要承担的事情已经不仅是把请求扔给一张 GPU，而是：

- 决定请求该进哪个 router
- 把 prefill 产物和 KV 状态送到后续 decode worker
- 配合调度器维持 worker affinity、拓扑感知和恢复路径
- 让 Kubernetes / service mesh / transfer library 与推理 runtime 连成一条低抖动数据面

<img src="assets/nvidia-k8s-disagg-serving-2026.webp" alt="Disaggregated LLM inference 将 ingress router、prefill 和 decode 分开部署" width="760">

> **图：** NVIDIA 在 Kubernetes 上展示的解耦式推理拓扑。它反映出 host 侧职责从"单机发命令"扩展为 `router + stage scheduling + transfer orchestration`。  
> 来源：NVIDIA, *Deploying Disaggregated LLM Inference Workloads on Kubernetes*, 2026-03-23

对 CPU 的直接含义是：

- **单核尾延迟要稳**：router、状态机、完成队列和中断处理都怕抖动
- **核数不能过低**：agentic workload 带来更多并发的短控制任务，而非单一长核函数
- **PCIe / NIC / GPU 拓扑要友好**：否则 dispatch 决策和实际数据流不一致
- **软件栈要轻**：K8s、runtime、transfer library、监控 agent 本身都会吃 host 资源

### 2. NIXL / 数据搬运把 host CPU 从"控制面"推向"轻数据面"

2026-03-09 的 NVIDIA Inference Transfer Library（NIXL）文章强调它支持 `RDMA, NVLink, TCP sockets, NVMe-oF`，目标是给分布式推理提供统一的数据移动抽象。这里的重点不是"又多了一个通信库"，而是：

- agentic inference 的真实瓶颈正在转向 **跨阶段、跨节点的状态搬运**
- host CPU 需要持续驱动 completion、buffer lifecycle、memory registration 和异步管线
- 如果 CPU 侧内存子系统、IOMMU 或中断处理不稳，GPU 闲等会直接放大

基于这些资料，可以做出一个稳健推断：  
对 agentic 推理节点，**host CPU 的价值正在从"发命令"向"编排+搬运+恢复"三合一角色迁移**。

### 3. 平台信号：Vera、Rubin 与 BlueField-4 说明 CPU 正被前移为控制平面

2026 年的平台设计已经开始把 host CPU 明确当成 agentic / inference orchestration 的控制平面。其中最值得吸收的是两点：

- **Vera CPU 的设计方向**：88 核 Olympus、`1.2 TB/s` LPDDR5X、`1.8 TB/s` NVLink-C2C，显示厂商正在用"高每核带宽 + 高一致性互连"去服务机头调度，而不只是追传统通用 CPU 指标。
- **BlueField-4 / SuperNIC 的协同方向**：网络、存储与安全处理进一步从 host CPU 旁路出去，意味着平台设计已经默认"CPU 应该把预算优先花在推理编排，而不是杂项数据面任务上"。

这类信号本身不直接证明某个具体 serving 栈一定会获得多少推理收益，但它强化了主判断：  
**机头 CPU 已经不是 GPU 服务器里的配套部件，而是在新平台里被当成第一层编排器来定义。**

<img src="assets/nvidia-vera-cpu-architecture.png" alt="NVIDIA Vera CPU 架构概览" width="760">

> **图：** Vera CPU 的高带宽内存与高带宽 CPU-GPU 互连，说明平台设计正在围绕 agentic orchestration 的控制平面需求展开。  
> 来源：NVIDIA, *NVIDIA Vera CPU Delivers High Performance, Bandwidth, and Efficiency for AI Factories*, 2026-03

<img src="assets/nvidia-vera-rubin-6chips.png" alt="NVIDIA Vera Rubin 六芯片协同架构" width="760">

> **图：** Vera、Rubin、NVLink Switch、BlueField-4 和以太网交换机构成更强耦合的平台，反映出机头 CPU 在系统级编排中的位置前移。  
> 来源：StorageReview, *NVIDIA Launches Vera Rubin Architecture at CES 2026*, 2026-01

<img src="assets/nvidia-bluefield4.png" alt="BlueField-4 DPU 架构" width="760">

> **图：** BlueField-4 进一步卸载网络、存储和安全路径，使 host CPU 更聚焦于调度、状态和推理编排。  
> 来源：StorageReview, *NVIDIA Launches Vera Rubin Architecture at CES 2026*, 2026-01

### 4. 机头 CPU 产品横向对比：Vera vs EPYC Turin vs Xeon 6

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


---

## 场景一：算子下发与状态驱动调度

### 1. 为什么 agentic 比传统 chat 更吃 CPU

传统 chat serving 更接近单条请求连续 decode。agentic inference 则会频繁经历：

1. prefill
2. decode
3. 外部阶段切换
4. 暂停
5. KV 保留或卸载
6. 恢复
7. 可能再分叉给多个 subagents

这一流程让 CPU 需要不断做：

- request state transition
- worker affinity 决策
- KV object 生命周期跟踪
- transfer/prefetch 触发
- 多 agent fan-out / fan-in 的回收与复用

### 2. 微观瓶颈：从"内存墙"到"调度墙"

2026 年 3 月的一篇深度工程实测揭示了一个被忽视的新范式：当模型通过 IQ4/FP4 等激进量化手段被压缩到可完全驻留 GPU L2 Cache 时，内存带宽瓶颈消失，但**算子下发（Dispatch）瓶颈凸显**。一个 135M 参数的量化模型单次前向传播发射 301 个 Kernel，每个 Launch 约 2.5 μs，总计 **750 μs** 的纯下发税，几乎等于单 Token 总时间（792 μs）。Kernel Fusion 将发射次数降至 181 次后，吞吐提升 **20%**（1255 → 1508 tok/s）。

这一因果链对机头 CPU 的选型有直接影响：
- **量化降低显存压力 → 模型更小 → Batch 内可容纳更多请求 → Kernel 发射频率更高 → CPU 调度负载更重。**
- 2026 年 1 月的 LongCat-Flash-Lite 论文同样观察到，在轻量模型+大有效 Batch Size 场景下，瓶颈从内存带宽转向 Kernel Launch Overhead，需通过 extensive kernel fusion 与 NVIDIA PDL（Programmatic Dependent Launch）缓解。
- FlashNorm（2026-04）的微观分析指出，单次 Kernel Launch 在 A100 上约 **10–15 μs**，加上中间张量分配（~5 μs）和 HBM 往返，每次融合可节省 **15–25 μs** 固定开销——这部分开销与模型规模无关，纯粹由 CPU 驱动栈决定。

### 3. CPU 竞争将微秒级开销放大为毫秒级集群停滞

2026 年 3 月论文《Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference》系统量化了该问题：
- vLLM 在 H100 上运行 Llama 3 8B 时，HTTP 服务占 **33%** 执行时间，调度+输入准备占 **29%**，GPU 实际计算仅 **38%**。
- 当 CPU 进程数超过可用核心数时，Kernel Launch 延迟从 μs 级恶化到 ms 级；在 NCCL 集合通信中，若某一 Rank 的 CPU 被抢占 1 ms，所有 GPU 忙等放大为集群级停滞。
- vLLM 的 `shm_broadcast.py` 广播队列在 5 req/s、100k Token 输入的 TP=4 场景下，dequeue 延迟从 12 ms 恶化到 **228 ms**（19 倍），是 GPU 单步解码时间（44 ms）的 5 倍以上。

### 4. 推理引擎层面的 CPU 优化——以 vLLM V1 为例

2025 年 1 月发布的 vLLM V1（于 2025 下半年成为默认引擎）是一次针对机头 CPU 开销的系统性重构：
- **Persistent Batch：** 缓存输入张量，每步仅应用增量 diffs，避免每步重建张量的 Python 开销。
- **Numpy 替代 Python Native：** 在调度器与数据准备路径上用 Numpy 操作替代原生 Python，显著降低 CPU 占用。
- **Zero-Overhead Prefix Caching：** 即使 Cache 命中率为 0%，吞吐损失也 < 1%，消除了 V0 中因前缀缓存数据结构导致的 CPU 瓶颈。
- **多模态输入预处理 Offload：** 将图像解码与预处理放到独立进程，避免阻塞 GPU Worker。
- **Piecewise CUDA Graphs：** 在保持动态调度能力的同时，尽可能捕获静态子图的 CUDA Graph，减少重复 Kernel Launch。

实测显示，V1 在文本模型上吞吐比 V0 提升最高 **1.7×**；在视觉语言模型（Qwen2-VL）上提升更为显著。vLLM 2026 Q1 Roadmap 进一步将 "Python overhead reduction"、"CPU KV cache production ready" 与 "disaggregated prefilling" 列为重点，表明社区已明确意识到机头 CPU 是下一阶段的优化主战场。

### 5. PD 分离：算子下发在解耦架构下的延伸

#### 5.1 PD 分离已从研究概念变为 2025–2026 年生产默认架构
2024 年的 DistServe 与 Splitwise 首次系统论证了 PD 分离的收益，而到 2025 年底，Hao AI Lab 的回顾性分析确认该架构已成为"几乎每个主要 LLM 服务栈的默认手册"。vLLM、SGLang、NVIDIA Dynamo、TensorRT-LLM 与 llm-d 均已原生支持 PD 分离。

对机头 CPU 而言，PD 分离意味着**调度器不再只管理单节点 GPU，而是需要跨节点协调 KV Cache 的序列化、传输与反序列化**。vLLM 的 `vllm/distributed/kv_transfer` 模块通过 Connector + LookupBuffer + Pipe 三层抽象实现跨实例 KV 搬运；TensorRT-LLM 则在预填充节点与解码节点之间通过网络层传输 KV Cache Block。

#### 5.2 KV Cache 传输开销高度依赖 CPU 侧网络栈与调度效率
- **同节点 NVLink：** DistServe 报告传输开销 < 总服务时间的 **0.1%**，可忽略。
- **跨节点网络：** Splitwise 计算表明，OPT-66B 在 512 Token 输入下产生约 **1.13 GB** KV Cache；若请求率达到 10 req/s，需约 **90 Gbps** 带宽才能避免瓶颈。
- **llm-d 0.5（2026-02）的 UCCL Backend：** 采用 host-resident software transport stack，由 CPU 管理传输逻辑而非完全依赖硬件卸载，在网络拥塞下尾延迟恶化仅 **7.1%**（对比 UCX 的 17.1%），验证了机头 CPU 在拥塞控制中的关键作用。

#### 5.3 Agentic 长交互进一步放大 CPU 调度压力
Agentic 工作负载通常表现为**短输入 + 极长输出**（多轮工具调用后的推理链），这意味着 decode 阶段持续时间远超 prefill。PD 分离后，decode 池需要长时间维持大量并发流的 KV Cache 状态，而 prefill 池则需快速处理频繁到达的新工具调用结果。机头 CPU 的调度器必须在两个池之间做动态负载均衡，并处理 KV Cache 的跨池预热、迁移与回收。vLLM 2026 Q1 Roadmap 明确将"CPU KV cache production ready"和"disaggregated prefilling & KV transfer support"列为核心目标，侧面反映了 CPU 侧调度复杂度正在快速上升。

### 6. 缓解路径汇总

- **Kernel Fusion：** 将残差相加、RMSNorm、RoPE、KV Cache 写入、通信操作（AllReduce + Residual Add + RMSNorm）融合为单 Kernel，减少 Launch 次数。
- **Persistent Kernel / Megakernel：** Event Tensor（2026-04）将动态控制流编码为 Tile 级依赖图，生成跨算子的持久化 Kernel，消除跨 Kernel 边界同步。
- **PDL（Programmatic Dependent Launch）：** NVIDIA 2026 年技术允许有依赖关系的 Kernel 提前触发、重叠执行间隙，提升 SM 利用率。
- **推理引擎重构：** vLLM V1 的 Persistent Batch、Zero-Copy DMA、前缀缓存优化与进程结构扁平化。
- **CPU 核心扩容与隔离：** 确保每 GPU 配有足够且隔离的 Host 核心，避免 OS 调度器介入关键路径；多模态预处理等任务应 offload 到独立进程。
- **PD 分离的网络栈优化：** 采用 UCCL 等 host-driven 传输后端以提升拥塞恢复能力；在跨节点场景确保 90 Gbps+ 可用带宽。

---

## 场景二：KV 卸载要求什么样的机头 CPU

### 1. CPU 内存已经是温热层，而不是单纯 spill 层

2025-09-18 的 NVIDIA Dynamo KV 文章把 KV offload 明确写成可落到 `CPU RAM`、`local SSD` 和 `network storage`。这不是简单的容量扩展表述，而是在承认推理系统正在走向分层状态存储。

2025-09-05 的 NVIDIA CPU-GPU memory sharing 文章进一步指出，Grace Hopper / Grace Blackwell 可以通过 **NVLink-C2C 900 GB/s** 的 coherent interconnect 共享统一内存地址空间。对机头 CPU 的含义非常直接：

- CPU DRAM 不再只是慢速备份，而是恢复路径中的 **首层温热 KV 池**
- 页表、pinning、IOMMU、NUMA 选路开始真实影响 token latency
- host 内存容量和持续带宽会比传统"GPU 服务器配个普通 CPU"更关键

<img src="assets/cpu-gpu-unified-memory.webp" alt="CPU-GPU 统一地址空间让 CPU 内存更适合作为 KV 温热层" width="760">

> **图：** CPU 与 GPU 通过高带宽一致性互连共享地址空间，使 host memory 更适合承担 KV 的 overflow / staging / warm tier。  
> 来源：NVIDIA, *Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing*, 2025-09-05

### 2. Agentic AI 把 KV 管理从"容量问题"升级成"生命周期问题"

2026-04-17 的 Dynamo agentic inference 文章把 `retention`、`routing`、`prefetch` 和 `WORM-like reuse` 放在同一套优化框架里。对应到机头 CPU，意味着：

- 不能只看 DRAM 容量，还要看 **恢复时能多快把 KV 拉回 GPU**
- 不能只看平均带宽，还要看 **pause-resume 的尾延迟**
- 不能只看 page cache，还要看 **pinned memory / hugepage / IOMMU map 更新** 的管理成本

以 Qwen3-32B FP16、64K 序列为例，KV Cache 占用约 **16 GB**。在多轮 Agentic 交互或长文档推理中，Cache 规模随 Batch Size 与序列长度线性增长，显存溢出成为常态。

### 3. 稀疏化 + 卸载是 2025 下半年以来的主攻方向

- **NOSA（2025-10，arXiv）：** 首个"原生为 KV Cache Offloading 设计"的可训练稀疏注意力机制。它显式约束 CPU-GPU KV 传输量，在 1B/3B/8B 模型上相比全注意力实现最高 **5.04×** 解码吞吐提升，相比 InfLLMv2 和 ShadowKV 分别提升 **1.92×** 和 **1.83×**。
- **ScoutAttention（2026-03，arXiv）：** 提出 Layer-Ahead CPU Pre-computation 算法，让 CPU 提前一层启动 Attention 计算，并通过异步周期性召回机制保持极低 CPU 负载。在保持精度损失 < 2.4% 的前提下，相比现有卸载方法实现 **2.1×** 加速。
- **CoMEM（2025，OpenReview）：** 针对 Agentic 长上下文，将历史压缩任务卸载到轻量级异步记忆模型，通过 k-step-off Pipeline 重叠记忆摘要与 Agent 执行，解码开销降低 **1.4×**。

### 4. CPU 在卸载系统中的角色转变

CPU 不再只是"数据搬运工"，而是 KV Cache 的**分级内存管理者**：
- **页级调度：** 类似操作系统 Swap，决定哪些 KV 页驻留 GPU、哪些降级到主机/CXL/SSD。
- **检索与预取：** 在稀疏注意力场景中，CPU 需动态估计 Token 重要性并预取相关 KV 块；若每步都触发检索，累积开销会抵消卸载收益。
- **协同计算：** ScoutAttention 等方案让 CPU 直接参与部分 Attention 计算，而非单纯传输数据，这要求机头 CPU 具备更强的向量/矩阵运算能力。

### 5. CXL 进一步把"主机内存层"从技术问题变成经济问题

Astera Labs 的 Leo CXL Smart Memory Controller（2025-11 实测数据）显示，在生产级 LLM 推理负载中：
- KV Cache 存储于 CXL 内存可减少 GPU 需求达 **87%**
- Prefill 阶段 GPU 利用率提升 **75%**
- 每查询 CPU 利用率降低 **40%**
- 系统可支持 **2 倍** 并发 LLM 实例

这些数字本身来自厂商建模，不应当被当成通用部署结果；但它们足以说明一件更稳定的事：  
**当 KV cache 的 warm tier 从"只能放主机 DRAM"扩展到"主机 DRAM + CXL memory"，机头 CPU 的价值就不只是容量兜底，而是整个推理经济模型的一部分。**

### 6. KV 卸载对 CPU 选型的具体启示

- **内存容量优先级上升：** agentic session、长前缀、共享模板和多 agent 扇出会一起抬高 host-side warm KV 容量需求
- **内存带宽要足够持续：** 不是一次性爆发，而是持续 serving 下的平稳数据回填
- **NUMA 拓扑要和 GPU/NIC 对齐：** 否则 offload/reload 的路径绕远，CPU 只会更忙
- **I/O 栈要低抖动：** KV tiering 接入本地 SSD 或网络存储后，host 成为真正的 state broker
- **对 co-located GPU 节点，重点是 `HBM -> coherent CPU memory` 的快速回填**
- **对容量优先型节点，重点是 `HBM -> host DRAM / CXL tier` 的分层经济性**
- **对多租户/长上下文场景，重点不只是"能不能卸"，而是"卸到哪一层最划算"**

---

## 场景三：MoE 把"机头 CPU"变成路由与通信编排器

### 1. 2025H2 之后的公开资料显示，MoE 已经不是单卡 kernel 问题

2025-12-18 的 NVIDIA Wide Expert Parallelism 文章讨论在 NVL72 机架级系统上扩展大规模 MoE。文章核心不是单个 expert 算得多快，而是：

- token 如何按 expert 路由
- expert 如何在大规模 GPU 间放置
- 通信如何被压平、隐藏或重叠
- 系统如何把 expert parallel 从"能跑"推进到"高吞吐可部署"

<img src="assets/nvidia-wide-ep-moe-2025.webp" alt="Wide expert parallelism 展示了 MoE 在大规模 GPU 系统上的跨设备组织方式" width="760">

> **图：** NVIDIA 关于 wide expert parallelism 的示意图，强调 MoE 推理的关键已经扩展到 expert 路由、并行放置和通信拓扑，而非仅是单个 GEMM。  
> 来源：NVIDIA, *Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack-Scale Systems*, 2025-12-18

### 2. 专家权重卸载是内存受限部署的必然选择

以 DeepSeek-R1（671B 总参 / 37B 激活参）为例，单节点 GPU 无法容纳全部专家权重。当专家权重被卸载到 CPU 内存时，每次 Token 路由命中冷专家都会触发同步 CPU→GPU 拷贝，成为解码阶段的决定性瓶颈。

Mixtral-8x7B 中每个 Token 可访问 47B 总参数，但仅 13B 参与计算，实现约 **3.6×** 的激活计算削减。这种"稀疏激活"特性使 MoE 在推理时具有天然效率优势，但也引入了独特的 host-side 复杂性。

### 3. 推测预取与异步流水线是 2026 年的主要突破

- **Speculating Experts（2026-03，arXiv）：** 利用当前层已计算的内部表示（归一化残差流 + 默认向量）推测下一层将激活的专家，实现权重预取与 GPU 计算的重叠。在 Qwen-30B-A3B 等模型上，相比按需加载实现 **14%** 的 TPOT 降低。若推测执行精度不足，还可叠加轻量级估计器提升命中率。
- **FluxMoE（2026-04，arXiv）：** 解耦"逻辑专家身份"与"物理驻留位置"，通过带宽均衡的存储层次（压缩 GPU 内存 + 主机 DRAM）动态流式化参数，摆脱对路由预测准确率的依赖。
- **中国科学技术大学专利（2025，CN）：** 提出异步并行推理方法，将 GPU 计算与 Expert Parallelism 固有的 All-to-All 通信解耦，允许 Token 数据通信与模型计算异步并行；同时策略性将热点专家常驻 GPU、冷点专家卸载 CPU。

### 4. CPU 在 MoE 中的三重负载

1. **权重搬运：** PCIe / C2C 带宽有限，CPU 负责将专家权重从主机内存拷贝到 GPU。
2. **路由协调：** All-to-All 集合通信的同步信号由 CPU 侧进程驱动；若任一 Rank 的 CPU 延迟，全网 GPU 等待。
3. **负载均衡与调度：** 动态专家剪枝、容量因子调整、冷热专家分级策略均需在 CPU 侧实时决策。

MoE 看起来是 GPU 场景，但对 host CPU 的要求反而更尖锐，因为 CPU 必须配合做：
- **批级路由组织：** token/expert 映射、微批分组、跨 worker 调度
- **拓扑感知编排：** 哪些 expert 留在本机、哪些跨节点，直接决定通信图
- **数据搬运与完成队列管理：** EP/all-to-all 的控制与生命周期管理并不全在 GPU 内部自发完成
- **与服务框架协同：** MoE 常和 disaggregated serving、prefill/decode split、KV reuse 一起出现，而不是单独出现

一个重要推断是：  
随着 agentic workload 引入更多多阶段推理、上下文复用和多代理并发，MoE 的 host-side 复杂度不会下降，反而会与 `route + transfer + KV` 三者叠加。因此，机头 CPU 更像 **orchestration processor**，而不是"配套 CPU"。


---

## 典型 Agentic Workload 反推：还有哪些 CPU 需求容易漏掉

> 这一节只看 **agentic LLM inference 对 CPU 的影响**。不把工具本身的执行开销算进来，例如浏览器渲染、ADB 点击、代码编译、搜索抓取等直接工具负载都排除在外。

### 1. OpenClaw / 手机 GUI Agent 类：不能低估多模态 prefill 和高频短回合切换

OpenClaw 官方仓库已经把产品形态定义为 `always-on` 的 personal AI assistant，并且覆盖 Android node、screen recording、camera、Canvas、device pairing 等持续在线入口。火山引擎在 2026-04-29 发布的 Mobile Use Agent 文章，则把这类产品进一步明确成基于云手机与豆包视觉模型的 `Mobile Use Agent`。

基于这两类产品形态，可以做出较强但仍属推断的判断：对 host CPU 来说，即使完全不算工具执行本身，仍然会新增三类推理侧压力：

- **多模态 prefill 压力：** 这类 GUI agent 往往需要把截图或界面状态送入模型，prefill 很可能比纯文本 agent 更重
- **高频短回合调度：** 这类交互常表现为短回合、频繁状态刷新，decode 未必长，但请求切换频繁
- **更细粒度的 KV 生命周期管理：** 单步推理可能较短，但状态连续性要求更高，host 更可能频繁做 session pinning、warm KV 保留和 resume

这意味着，"KV 卸载"和"算子下发"的判断是对的，但还少强调了一点：  
**GUI agent 会把 CPU 压力从长 decode 进一步推向高频 prefill + 高频状态切换。**

### 2. Claude Code 类：不能低估长驻上下文和多 subagent 并发

Anthropic 官方文档已明确说明，Claude Code `subagents` 各自拥有 `separate context window`，并且会因单独收集所需上下文而带来额外延迟。这件事对推理侧 CPU 的真正含义不是"工具多"，而是：

- **会话数暴增：** 一个主代理外加多个 subagents，等价于更多并行或准并行上下文
- **prefill 占比上升：** subagent 带着干净上下文启动，天然更容易形成 "短 burst + 重 prefill"
- **KV 复用更偏局部而不是全局：** 主代理和子代理不会天然共享同一整块上下文，host 需要更细地做 session-level placement 和复用决策

因此，对 Claude Code 这类 workload，还漏了一项明确表述：  
**host CPU 需要为"多上下文并存"而不是"单上下文超长"做优化。**  
这会抬高 admission control、per-session queue、KV placement 和尾延迟隔离的重要性。

### 3. Kimi Agent Swarm 类：不能低估 fan-out/fan-in 的瞬时并发宽度

Kimi 官方在 2026-04-11 的 Agent Swarm 文章里给出的产品形态非常直接：`up to 100 sub-agents working in parallel`。如果只看推理，不看工具执行本身，这种 workload 依然会给机头 CPU 带来一个此前还不够突出的要求：

- **瞬时 fan-out 调度能力：** 大量子代理会在短时间内同时进入 prefill 或 decode
- **返回汇总时的 fan-in 压力：** 上层代理需要消化来自多子代理的中间输出，再触发下一轮推理
- **批处理与公平性冲突：** 为了提吞吐，系统会想做 batch；但 swarm workload 又容易因为宽并发而拖高尾延迟

因此，Kimi Swarm 补充出的遗漏点是：  
**机头 CPU 不只是要"能跑高并发"，还要能处理极宽的 burst 并发和多层级调度。**

### 4. 这些真实 workload 共同补出的三项遗漏

如果把 OpenClaw、Claude Code、豆包手机/Mobile Use Agent、Kimi Swarm 放在一起看，还应补上三条更聚焦的 CPU 诉求：

- **遗漏一：prefill-first CPU 观念**  
  过去容易把 host 压力理解成 decode 陪跑，但真实 agentic workload 很多时候更像 `频繁 prefill + 短 decode + 快速 resume`
- **遗漏二：session multiplicity 而非单 session 长度**  
  Claude Code 和 Kimi Swarm 说明问题不只是上下文长，而是同时活跃的上下文条目太多
- **遗漏三：多模态 ingress 开销**  
  豆包手机/OpenClaw phone agent 说明，即便不算工具 CPU，截图/视觉输入进入推理链路本身也会放大 host 侧请求编排与内存压力

换句话说，若只从底层 serving 论文出发，很容易得出"CPU 主要是 KV tiering + transfer"这个结论；但把真实产品 workload 放进来后，应该把结论修正为：

**agentic LLM inference 对机头 CPU 的新增要求，除了 KV tiering 和 transfer，还包括 `高频 prefill 调度`、`多上下文并存管理`、`极宽 fan-out/fan-in` 和 `多模态 ingress 编排`。**

---

## 机头 CPU 的工程画像

结合上述证据，更合理的 agentic AI 节点 CPU 画像是：

- **更多可用核心，而不是只追单核峰值：** 因为要扛 router、scheduler、prefetch、completion、存储 client、K8s/runtime sidecar
- **更大的主机内存：** KV warm tier、agent state、prefill/decode 中间态都会吃容量
- **更好的内存带宽与 NUMA 可控性：** 影响 KV reload/prefetch 的可持续效率
- **更强的 I/O 与 DMA 协同能力：** RDMA / NVLink / NVMe-oF / NIC 队列管理都会经过 host 协调
- **更明确的分层部署思路：** co-located GPU 节点更看重一致性互连与主机内存带宽，容量型节点更看重 DRAM/CXL tier 的成本效率
- **更稳的尾延迟而不是只看均值：** agentic 场景对 pause-resume 和多阶段拼接极度敏感

如果要把这几条压缩成一句话：  
**agentic AI 时代的机头 CPU，核心任务不是"替 GPU 算"，而是"确保 GPU 不因路由、状态和数据搬运而空等"。**

---

## 采购与架构建议

### 1. 适合优先加码 CPU 的场景

- 多 agent / 长生命周期会话 / 多阶段推理
- 明确依赖 KV offload、resume、prefix reuse
- 使用 disaggregated serving 或者准备上 Kubernetes 化部署
- 准备在 MoE 上做大规模 expert parallel
- 存在高并发短回合切换或多模态输入（GUI Agent、Mobile Use Agent）
- 需要支持极宽 fan-out/fan-in（Agent Swarm、subagent 并发）

### 2. 可以相对弱化 CPU 的场景

- 单轮短对话、前缀复用弱、几乎不做多阶段推理切换
- 不做 KV offload，也不做 prefill/decode 解耦
- 模型规模和并发都小，系统仍接近单机直连推理
- 非 MoE 架构，且无跨 GPU 通信需求

### 3. 一个实用判断标准

如果你的服务已经出现下面任一迹象，就不该再把 host CPU 当成配角：

- GPU 利用率起伏很大，但显存和 FLOPS 并未打满
- 多阶段 resume 的尾延迟明显高于纯 decode
- KV 命中率高，但端到端时延改善不成比例
- MoE 扩容后吞吐没按 GPU 数线性增长
- K8s / runtime / transfer sidecar 一开就吃掉大量 host core
- 引入多模态输入后，prefill 延迟显著增加但 GPU 计算时间未变
- subagent 或 swarm 并发时，调度延迟出现阶跃式恶化

### 4. 机头 CPU 选型分层建议

| 节点类型 | 首选平台 | 关键理由 |
|---|---|---|
| **GPU 伴随型推理节点**（co-located） | NVIDIA Vera（或 Grace） | NVLink-C2C 1.8 TB/s + 统一内存地址空间，KV reload/prefetch 路径最短 |
| **通用推理网关 / 纯 CPU 编排节点** | AMD EPYC Turin | 192 核密度 + 成熟软件生态 + 最优 TCO；适合 router/scheduler/K8s 控制面 |
| **极致延迟敏感型边缘节点** | Intel Xeon 6 Granite Rapids | 5.0–5.7 GHz 单核频率，tokenization/序列化/API 解析尾延迟最低 |
| **容量优先型 KV 存储节点** | EPYC Turin + CXL 扩展 | 大容量 DRAM + CXL Memory Pooling，分层经济性最佳 |

---

## 证据表

| 时间 | 标题 | 类型 | 主题 | 主要发现 | 关联场景 |
|---|---|---|---|---|---|
| 2025-09-05 | Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing | NVIDIA 技术文章 | KV offload / unified memory | Grace Hopper / Grace Blackwell 通过 NVLink-C2C 900 GB/s coherent interconnect 共享内存地址空间 | KV 卸载 |
| 2025-09-18 | How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo | NVIDIA 技术文章 | KV offload hierarchy | KV cache 可卸到 CPU RAM、local SSD、network storage | KV 卸载 |
| 2025-10 | NOSA: Native and Offloadable Sparse Attention | 论文 (arXiv) | 稀疏注意力 + KV 卸载 | 显式约束 CPU-GPU KV 传输量，解码吞吐最高提升 5.04× | KV 卸载 |
| 2025-11 | Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective | 论文 (arXiv) | Agentic AI CPU 瓶颈量化 | 工具处理占 E2E 延迟 50%–90.6%；CPU 并行效率远低于 GPU；提出 COMB/MAS 调度优化 | Agentic 编排 |
| 2025-11 | How CXL Transforms RAG and KV Cache Performance | 厂商技术博客 | CXL / KV tiering | CXL 内存使 GPU 需求降低 87%，GPU 利用率提升 75%，每查询 CPU 利用率降低 40% | KV 卸载 |
| 2025-12-18 | Scaling Large MoE Models with Wide Expert Parallelism on NVL72 | NVIDIA 技术文章 | MoE / expert parallel | 重点落在 expert 路由、放置和跨 GPU 通信，而非单一算子 | MoE |
| 2026-01 | LongCat-Flash-Lite: Scaling Embeddings | 论文 (arXiv) | 轻量模型 Kernel Launch 瓶颈 | Extreme sparsity + large batch 使瓶颈从内存带宽转向 Kernel Launch Overhead | 算子下发 |
| 2026-02 | llm-d v0.5: UCCL Backend for PD Transfer | 工程博客 | PD 分离网络传输 | Host-resident software transport 由 CPU 管理拥塞控制，网络压力下尾延迟恶化仅 7.1%（vs UCX 17.1%） | PD 分离 |
| 2026-03 | NVIDIA Vera CPU Delivers High Performance, Bandwidth, and Efficiency for AI Factories | NVIDIA 技术文章 | 机头 CPU 架构 | Vera 88 核 Olympus，1.2 TB/s 内存带宽，独立部署商业模式确立 | 机头 CPU |
| 2026-03 | Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference | 论文 (arXiv) | CPU 竞争导致 Kernel Launch 延迟 | CPU Oversubscription 使 dequeue 延迟放大 19×；GPU 利用率可降至 70% 以下 | 算子下发 |
| 2026-03 | Speculating Experts Accelerates Inference for Mixture-of-Experts | 论文 (arXiv) | MoE 专家权重预取 | 利用内部表示推测未来专家，重叠 CPU-GPU 传输与计算，TPOT 降低 14% | MoE |
| 2026-03 | ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation | 论文 (arXiv) | 协同式 GPU-CPU Attention | CPU 提前一层预计算 Attention，异步召回，精度损失 < 2.4%，加速 2.1× | KV 卸载 |
| 2026-03 | Disaggregated Serving in TensorRT-LLM | 厂商技术博客 | PD 分离生产实践 | 分离上下文与生成阶段以消除干扰；KV Cache Block 跨节点传输成为关键路径 | PD 分离 |
| 2026-03 | Deploying Disaggregated LLM Inference on Kubernetes | NVIDIA 技术文章 | Disaggregated serving | ingress-router、prefill worker、decode worker 解耦部署 | 算子下发 |
| 2026-03 | Enhancing Distributed Inference with NIXL | NVIDIA 技术文章 | transfer dispatch | NIXL 统一 RDMA、NVLink、TCP sockets、NVMe-oF 等数据通路 | 算子下发 |
| 2026-03 | What Actually Bottlenecks LLM Inference on Modern GPUs | 工程博客 | Kernel Launch 税（量化小模型） | 301 次 Kernel Launch 中纯驱动开销占 750 μs；IQ4 量化+L2 Pinning 后内存墙消失，调度墙凸显 | 算子下发 |
| 2026-04 | Event Tensor: Dynamic Megakernels for LLM Serving | 论文 (MLSys 审稿) | 持久化 Kernel 消除 Launch 开销 | 将动态形状与数据依赖编码为 Tile 依赖图，生成 Persistent Kernel | 算子下发 |
| 2026-04 | FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving | 论文 (arXiv) | MoE 专家动态驻留 | 解耦逻辑专家身份与物理驻留，动态流式化参数，无需依赖路由预测 | MoE |
| 2026-04-17 | Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo | NVIDIA 技术文章 | Agentic inference / KV lifecycle | 85%-97% hit、97.2% aggregate hit、11.7x read/write ratio | KV 卸载 |
| 2026-04-11 | Kimi Introduces Agent Swarm | Kimi 官方博客 | Multi-agent fan-out | up to 100 sub-agents working in parallel | Workload 反推 |
| 2026-04-29 | 火山引擎 Mobile Use Agent 全新升级 | 火山引擎官方文章 | Mobile GUI agent | 基于云手机底座与豆包视觉模型构建 enterprise Android agent | Workload 反推 |
| 2026-03 | NVIDIA Vera CPU: Performance compared to AMD and Intel x86 chips | 产业评测 | 机头 CPU 横向对比 | Vera sandbox 性能 1.5× 于 x86；Redpanda cross-core 吞吐 +73%；64 核后持续扩展 | 机头 CPU |

---

## 图表附录

### 图 1：NVIDIA Vera Rubin 六芯片协同架构

<img src="assets/nvidia-vera-rubin-6chips.png" alt="NVIDIA Vera Rubin 六芯片架构" width="760">

> **图：** Vera Rubin 平台采用"极端协同设计"，将 Vera CPU、Rubin GPU、NVLink 6 Switch、ConnectX-9、BlueField-4 DPU 与 Spectrum-6 以太网交换机构建为统一系统。Vera CPU 作为编排与内存中枢，直接决定 Agentic 工作流的延迟与 GPU 利用率。来源：NVIDIA GTC 2026 / StorageReview。

### 图 2：BlueField-4 DPU — 卸载网络、存储与安全以释放 Vera CPU

<img src="assets/nvidia-bluefield4.png" alt="BlueField-4 DPU 架构" width="760">

> **图：** BlueField-4 集成 64 核心 CPU 与 ConnectX-9 SuperNIC，将网络、存储和安全处理从 Vera CPU 与 Rubin GPU 上卸载，使机头 CPU 能专注于 Agentic 编排与 Kernel 调度。来源：NVIDIA GTC 2026 / StorageReview。

### 图 3：Agentic Inference KV 读写比 — 状态保留成为核心

<img src="assets/nvidia-dynamo-agentic-kv-readwrite-2026.webp" alt="Agentic inference KV 读写曲线" width="760">

> **图：** NVIDIA Dynamo 给出的 agentic inference KV 读写曲线，read/write ratio 高达 11.7x，说明系统价值重心从"写新 KV"转向"保留、路由和预取旧状态"。来源：NVIDIA, 2026-04-17。

### 图 4：Disaggregated LLM Inference on Kubernetes — 机头职责扩展

<img src="assets/nvidia-k8s-disagg-serving-2026.webp" alt="K8s 解耦式推理拓扑" width="760">

> **图：** NVIDIA 在 Kubernetes 上展示的解耦式推理拓扑，host 侧职责从"单机发命令"扩展为 router + stage scheduling + transfer orchestration。来源：NVIDIA, 2026-03-23。

### 图 5：CPU-GPU 统一内存地址空间 — KV 温热层基础

<img src="assets/cpu-gpu-unified-memory.webp" alt="CPU-GPU 统一地址空间" width="760">

> **图：** CPU 与 GPU 通过高带宽一致性互连共享地址空间，使 host memory 更适合承担 KV 的 overflow / staging / warm tier。来源：NVIDIA, 2025-09-05。

### 图 6：Wide Expert Parallelism — MoE 跨设备组织方式

<img src="assets/nvidia-wide-ep-moe-2025.webp" alt="Wide expert parallelism" width="760">

> **图：** NVIDIA wide expert parallelism 示意图，强调 MoE 推理的关键已经扩展到 expert 路由、并行放置和通信拓扑。来源：NVIDIA, 2025-12-18。

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
10. "LongCat-Flash-Lite: Scaling Embeddings Outperforms Scaling Experts in Language Models." arXiv:2601.21204 (Jan 2026). https://arxiv.org/abs/2601.21204
11. "FlashNorm: Fast Normalization for Transformers." arXiv:2407.09577v4 (Apr 2026). https://arxiv.org/abs/2407.09577

### 厂商官方资料
12. "NVIDIA Vera CPU Delivers High Performance, Bandwidth, and Efficiency for AI Factories." NVIDIA Developer Blog (Mar 2026). https://developer.nvidia.com/blog/nvidia-vera-cpu-delivers-high-performance-bandwidth-and-efficiency-for-ai-factories/
13. "NVIDIA Vera CPU Enters Full Production." Data Center Dynamics (Mar 2026). https://www.datacenterdynamics.com/en/news/nvidia-vera-cpu-enters-full-production-pitched-at-agentic-ai-workloads/
14. "NVIDIA Launches Vera Rubin Architecture at CES 2026." StorageReview (Jan 2026). https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack
15. "Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing." NVIDIA Developer Blog (Sep 2025). https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/
16. "How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo." NVIDIA Developer Blog (Sep 2025). https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/
17. "Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo." NVIDIA Developer Blog (Apr 2026). https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/
18. "Deploying Disaggregated LLM Inference Workloads on Kubernetes." NVIDIA Developer Blog (Mar 2026). https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/
19. "Enhancing Distributed Inference Performance with the NVIDIA Inference Transfer Library." NVIDIA Developer Blog (Mar 2026). https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/
20. "Scaling Large MoE Models with Wide Expert Parallelism on NVL72." NVIDIA Developer Blog (Dec 2025). https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/
21. "Disaggregated Serving in TensorRT-LLM." NVIDIA GitHub / TensorRT-LLM Blog (Mar 2026). https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html

### 产业与市场分析
22. "How Agentic AI Is Reshaping the CPU:GPU Ratio." TrendForce Insights (Apr 2026). https://insights.trendforce.com/p/agentic-ai-cpu-gpu
23. "The Forgotten Chip: CPUs the New Bottleneck of the Agentic AI Era." Uncover Alpha (Feb 2026). https://www.uncoveralpha.com/p/the-forgotten-chip-cpus-the-new-bottleneck
24. "Secret Agent CPU." The Diligence Stack / Ben Bajarin (Mar 2026). https://thediligencestack.com/p/secret-agent-cpu
25. "NVIDIA Vera CPU: Performance compared to AMD and Intel x86 chips." Digit.in (Mar 2026). https://www.digit.in/features/general/nvidia-vera-cpu-performance-compared-to-amd-and-intel-x86-chips.html
26. "NVIDIA’s Vera CPU in Detail." ServeTheHome (Mar 2026). https://www.servethehome.com/nvidias-vera-cpu-in-detail-high-perf-chip-takes-aim-at-broader-ai-server-market/2/

### 开源项目与工程实践
27. "vLLM V1 Alpha Release." vLLM Blog (Jan 2025). https://openlm.ai/vllm-v1/
28. "vLLM Roadmap Q1 2026." GitHub Issue #32455 (Jan 2026). https://github.com/vllm-project/vllm/issues/32455
29. "Disaggregated Prefilling (experimental)." vLLM Documentation. https://docs.vllm.ai/en/latest/features/disagg_prefill/
30. "llm-d 0.5: Sustaining Performance at Scale." llm-d Blog (Feb 2026). https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale
31. "Prefill-decode disaggregation." BentoML LLM Inference Handbook. https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation
32. "Disaggregated Inference: 18 Months Later." Hao AI Lab (Nov 2025). https://haoailab.com/blogs/distserve-retro

### 真实产品 Workload 参考
33. "Kimi Introduces Agent Swarm: Let 100 AI Agents Work for You." Kimi Blog (Apr 2026). https://www.kimi.com/blog/agent-swarm.html
34. "不止对话，更能执行！火山引擎 Mobile Use Agent 全新升级." 火山引擎 (Apr 2026). https://developer.volcengine.com/articles/7628489608359395369
35. "openclaw/openclaw README." GitHub (release window 2026-04-14). https://github.com/openclaw/openclaw
36. "Subagents." Anthropic Official Documentation (current, no exposed publish date; supplementary only). https://docs.anthropic.com/en/docs/claude-code/sub-agents

### 技术方案与专利
37. "How CXL Transforms RAG and KV Cache Performance." Astera Labs (Nov 2025). https://www.asteralabs.com/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance/
38. "KV cache offloading — CPU RAM vs. storage." NetApp Community (Nov 2025). https://community.netapp.com/t5/Tech-ONTAP-Blogs/KV-cache-offloading-CPU-RAM-vs-storage/ba-p/464463
39. "MoE inference cost cuts: 30+ patents analyzed." PatSnap (Apr 2026). https://www.patsnap.com/resources/blog/articles/moe-inference-cost-cuts-30-patents-analyzed/
40. "AI Inference Optimization Techniques (2025-2026)." Zylos Research (Jan 2026). https://zylos.ai/research/2026-01-11-ai-inference-optimization
41. "异步并行推理方法专利 (CN)." 中国科学技术大学 (2025). 见 PatSnap MoE 专利分析引用。

---

> **免责声明：** 本洞察基于 2025-07-01 至 2026-04-24 期间公开发表的技术论文、厂商公告、开源项目演进与产业分析整理而成。涉及尚未量产的产品（如 NVIDIA Vera Rubin 大规模部署、Intel Xeon 7 等）时间表存在延期风险；性能数据来源于论文、厂商受控测试或第三方早期 benchmark（如 Redpanda），实际部署收益取决于具体工作负载与系统配置。Anthropic Subagents 文档因未暴露公开日期，仅作为补充参考而非日期边界内主证据。
