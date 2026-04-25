# Agentic AI 推理中机头 CPU 的角色、瓶颈与系统演化综述

> Updated: 2026-04-24  
> Date boundary: 主要覆盖 `2025-07-01` 及之后公开资料。  
> Scope: 本文聚焦 GPU 推理节点上的 host CPU / control-plane CPU（下文统称“机头 CPU”）在 agentic AI 推理中的系统作用，不讨论训练场景；对工具执行本身的 CPU 消耗仅在必要时作为背景，不作为主分析对象。

**关键词：** Agentic AI；LLM inference；host CPU；operator dispatch；KV cache offloading；Mixture of Experts；prefill-decode disaggregation

## 摘要

随着 agentic AI 从单轮问答式推理转向多阶段、可暂停、可恢复、可分叉的长生命周期执行，推理系统的关键瓶颈正在从纯 GPU 计算逐步外溢到 host 侧编排链路。基于 现有材料中的报告、图表和引用材料，本文综述 2025 年下半年以来有关机头 CPU 的主要公开证据，并将现有研究归纳为四条相互耦合的技术主线：算子下发与状态驱动调度、KV cache 卸载与生命周期管理、MoE 推理下的 expert orchestration，以及真实 agentic workload 对传统 serving 假设的修正。现有证据表明，机头 CPU 的核心功能已从传统 host 演化为 inference orchestration layer：其职责不再局限于 kernel launch，而扩展到请求接入、prefill/decode 切分、KV 保留与预取、跨节点传输、专家放置及多代理并发控制等多个方面。与此同时，Vera、Rubin、BlueField-4、CXL 等平台与互连信号说明，硬件路线图也在围绕“CPU 作为 AI factory 控制平面”这一假设收敛。本文进一步指出，现有研究虽已较好揭示了 launch overhead、KV warm tier、PD 分离和 MoE expert residency 等问题，但在统一 benchmark、真实产品 workload 与底层机制之间的映射，以及平台方案的长期可迁移性方面仍存在明显空白。

## 1. 引言

近两年，大模型推理系统的优化重点经历了显著迁移。早期工作主要关注 GPU 侧的算力利用率、注意力算子实现和显存容量边界；而在 agentic AI 兴起之后，系统行为从“单次请求、连续 decode”转向“多阶段推理、状态保留、外部中断、上下文复用与多代理并发”的复合执行模式。该变化直接改变了系统瓶颈的空间分布。

现有材料中的多份材料虽然来源异构，既包含技术报告，也包含平台资料、厂商博客和产品工作负载分析，但它们共同支持一个核心判断：  
**agentic AI 推理将机头 CPU 从外围控制器推到了推理关键路径。**

需要强调的是，本文不采用“CPU 是所有 agentic 系统瓶颈”的宽泛表述，而将分析严格限定在 **agentic LLM inference 对 host CPU 的影响**。在这一边界下，机头 CPU 的重要性并不来自其替代 GPU 进行大规模矩阵计算，而来自其对系统编排链路的决定性影响：一旦调度、状态管理、传输协调或专家放置失配，GPU 计算资源即可能处于等待状态，进而使端到端推理效率显著下降。

<img src="assets/nvidia-dynamo-agentic-kv-readwrite-2026.webp" alt="Agentic inference 中 KV 读取显著高于写入" width="760">

**图1** Agentic inference 的 KV 读写关系。该图展示了累计读取明显快于累计写入的趋势，说明 agentic workload 的核心压力正从“持续写入新状态”转向“保留、路由、预取与恢复既有状态”。来源：NVIDIA, 2026-04-17 [9]。

## 2. 问题定义与分析边界

### 2.1 机头 CPU 的定义

本文所称“机头 CPU”是指 GPU 推理节点中的 host CPU 或 control-plane CPU。与传统服务器中的通用主机 CPU 相比，其在 agentic AI 推理中的角色更接近：

- request ingress / routing coordinator
- stage scheduler
- KV lifecycle manager
- transport orchestrator
- multi-agent execution controller

因此，本文讨论的不是“CPU 通用性能”，而是 **CPU 在推理系统控制链路中的系统作用**。

### 2.2 相关概念

- **算子下发（operator dispatch / kernel launch）**：CPU host 进程经由 runtime 或 driver 将 GPU kernel、图执行或相关控制事件提交至设备侧的过程。
- **KV 卸载（KV cache offloading）**：将推理过程中产生的 key-value cache 从 GPU HBM 分层迁移到 host DRAM、CXL memory、SSD 或网络存储。
- **PD 分离（prefill-decode disaggregation）**：将 prefill 与 decode 两类负载在物理上部署到不同资源池，以分别优化计算密集和带宽密集阶段。
- **MoE expert orchestration**：在 Mixture-of-Experts 推理中，对 expert 路由、驻留、预取、迁移和通信拓扑所做的 host-side 编排。

### 2.3 文献范围说明

本文以 `cited-materials` 中的公开论文、博客、产品文档和产业资料为基础，其中：

- 研究型主证据主要来自 arXiv / OpenReview 论文与技术文档 [1]-[12]。
- 平台与系统信号主要来自 NVIDIA、StorageReview、TrendForce、Astera Labs 以及相关产业材料 [13]-[23][28]。
- 真实产品工作负载推断主要来自 Anthropic、OpenClaw、Kimi、火山引擎等公开信息 [24]-[27]。

其中个别网页文档未暴露精确发布日期，本文仅将其作为补充性说明，而非时间边界内的主证据。

## 3. 资料来源与整理方法

本文的资料基础来自 现有材料中的既有综述、补充报告、图表与引用材料。为降低时效漂移，正文主分析尽量仅采用 `2025-07-01` 之后的公开资料，并按如下原则整理：

- **研究型主证据优先：** 优先采用 arXiv、OpenReview 与官方技术文档，用于支撑机制性结论。
- **厂商与平台资料用于系统信号：** 官方博客、平台发布和架构综述主要用于判断产品方向、部署模式与硬件路线图，不直接替代机制性测量。
- **真实产品材料用于 workload 反推：** Anthropic、OpenClaw、Kimi、火山引擎等来源主要用于说明产品形态，不把其产品描述直接当作底层机制实证。
- **边界不明来源降格处理：** 未暴露明确发布日期的文档仅作为补充说明，不作为时间边界内的主证据。

在此基础上，本文将资料分为四组：  
(1) 算子下发与调度；(2) KV 生命周期与分层内存；(3) MoE expert orchestration；(4) 真实 workload 与平台信号。  
文中的核心观点仅在至少两类来源能够相互印证时才上升为综述性判断。

## 4. 文献脉络：从 host bottleneck 到 orchestration bottleneck

现有材料可以被组织为一条逐步收敛的文献脉络。

第一阶段，研究开始重新审视 host 侧瓶颈在多 GPU 推理中的作用。CPU oversubscription、kernel launch latency、HTTP/service stack 占比、广播队列与同步点放大效应等现象，说明 GPU 利用率并不单纯由 GPU 算子决定，而与 host 侧排队和调度强相关 [2][3]。

第二阶段，KV cache 卸载研究将 host CPU 从“慢速兜底层”重新定义为“分级内存管理者”。NOSA、ScoutAttention、NVIDIA Dynamo 和 CPU-GPU coherent memory 相关材料共同表明，KV 问题已从容量扩展问题演化为生命周期与放置问题 [6][7][9][10]。

第三阶段，MoE 推理工作将 host 侧压力从“发起请求”进一步扩展到“参与专家驻留与通信拓扑决策”。Speculating Experts、FluxMoE 以及相关专利说明，在专家稀疏激活带来 GPU 计算节约的同时，host-side 的搬运、路由和同步问题被显著放大 [11][12]。

第四阶段，真实 agentic workload 与平台信号开始修正学术工作中过于简化的假设。Claude Code subagents、Kimi Agent Swarm、OpenClaw、Mobile Use Agent 等产品形态表明，真实系统并非“单上下文、长 decode”的单一模式，而是高频 prefill、多上下文并存、极宽 fan-out/fan-in 和多模态 ingress 的复合模式 [24]-[27]。与此同时，Vera、Rubin、BlueField-4、CXL 等平台资料表明，硬件路线图也开始围绕 host 侧编排能力重构 [13]-[16][22][23][28]。

换言之，当前研究关注点已经从“CPU 是否是瓶颈”演化为更具体的问题：  
**CPU 究竟在推理系统的哪些编排环节成为瓶颈，以及这些环节如何被软件栈和硬件平台共同重塑。**

## 5. 算子下发与状态驱动调度

### 5.1 从 kernel launch overhead 到系统调度墙

关于算子下发，现有材料中的材料提供了较一致的结论：在量化小模型、高并发和细粒度执行场景下，固定的 host-side launch 成本会迅速上升为主导瓶颈 [2][3][4]。结合解耦部署拓扑可以更清楚地看到，host 侧不仅要发起 kernel，还要持续承担 route、stage scheduling 和 transfer orchestration（见图2）。该现象的关键不在于单次 launch 本身的微秒级代价，而在于其与以下因素的耦合：

- 请求粒度更小
- batch 变化更频繁
- 状态切换更多
- CPU 调度链路更长
- 多 GPU 同步点更密集

因此，所谓“调度墙”并不应被狭义理解为 CUDA launch 本身，而应理解为 **GPU 开始有效计算之前的一整条 host 参与控制路径**。凡是这条路径中需要 CPU 排队、分配、广播、同步或提交的步骤，都可能在 agentic inference 中成为关键瓶颈。

<img src="assets/nvidia-k8s-disagg-serving-2026.webp" alt="解耦式 LLM inference 拓扑" width="760">

**图2** 基于 Kubernetes 的解耦式 LLM inference 拓扑。图中 ingress router、prefill worker 与 decode worker 的拆分说明，host 侧职责已从单机发起请求扩展为跨阶段路由、状态搬运与资源协调。来源：NVIDIA, 2026-03-23 [22]。

### 5.2 CPU oversubscription 的系统后果

《Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference》指出，CPU oversubscription 会使本应处于微秒级的 launch 或 dequeue 延迟放大至毫秒级，并在多 GPU 通信场景中产生级联等待 [2]。其重要含义在于：  
GPU 集群中的慢节点不一定来自设备侧，也可能来自某一 rank 上被抢占的 host 线程。

这意味着，机头 CPU 的问题不是简单地“多给几个核心”即可解决，而是必须从以下层面同时处理：

- 核心数与线程隔离
- runtime/service stack 的 CPU 占用
- NCCL/collective 路径上的 host 抖动
- 队列与广播机制的尾延迟控制

小结：算子下发问题的实质并非单个 kernel launch 的微观税费，而是 agentic inference 将 host 侧控制链路拉长后，调度、排队与同步成本被系统性放大。

### 5.3 Agentic 场景为何更易放大调度墙

与单轮 chat 相比，agentic inference 更常出现：

1. prefill
2. decode
3. 状态中断
4. 状态恢复
5. 上下文分叉
6. 子代理并发

这一流程使得 host 侧需要更频繁地处理状态迁移与执行阶段切换。因此，agentic workload 会把 launch overhead 从单次前向传播问题，放大成 **状态驱动调度问题**。这也是为什么 persistent batch、zero-overhead prefix caching、CUDA graphs、persistent kernels 等优化，会在 agentic 语境下变得尤为重要 [3][4][5]。

## 6. KV 卸载：从容量扩展到生命周期管理

### 6.1 KV 的问题定义已经改变

在传统推理系统中，KV cache 常被视作长上下文支持所需的容量对象；而在 agentic AI 中，它更像一个长期状态对象。Dynamo 的 agentic inference 材料给出的高命中率和 `11.7x` 的 read/write ratio 清楚说明，真实压力已从“写入新 KV”转向“保留、路由、预取与恢复旧 KV” [9]。

因此，KV 卸载的核心不再只是“把放不下的 KV 挪出去”，而是：

- 哪些 KV 应保留在最热层
- 哪些 KV 应放入 warm tier
- 如何在恢复路径中以低尾延迟取回
- 如何在多 worker / 多 agent 间保持状态可复用

### 6.2 Host CPU 从搬运者变为 warm-tier manager

在这一语境下，机头 CPU 的角色已经从“数据搬运工”转为：

- tier placement manager
- prefetch coordinator
- resume latency controller
- page-level policy executor

NOSA 通过约束 CPU-GPU KV 传输量来提高解码吞吐 [6]；ScoutAttention 则进一步让 CPU 参与 layer-ahead 的 attention 相关预计算 [7]。这些工作共同说明：  
**现代 KV 卸载系统中的 CPU，已不再是被动等待请求的存储代理，而是主动参与生命周期控制的 warm-tier 编排器。**

### 6.3 CXL 与 coherent memory 的系统意义

Grace Hopper / Grace Blackwell 的 coherent CPU-GPU memory 与 Astera Labs 的 CXL 材料，分别从两个方向拓展了 warm tier 的设计空间 [10][15]。其中 coherent memory 的意义在于降低 host memory 参与恢复路径时的摩擦，如图3所示：

- coherent memory 更适合低摩擦、低延迟的近端 warm tier
- CXL memory 更适合容量优先、经济性驱动的扩展层

因此，KV warm tier 的架构已经不应被简化为“HBM vs DRAM”，而应被视作一个多层次的放置问题；面向工程部署的存储分层讨论也已从社区和厂商侧得到补充 [15][29]：

- HBM
- coherent CPU memory
- host DRAM
- CXL memory
- local / remote storage

该分层问题直接决定了机头 CPU 的选型目标：某些节点强调一致性互连和高每核带宽，另一些节点则强调大容量主机内存与分层成本效率。

<img src="assets/cpu-gpu-unified-memory.webp" alt="CPU-GPU 统一地址空间与高带宽一致性互连" width="760">

**图3** CPU-GPU 统一地址空间示意。高带宽一致性互连使 host memory 更适合承担 KV 的 overflow、staging 与 warm-tier 职责，从而将 CPU 从“远端内存宿主”推进为恢复路径中的有效参与者。来源：NVIDIA, 2025-09-05 [10]。

小结：KV 卸载研究已经从“把数据搬出去”转向“如何管理长期状态对象”，而机头 CPU 正是这一生命周期管理的第一执行者。

## 7. MoE：稀疏计算优势与 host-side orchestration 压力

### 7.1 MoE 的效率收益并不自动转化为系统收益

MoE 通常被理解为“以更少的激活计算获得更大模型能力”，但 现有材料中的 MoE 材料显示，这一说法忽略了系统代价：当专家总量超出单节点 GPU 驻留能力后，权重搬运、路由预测、同步通信和拓扑放置都会显著增加 host-side 压力 [11][12]。在大规模 GPU 系统中，这一点会进一步表现为 expert parallel 拓扑与通信图的复杂化，如图4所示。

因此，MoE 推理系统的关键问题不只是“专家是否稀疏”，而是：

- 专家驻留在哪里
- 权重何时搬运
- 路由如何预测
- all-to-all 通信如何与计算重叠

### 7.2 预取与驻留解耦是当前主流方向

Speculating Experts 通过提前预测未来激活专家，将 CPU→GPU 权重搬运从同步阻塞路径挪到异步预取路径 [11]；FluxMoE 则通过 decoupled residency，使逻辑 expert 身份与物理驻留位置解耦 [12]。与此同时，专利与产业资料也开始把 expert residency、异步通信和成本优化视为同一问题域 [30]。这几类工作共同指向同一个结论：

**MoE 推理真正稀缺的资源，不只是 GPU 算力，还包括 host 侧能否提前、稳定、低抖动地完成权重与通信编排。**

### 7.3 机头 CPU 在 MoE 中承担的三类职责

从现有汇总材料看，MoE 至少将以下三类职责持续压到 host 侧：

- weight movement
- routing coordination
- topology-aware load balancing

也就是说，MoE 并非只把计算问题稀疏化，而是把系统问题显性化。对 agentic workload 而言，这一压力还会进一步与 KV 生命周期和多代理并发叠加。

小结：MoE 的系统代价主要不是“多一些通信”，而是 host 侧必须持续承担专家路由、驻留与同步的编排责任。

<img src="assets/nvidia-wide-ep-moe-2025.webp" alt="Wide expert parallelism 拓扑示意" width="760">

**图4** Wide expert parallelism 在机架级系统中的组织方式。该图说明 MoE 推理的关键瓶颈已经扩展到 expert 路由、放置与跨设备通信，因此 host 侧的协调与同步能力直接影响系统吞吐。来源：NVIDIA, 2025-12-18 [28]。

## 8. 真实工作负载对传统 serving 假设的修正

### 8.1 传统假设：单上下文、长 decode、稳定批次

许多 serving 优化隐含采用了一个较理想化的负载模型：请求主要表现为较长 decode，批次变化平滑，会话数可控，输入形态以文本为主。现有材料中关于 OpenClaw、Claude Code、Kimi Swarm、Mobile Use Agent 的材料则表明，该假设在真实 agent 产品中已明显不足 [24]-[27]。

### 8.2 Prefill-first、session multiplicity、fan-out/fan-in、multimodal ingress

综合这些产品形态，可以将 agentic AI 对机头 CPU 的新增要求概括为四点：

- **prefill-first**：高频短回合和多模态重新入模使 prefill 压力被重新放大。
- **session multiplicity**：瓶颈不再只是单条上下文过长，而是同时活跃的上下文过多。
- **fan-out/fan-in width**：大量子代理并发引入极宽瞬时并发，而非仅仅提高平均 QPS。
- **multimodal ingress**：视觉输入重入推理链路后，host 侧排队、状态切换与内存压力都会同步上升。

这四点共同说明：  
**真实 agentic workload 会将机头 CPU 的问题从“服务单条请求”转变为“管理异构、并发、分叉且可恢复的状态集合”。**

小结：真实 agent 产品形态使 host 侧问题从平均吞吐问题演化为高频 prefill、多会话并存与宽并发 burst 的复合调度问题。

## 9. 平台演化：硬件路线图为何开始围绕 CPU 控制平面收敛

Vera、Rubin、BlueField-4、TrendForce 等材料的共同意义，不在于证明某一厂商方案绝对优于另一方案，而在于它们传递了一个平台级共识 [13]-[16][17][20][21][28]：机头 CPU 已被当成 AI factory 控制平面的承载者来设计，相关平台组织方式如图5所示。

- 主机内存带宽正被重新提升到核心指标位置
- CPU-GPU 一致性互连正成为 warm-tier 设计的关键
- DPU / SuperNIC 正在将网络、存储与安全职责旁路出去
- CPU:GPU 配比正在向更高 CPU 比重移动

这些信号说明，系统平台已经开始默认：  
**agentic inference 的性能上限不只取决于 GPU FLOPS，还取决于 host-side orchestration budget。**

小结：平台演化并不是对现有研究的附会，而是在硬件层面确认了机头 CPU 已成为推理控制平面的关键承载者。

<img src="assets/nvidia-vera-rubin-6chips.png" alt="Vera Rubin 六芯片平台协同架构" width="760">

**图5** Vera、Rubin、NVLink Switch、BlueField-4 与网络交换的协同平台。该图表明平台设计已经将 CPU、GPU、DPU 与互连作为一体化系统来组织，机头 CPU 的位置因而从传统 host 前移为系统控制平面。来源：StorageReview, 2026 [14]。

## 10. 讨论：现有研究的共识与不足

### 10.1 当前较稳健的共识

基于 现有材料，至少可形成以下较稳健的共识：

1. 机头 CPU 已进入推理关键路径。  
2. CPU 瓶颈的本质在于 orchestration chain，而非纯算力不足。  
3. KV 卸载的关键问题已从容量扩展转向生命周期与分层经济性。  
4. MoE 会持续抬高 host-side orchestration 的系统价值。  
5. 节点选型应按角色分层，而非简单按 CPU 品牌或代际分层。

### 10.2 仍然缺失的部分

尽管证据已足够丰富，当前研究仍存在三个突出缺口：

- **统一 benchmark 缺失**：尚无被广泛接受的 `agentic inference host benchmark`。  
- **产品 workload 与底层机制间存在断层**：真实产品很少公开足够细粒度的系统测量。  
- **平台信号强于长期验证**：Vera / Rubin / BlueField-4 等方案的长期通用性仍需更多独立部署证据。

## 11. 结论

本文基于 现有材料中的现有内容，对 2025 年下半年以来 agentic AI 推理中的机头 CPU 问题进行了归纳。总体而言，现有材料所揭示的并不是“CPU 回来了”这样一个抽象判断，而是更具体的系统事实：

**agentic AI 推理正在把原本隐藏在 host 侧的控制、状态、传输与编排链路，重新暴露为系统效率的决定因素。**

在这一新格局下，GPU 仍然是最主要的数值计算资源，但机头 CPU 已成为决定推理系统能否稳定、高效、低尾延迟运行的第一层编排器。未来关于 agentic inference 的研究，不应继续将 host CPU 视为背景常量，而应将其作为系统优化中的一等公民来建模、测量与设计。

## 参考文献

[1] RAJ R, et al. Towards understanding, analyzing, and optimizing agentic AI execution: a CPU-centric perspective[EB/OL]. arXiv:2511.00739, 2025[2026-04-24]. https://arxiv.org/abs/2511.00739.

[2] Characterizing CPU-induced slowdowns in multi-GPU LLM inference[EB/OL]. arXiv:2603.22774, 2026[2026-04-24]. https://arxiv.org/abs/2603.22774.

[3] What actually bottlenecks LLM inference on modern GPUs[EB/OL]. AI.rs, 2026[2026-04-24]. https://ai.rs/ai-developer/memory-wall-disappears-llm-inference-bottlenecks.

[4] Event Tensor: dynamic megakernels for LLM serving[EB/OL]. arXiv:2604.13327, 2026[2026-04-24]. https://arxiv.org/abs/2604.13327.

[5] vLLM Project. vLLM V1 alpha release and subsequent public roadmap materials[EB/OL]. 2025-2026.

[6] HUANG Y, et al. NOSA: native and offloadable sparse attention[EB/OL]. arXiv:2510.13602, 2025[2026-04-24]. https://arxiv.org/abs/2510.13602.

[7] ZHANG Q, et al. ScoutAttention: efficient KV cache offloading via layer-ahead CPU pre-computation[EB/OL]. arXiv:2603.27138, 2026[2026-04-24]. https://arxiv.org/abs/2603.27138.

[8] CoMEM: a decoupled agent framework with asynchronous memory compression[EB/OL]. OpenReview, 2025[2026-04-24]. https://openreview.net/pdf?id=c7c6541f58ddaf647289d2523a9587312294301a.

[9] NVIDIA. Full-stack optimizations for agentic inference with NVIDIA Dynamo[EB/OL]. 2026[2026-04-24]. https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/.

[10] NVIDIA. Accelerate large-scale LLM inference and KV cache offload with CPU-GPU memory sharing[EB/OL]. 2025[2026-04-24]. https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/.

[11] Speculating experts accelerates inference for mixture-of-experts[EB/OL]. arXiv:2603.19289, 2026[2026-04-24]. https://arxiv.org/abs/2603.19289.

[12] FluxMoE: decoupling expert residency for high-performance MoE serving[EB/OL]. arXiv:2604.02715, 2026[2026-04-24]. https://arxiv.org/abs/2604.02715.

[13] NVIDIA. NVIDIA Vera CPU delivers high performance, bandwidth, and efficiency for AI factories[EB/OL]. 2026[2026-04-24]. https://developer.nvidia.com/blog/nvidia-vera-cpu-delivers-high-performance-bandwidth-and-efficiency-for-ai-factories/.

[14] StorageReview. NVIDIA launches Vera Rubin architecture at CES 2026[EB/OL]. 2026[2026-04-24]. https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack.

[15] Astera Labs. How CXL transforms RAG and KV cache performance[EB/OL]. 2025[2026-04-24]. https://www.asteralabs.com/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance/.

[16] TrendForce. How agentic AI is reshaping the CPU:GPU ratio[EB/OL]. 2026[2026-04-24]. https://insights.trendforce.com/p/agentic-ai-cpu-gpu.

[17] Data Center Dynamics. NVIDIA Vera CPU enters full production, pitched at agentic AI workloads[EB/OL]. 2026[2026-04-24]. https://www.datacenterdynamics.com/en/news/nvidia-vera-cpu-enters-full-production-pitched-at-agentic-ai-workloads/.

[18] The Diligence Stack. Secret agent CPU[EB/OL]. 2026[2026-04-24]. https://thediligencestack.com/p/secret-agent-cpu.

[19] rmmod. In the age of agentic, the CPU is the new bottleneck[EB/OL]. 2026[2026-04-24]. https://rmmod.com/posts/agent/agentic-cpu-bottleneck/.

[20] Uncover Alpha. The forgotten chip: CPUs the new bottleneck of the agentic AI era[EB/OL]. 2026[2026-04-24]. https://www.uncoveralpha.com/p/the-forgotten-chip-cpus-the-new-bottleneck.

[21] Zylos Research. AI inference optimization techniques (2025-2026)[EB/OL]. 2026[2026-04-24]. https://zylos.ai/research/2026-01-11-ai-inference-optimization.

[22] NVIDIA. Deploying disaggregated LLM inference workloads on Kubernetes[EB/OL]. 2026[2026-04-24]. https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/.

[23] NVIDIA. Enhancing distributed inference performance with the NVIDIA inference transfer library[EB/OL]. 2026[2026-04-24]. https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/.

[24] Anthropic. Subagents[EB/OL]. Claude Code Documentation[2026-04-24]. https://docs.anthropic.com/en/docs/claude-code/sub-agents.

[25] OpenClaw. openclaw/openclaw README[EB/OL]. [2026-04-24]. https://github.com/openclaw/openclaw.

[26] Kimi. Kimi introduces Agent Swarm: let 100 AI agents work for you[EB/OL]. 2026[2026-04-24]. https://www.kimi.com/blog/agent-swarm.html.

[27] 火山引擎. 不止对话，更能执行！火山引擎 Mobile Use Agent 全新升级，解锁企业级移动 AI 执行力[EB/OL]. 2026[2026-04-24]. https://developer.volcengine.com/articles/7628489608359395369.

[28] NVIDIA. Scaling large MoE models with wide expert parallelism on NVL72 rack-scale systems[EB/OL]. 2025[2026-04-24]. https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/.

[29] NetApp. KV cache offloading — CPU RAM vs. storage[EB/OL]. 2025[2026-04-24]. https://community.netapp.com/t5/Tech-ONTAP-Blogs/KV-cache-offloading-CPU-RAM-vs-storage/ba-p/464463.

[30] PatSnap. MoE inference cost cuts: 30+ patents analyzed[EB/OL]. 2026[2026-04-24]. https://www.patsnap.com/resources/blog/articles/moe-inference-cost-cuts-30-patents-analyzed/.
