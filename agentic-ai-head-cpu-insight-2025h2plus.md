# Agentic AI 推理所需“机头 CPU”洞察

> Updated: 2026-04-24  
> Date boundary: 本文只采纳 **2025-07-01 及以后**公开发布的资料。  
> Scope: 这里的“机头 CPU”指 GPU 推理节点上的 host CPU / control-plane CPU。本文重点关注三类场景：`算子下发（route / prefill / decode / transfer dispatch）`、`KV 卸载`、`MoE expert dispatch`。不讨论 2025 年 7 月之前的旧论文结论，除非仅用于术语解释。

## 执行摘要

- 2025 年下半年之后，agentic inference 的瓶颈不再只在 GPU FLOPS，而是越来越多地落在 **host 侧调度、KV 生命周期管理、跨阶段数据搬运和 expert 路由** 上。
- 对“机头 CPU”最关键的不是纯算力峰值，而是 **高并发小任务调度能力、足够的内存容量与带宽、低抖动 I/O 栈、以及和 GPU / NIC / DPU 协同的数据通路**。
- 在 `算子下发` 场景中，CPU 的角色正从“发 kernel 的传统 host”扩展为 **prefill / decode 解耦、router placement、NIXL/存储传输、Kubernetes control-plane 粘合层**。
- 在 `KV 卸载` 场景中，CPU 内存已经从“慢速兜底层”升级为 **GPU 之外的首层温热容量层**；关键指标变成容量、持续带宽、页表/pinning/IOMMU 开销和恢复路径尾延迟。
- 在 `MoE` 场景中，专家分发和 expert parallel 把 host 侧压力从“单请求驱动”推向 **批级路由、通信编排、拓扑感知放置和跨 GPU/跨节点数据流协同**。
- 2026 年的平台演进也在给出同一方向的产品信号：机头 CPU 正被当成 **AI factory 的控制平面** 来设计，高带宽主机内存、一致性互连，以及与 DPU/SuperNIC 的协同都开始直接服务于推理编排。
- 因此，面向 agentic AI 的服务器选型，不应只问“配几张 GPU”，而要问：`host CPU 是否能稳住 route + transfer + KV tiering + MoE orchestration 这条关键路径`。

## 概念边界

### 1. 什么叫“算子下发”

本文中的“算子下发”不是编译器语境里的 kernel fusion，而是更接近推理系统运行期的 host 侧下发与编排，包括：

- request 到达后的 `router / scheduler` 决策
- `prefill` 与 `decode` 的解耦调度
- KV block 的 `offload / reload / prefetch`
- MoE token/expert 的分发和跨设备搬运
- RDMA / NVLink / NIXL / storage 传输的触发与管线化

### 2. 为什么 agentic AI 会抬高 host CPU 重要性

2026-04-17 的 NVIDIA Dynamo agentic inference 文章给出一组很关键的数据：在 agentic workload 中，后续调用的 cache hit 可达 **85%-97%**，4 个 teammate agent 聚合后可到 **97.2%**，累计 **read/write ratio 为 11.7x**。这意味着系统的价值重心从“多写一点新 KV”转到“把旧状态留住、路由对、提前取回、避免重算”。

<img src="assets/nvidia-dynamo-agentic-kv-readwrite-2026.webp" alt="Agentic inference 中 KV 累积读取显著高于写入" width="760">

图：NVIDIA 给出的 agentic inference KV 读写曲线，说明 agentic 工作负载已经明显呈现 `write-once-read-many`。这直接抬高了 host 侧在保留、路由、预取和恢复上的权重。  
来源：NVIDIA, *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo*, 2026-04-17, https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/

## 现状判断：机头 CPU 已进入推理关键路径

### 1. 算子下发正在从“单机 kernel launch”演化为“分离式推理编排”

2026-03-23 的 NVIDIA Kubernetes 文章把 `disaggregated LLM inference` 明确拆成 `ingress-router`、`prefill worker`、`decode worker`，并用 NIXL 负责节点间高吞吐数据传输。这个拆法本身就说明，host 侧需要承担的事情已经不仅是把请求扔给一张 GPU，而是：

- 决定请求该进哪个 router
- 把 prefill 产物和 KV 状态送到后续 decode worker
- 配合调度器维持 worker affinity、拓扑感知和恢复路径
- 让 Kubernetes / service mesh / transfer library 与推理 runtime 连成一条低抖动数据面

<img src="assets/nvidia-k8s-disagg-serving-2026.webp" alt="Disaggregated LLM inference 将 ingress router、prefill 和 decode 分开部署" width="760">

图：NVIDIA 在 Kubernetes 上展示的解耦式推理拓扑。它反映出 host 侧职责从“单机发命令”扩展为 `router + stage scheduling + transfer orchestration`。  
来源：NVIDIA, *Deploying Disaggregated LLM Inference Workloads on Kubernetes*, 2026-03-23, https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/

对 CPU 的直接含义是：

- **单核尾延迟要稳**：router、状态机、完成队列和中断处理都怕抖动
- **核数不能过低**：agentic workload 带来更多并发的短控制任务，而非单一长核函数
- **PCIe / NIC / GPU 拓扑要友好**：否则 dispatch 决策和实际数据流不一致
- **软件栈要轻**：K8s、runtime、transfer library、监控 agent 本身都会吃 host 资源

### 2. NIXL/数据搬运把 host CPU 从“控制面”推向“轻数据面”

2026-03-09 的 NVIDIA Inference Transfer Library（NIXL）文章强调它支持 `RDMA, NVLink, TCP sockets, NVMe-oF`，目标是给分布式推理提供统一的数据移动抽象。这里的重点不是“又多了一个通信库”，而是：

- agentic inference 的真实瓶颈正在转向 **跨阶段、跨节点的状态搬运**
- host CPU 需要持续驱动 completion、buffer lifecycle、memory registration 和异步管线
- 如果 CPU 侧内存子系统、IOMMU 或中断处理不稳，GPU 闲等会直接放大

基于这些资料，可以做出一个稳健推断：  
对 agentic 推理节点，**host CPU 的价值正在从“发命令”向“编排+搬运+恢复”三合一角色迁移**。

### 3. 平台信号：Vera、Rubin 与 BlueField-4 说明 CPU 正被前移为控制平面

补充资料里有一组对当前稿有增量价值的材料：它不是再去证明“CPU 有瓶颈”，而是提供了一个更强的产品信号，即 2026 年的平台设计已经开始把 host CPU 明确当成 agentic / inference orchestration 的控制平面。

其中最值得吸收的是两点：

- **Vera CPU 的设计方向**：88 核 Olympus、`1.2 TB/s` LPDDR5X、`1.8 TB/s` NVLink-C2C，显示厂商正在用“高每核带宽 + 高一致性互连”去服务机头调度，而不只是追传统通用 CPU 指标。
- **BlueField-4 / SuperNIC 的协同方向**：网络、存储与安全处理进一步从 host CPU 旁路出去，意味着平台设计已经默认“CPU 应该把预算优先花在推理编排，而不是杂项数据面任务上”。

这类信号本身不直接证明某个具体 serving 栈一定会获得多少推理收益，但它强化了本文的主判断：  
**机头 CPU 已经不是 GPU 服务器里的配套部件，而是在新平台里被当成第一层编排器来定义。**

<img src="assets/nvidia-vera-cpu-architecture.png" alt="NVIDIA Vera CPU 架构概览" width="760">

图：Vera CPU 的高带宽内存与高带宽 CPU-GPU 互连，说明平台设计正在围绕 agentic orchestration 的控制平面需求展开。  
来源：NVIDIA, *NVIDIA Vera CPU Delivers High Performance, Bandwidth, and Efficiency for AI Factories*, 2026-03, https://developer.nvidia.com/blog/nvidia-vera-cpu-delivers-high-performance-bandwidth-and-efficiency-for-ai-factories/

<img src="assets/nvidia-vera-rubin-6chips.png" alt="NVIDIA Vera Rubin 六芯片协同架构" width="760">

图：Vera、Rubin、NVLink Switch、BlueField-4 和以太网交换机构成更强耦合的平台，反映出机头 CPU 在系统级编排中的位置前移。  
来源：StorageReview, *NVIDIA Launches Vera Rubin Architecture at CES 2026*, 2026-01, https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack

<img src="assets/nvidia-bluefield4.png" alt="BlueField-4 DPU 架构" width="760">

图：BlueField-4 进一步卸载网络、存储和安全路径，使 host CPU 更聚焦于调度、状态和推理编排。  
来源：StorageReview, *NVIDIA Launches Vera Rubin Architecture at CES 2026*, 2026-01, https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack

## 场景一：KV 卸载要求什么样的机头 CPU

### 1. CPU 内存已经是温热层，而不是单纯 spill 层

2025-09-18 的 NVIDIA Dynamo KV 文章把 KV offload 明确写成可落到 `CPU RAM`、`local SSD` 和 `network storage`。这不是简单的容量扩展表述，而是在承认推理系统正在走向分层状态存储。

2025-09-05 的 NVIDIA CPU-GPU memory sharing 文章进一步指出，Grace Hopper / Grace Blackwell 可以通过 **NVLink-C2C 900 GB/s** 的 coherent interconnect 共享统一内存地址空间。对机头 CPU 的含义非常直接：

- CPU DRAM 不再只是慢速备份，而是恢复路径中的 **首层温热 KV 池**
- 页表、pinning、IOMMU、NUMA 选路开始真实影响 token latency
- host 内存容量和持续带宽会比传统“GPU 服务器配个普通 CPU”更关键

<img src="assets/cpu-gpu-unified-memory.webp" alt="CPU-GPU 统一地址空间让 CPU 内存更适合作为 KV 温热层" width="760">

图：CPU 与 GPU 通过高带宽一致性互连共享地址空间，使 host memory 更适合承担 KV 的 overflow / staging / warm tier。  
来源：NVIDIA, *Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing*, 2025-09-05, https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/

### 2. Agentic AI 把 KV 管理从“容量问题”升级成“生命周期问题”

2026-04-17 的 Dynamo agentic inference 文章把 `retention`、`routing`、`prefetch` 和 `WORM-like reuse` 放在同一套优化框架里。对应到机头 CPU，意味着：

- 不能只看 DRAM 容量，还要看 **恢复时能多快把 KV 拉回 GPU**
- 不能只看平均带宽，还要看 **pause-resume 的尾延迟**
- 不能只看 page cache，还要看 **pinned memory / hugepage / IOMMU map 更新** 的管理成本

### 3. KV 卸载对 CPU 选型的具体启示

- **内存容量优先级上升**：agentic session、长前缀、共享模板和多 agent 扇出会一起抬高 host-side warm KV 容量需求
- **内存带宽要足够持续**：不是一次性爆发，而是持续 serving 下的平稳数据回填
- **NUMA 拓扑要和 GPU/NIC 对齐**：否则 offload/reload 的路径绕远，CPU 只会更忙
- **I/O 栈要低抖动**：KV tiering 接入本地 SSD 或网络存储后，host 成为真正的 state broker

### 4. CXL 进一步把“主机内存层”从技术问题变成经济问题

补充资料还带来了一条当前稿里原先没有展开的证据链：CXL 扩展内存不是单纯的“更大内存”，而是在把 host-side KV tiering 变成一项可量化的资源替代策略。

Astera Labs 在 2025-11 的公开材料中给出了一组生产建模信号：

- GPU 需求可下降 **87%**
- Prefill 阶段 GPU 利用率可提升 **75%**
- 每查询 CPU 利用率可下降 **40%**
- 可支持约 **2x** 并发实例

这些数字本身来自厂商建模，不应当被当成通用部署结果；但它们足以说明一件更稳定的事：  
**当 KV cache 的 warm tier 从“只能放主机 DRAM”扩展到“主机 DRAM + CXL memory”，机头 CPU 的价值就不只是容量兜底，而是整个推理经济模型的一部分。**

因此，KV 卸载章节还应补上一个更完整的判断：

- 对 co-located GPU 节点，重点是 `HBM -> coherent CPU memory` 的快速回填
- 对容量优先型节点，重点是 `HBM -> host DRAM / CXL tier` 的分层经济性
- 对多租户/长上下文场景，重点不只是“能不能卸”，而是“卸到哪一层最划算”

## 场景二：MoE 把“机头 CPU”变成路由与通信编排器

### 1. 2025H2 之后的公开资料显示，MoE 已经不是单卡 kernel 问题

2025-12-18 的 NVIDIA Wide Expert Parallelism 文章讨论在 NVL72 机架级系统上扩展大规模 MoE。文章核心不是单个 expert 算得多快，而是：

- token 如何按 expert 路由
- expert 如何在大规模 GPU 间放置
- 通信如何被压平、隐藏或重叠
- 系统如何把 expert parallel 从“能跑”推进到“高吞吐可部署”

<img src="assets/nvidia-wide-ep-moe-2025.webp" alt="Wide expert parallelism 展示了 MoE 在大规模 GPU 系统上的跨设备组织方式" width="760">

图：NVIDIA 关于 wide expert parallelism 的示意图，强调 MoE 推理的关键已经扩展到 expert 路由、并行放置和通信拓扑，而非仅是单个 GEMM。  
来源：NVIDIA, *Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack-Scale Systems*, 2025-12-18, https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/

### 2. 对 CPU 的实际要求是什么

MoE 看起来是 GPU 场景，但对 host CPU 的要求反而更尖锐，因为 CPU 必须配合做：

- **批级路由组织**：token/expert 映射、微批分组、跨 worker 调度
- **拓扑感知编排**：哪些 expert 留在本机、哪些跨节点，直接决定通信图
- **数据搬运与完成队列管理**：EP/all-to-all 的控制与生命周期管理并不全在 GPU 内部自发完成
- **与服务框架协同**：MoE 常和 disaggregated serving、prefill/decode split、KV reuse 一起出现，而不是单独出现

一个重要推断是：  
随着 agentic workload 引入更多多阶段推理、上下文复用和多代理并发，MoE 的 host-side 复杂度不会下降，反而会与 `route + transfer + KV` 三者叠加。因此，机头 CPU 更像 **orchestration processor**，而不是“配套 CPU”。

## 场景三：agentic 推理里的算子下发，本质上是状态驱动调度

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

### 2. 因此机头 CPU 最怕什么

- **低核数但高 GPU 密度**：GPU 多、host 少，会让 router 和 transfer 成为新的窄口
- **内存小但想靠 NVMe 补**：会导致 warm KV 下沉过深，恢复路径抖动明显
- **NUMA 与 GPU/NIC 拓扑错配**：CPU 线程和 DMA 路径对不上时，理论带宽没有意义
- **把所有系统服务都堆到同一 host**：K8s、telemetry、storage client、runtime sidecar 都会抢 CPU

## 机头 CPU 的工程画像

结合上述证据，更合理的 agentic AI 节点 CPU 画像是：

- **更多可用核心，而不是只追单核峰值**：因为要扛 router、scheduler、prefetch、completion、存储 client、K8s/runtime sidecar
- **更大的主机内存**：KV warm tier、agent state、prefill/decode 中间态都会吃容量
- **更好的内存带宽与 NUMA 可控性**：影响 KV reload/prefetch 的可持续效率
- **更强的 I/O 与 DMA 协同能力**：RDMA / NVLink / NVMe-oF / NIC 队列管理都会经过 host 协调
- **更明确的分层部署思路**：co-located GPU 节点更看重一致性互连与主机内存带宽，容量型节点更看重 DRAM/CXL tier 的成本效率
- **更稳的尾延迟而不是只看均值**：agentic 场景对 pause-resume 和多阶段拼接极度敏感

如果要把这几条压缩成一句话：  
**agentic AI 时代的机头 CPU，核心任务不是“替 GPU 算”，而是“确保 GPU 不因路由、状态和数据搬运而空等”。**

## 采购与架构建议

### 1. 适合优先加码 CPU 的场景

- 多 agent / 长生命周期会话 / 多阶段推理
- 明确依赖 KV offload、resume、prefix reuse
- 使用 disaggregated serving 或者准备上 Kubernetes 化部署
- 准备在 MoE 上做大规模 expert parallel

### 2. 可以相对弱化 CPU 的场景

- 单轮短对话、前缀复用弱、几乎不做多阶段推理切换
- 不做 KV offload，也不做 prefill/decode 解耦
- 模型规模和并发都小，系统仍接近单机直连推理

### 3. 一个实用判断标准

如果你的服务已经出现下面任一迹象，就不该再把 host CPU 当成配角：

- GPU 利用率起伏很大，但显存和 FLOPS 并未打满
- 多阶段 resume 的尾延迟明显高于纯 decode
- KV 命中率高，但端到端时延改善不成比例
- MoE 扩容后吞吐没按 GPU 数线性增长
- K8s / runtime / transfer sidecar 一开就吃掉大量 host core

## 典型 Agentic Workload 反推：还有哪些 CPU 需求容易漏掉

> 这一节只看 **agentic LLM inference 对 CPU 的影响**。不把工具本身的执行开销算进来，例如浏览器渲染、ADB 点击、代码编译、搜索抓取等直接工具负载都排除在外。

### 1. OpenClaw / 手机 GUI Agent 类：不能低估多模态 prefill 和高频短回合切换

OpenClaw 官方仓库已经把产品形态定义为 `always-on` 的 personal AI assistant，并且覆盖 Android node、screen recording、camera、Canvas、device pairing 等持续在线入口。火山引擎在 2026-04-29 发布的 Mobile Use Agent 文章，则把这类产品进一步明确成基于云手机与豆包视觉模型的 `Mobile Use Agent`。

基于这两类产品形态，可以做出较强但仍属推断的判断：对 host CPU 来说，即使完全不算工具执行本身，仍然会新增三类推理侧压力：

- **多模态 prefill 压力**：这类 GUI agent 往往需要把截图或界面状态送入模型，prefill 很可能比纯文本 agent 更重
- **高频短回合调度**：这类交互常表现为短回合、频繁状态刷新，decode 未必长，但请求切换频繁
- **更细粒度的 KV 生命周期管理**：单步推理可能较短，但状态连续性要求更高，host 更可能频繁做 session pinning、warm KV 保留和 resume

这意味着，原报告里“KV 卸载”和“算子下发”的判断是对的，但还少强调了一点：  
**GUI agent 会把 CPU 压力从长 decode 进一步推向高频 prefill + 高频状态切换。**

### 2. Claude Code 类：不能低估长驻上下文和多 subagent 并发

Anthropic 官方文档已明确说明，Claude Code `subagents` 各自拥有 `separate context window`，并且会因单独收集所需上下文而带来额外延迟。这件事对推理侧 CPU 的真正含义不是“工具多”，而是：

- **会话数暴增**：一个主代理外加多个 subagents，等价于更多并行或准并行上下文
- **prefill 占比上升**：subagent 带着干净上下文启动，天然更容易形成 “短 burst + 重 prefill”
- **KV 复用更偏局部而不是全局**：主代理和子代理不会天然共享同一整块上下文，host 需要更细地做 session-level placement 和复用决策

因此，对 Claude Code 这类 workload，原报告还漏了一项明确表述：  
**host CPU 需要为“多上下文并存”而不是“单上下文超长”做优化。**  
这会抬高 admission control、per-session queue、KV placement 和尾延迟隔离的重要性。

### 3. Kimi Agent Swarm 类：不能低估 fan-out/fan-in 的瞬时并发宽度

Kimi 官方在 2026-04-11 的 Agent Swarm 文章里给出的产品形态非常直接：`up to 100 sub-agents working in parallel`。如果只看推理，不看工具执行本身，这种 workload 依然会给机头 CPU 带来一个此前还不够突出的要求：

- **瞬时 fan-out 调度能力**：大量子代理会在短时间内同时进入 prefill 或 decode
- **返回汇总时的 fan-in 压力**：上层代理需要消化来自多子代理的中间输出，再触发下一轮推理
- **批处理与公平性冲突**：为了提吞吐，系统会想做 batch；但 swarm workload 又容易因为宽并发而拖高尾延迟

因此，Kimi Swarm 补充出的遗漏点是：  
**机头 CPU 不只是要“能跑高并发”，还要能处理极宽的 burst 并发和多层级调度。**

### 4. 这些真实 workload 共同补出的三项遗漏

如果把 OpenClaw、Claude Code、豆包手机/Mobile Use Agent、Kimi Swarm 放在一起看，原报告还应补上三条更聚焦的 CPU 诉求：

- **遗漏一：prefill-first CPU 观念**  
  过去容易把 host 压力理解成 decode 陪跑，但真实 agentic workload 很多时候更像 `频繁 prefill + 短 decode + 快速 resume`
- **遗漏二：session multiplicity 而非单 session 长度**  
  Claude Code 和 Kimi Swarm 说明问题不只是上下文长，而是同时活跃的上下文条目太多
- **遗漏三：多模态 ingress 开销**  
  豆包手机/OpenClaw phone agent 说明，即便不算工具 CPU，截图/视觉输入进入推理链路本身也会放大 host 侧请求编排与内存压力

换句话说，若只从底层 serving 论文出发，很容易得出“CPU 主要是 KV tiering + transfer”这个结论；但把真实产品 workload 放进来后，应该把结论修正为：

**agentic LLM inference 对机头 CPU 的新增要求，除了 KV tiering 和 transfer，还包括 `高频 prefill 调度`、`多上下文并存管理`、`极宽 fan-out/fan-in` 和 `多模态 ingress 编排`。**

## 证据表

| Date | Title | Type | Topic | Main finding | Relevance | Link |
| --- | --- | --- | --- | --- | --- | --- |
| 2025-09-05 | Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing | NVIDIA 技术文章 | KV offload / unified memory | Grace Hopper / Grace Blackwell 通过 NVLink-C2C 900 GB/s coherent interconnect 共享内存地址空间 | 直接支撑“CPU 内存成为 warm KV 层” | https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/ |
| 2025-09-18 | How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo | NVIDIA 技术文章 | KV offload hierarchy | KV cache 可卸到 CPU RAM、local SSD、network storage | 说明 host CPU 已是分层状态存储的中枢 | https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/ |
| 2025-12-18 | Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack-Scale Systems | NVIDIA 技术文章 | MoE / expert parallel | 重点落在 expert 路由、放置和跨 GPU 通信，而非单一算子 | 支撑“MoE 抬高 host-side orchestration 要求” | https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/ |
| 2026-03-09 | Enhancing Distributed Inference Performance with the NVIDIA Inference Transfer Library | NVIDIA 技术文章 | transfer dispatch | NIXL 统一 RDMA、NVLink、TCP sockets、NVMe-oF 等数据通路 | 支撑 host CPU 正在进入轻数据面 | https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/ |
| 2026-03-23 | Deploying Disaggregated LLM Inference Workloads on Kubernetes | NVIDIA 技术文章 | disaggregated serving / operator dispatch | ingress-router、prefill worker、decode worker 解耦部署 | 直接对应“算子下发”从单机发命令变成运行期编排 | https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/ |
| 2026-04-17 | Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo | NVIDIA 技术文章 | agentic inference / KV lifecycle | 85%-97% hit、97.2% aggregate hit、11.7x read/write ratio | 说明 agentic workload 已让保留、路由和预取成为关键路径 | https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/ |
| 2026-03 | NVIDIA Vera CPU Delivers High Performance, Bandwidth, and Efficiency for AI Factories | NVIDIA 技术文章 | platform / head CPU | Vera 采用高带宽 LPDDR5X 与 NVLink-C2C，明确面向 AI factory 控制平面 | 强化“机头 CPU 被前移为系统编排层”的产品信号 | https://developer.nvidia.com/blog/nvidia-vera-cpu-delivers-high-performance-bandwidth-and-efficiency-for-ai-factories/ |
| 2026-01 | NVIDIA Launches Vera Rubin Architecture at CES 2026 | 产业评测 / 架构综述 | platform topology | Vera、Rubin、BlueField-4、Switch 被作为一体化平台描述 | 补充系统平台级信号与图示 | https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack |
| 2025-11 | How CXL Transforms RAG and KV Cache Performance | 厂商技术博客 | CXL / KV tiering economics | CXL memory 扩展被用于降低 GPU 需求、提升利用率和并发 | 强化“KV warm tier 是成本与容量共同问题” | https://www.asteralabs.com/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance/ |
| 2026-04-11 | Kimi Introduces Agent Swarm: Let 100 AI Agents Work for You | Kimi 官方博客 | multi-agent fan-out | up to 100 sub-agents working in parallel | 支撑“极宽并发会把 host 压力推向 fan-out/fan-in 调度” | https://www.kimi.com/blog/agent-swarm.html |
| 2026-04-29 | 不止对话，更能执行！火山引擎 Mobile Use Agent 全新升级，解锁企业级移动 AI 执行力 | 火山引擎官方文章 | mobile GUI agent | 基于云手机底座与豆包视觉模型构建 enterprise Android agent | 支撑“手机 GUI agent 会抬高多模态 prefill 与高频状态切换压力” | https://developer.volcengine.com/articles/7628489608359395369 |
| Current doc, publish date not exposed | Subagents | Anthropic 官方文档 | Claude Code subagents | subagents use a separate context window and may add latency as they gather context | 作为补充说明，支撑“多上下文并存、额外上下文收集延迟”这一 CPU 负载特征；发布时间未公开，因此不作为日期边界内主证据 | https://docs.anthropic.com/en/docs/claude-code/sub-agents |
| 2026-04-14 release window | openclaw/openclaw README | OpenClaw 官方仓库 | always-on personal agent / Android node | always-on personal assistant with Android node, screen recording, camera and canvas surfaces | 支撑“手机/常驻 agent 会引入高频多模态 inference ingress” | https://github.com/openclaw/openclaw |

## 结论

只看 2025 年下半年及以后资料，可以得到一个比 2024-2025 上半年更明确的结论：  
**agentic AI 负载下，机头 CPU 已经从“辅助控制器”升级为 inference orchestration engine。**

它至少同时承担四件事：

- `dispatch`：负责 route、prefill/decode split、恢复与 fan-out/fan-in
- `tiering`：负责 KV 的 warm-tier 管理、offload 与 reload
- `transfer`：负责 NIXL / RDMA / NVMe-oF / NIC / DPU 协同的数据流触发
- `coordination`：负责 MoE expert placement、拓扑感知和运行期状态一致性

因此，如果未来要为 agentic AI 服务器定义“好 CPU”，最应优先关注的不是传统通用 benchmark，而是：

- 能否扛高并发控制任务
- 能否提供足够大的、低抖动的 host memory tier
- 能否与 GPU/NIC/storage 构成稳定的数据通路
- 能否根据节点类型在 `coherent CPU memory` 与 `DRAM/CXL capacity tier` 之间做合理分层
- 能否在 MoE 与 disaggregated serving 下维持可预测的尾延迟

## 来源列表

- NVIDIA, *Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing* (2025-09-05): https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/
- NVIDIA, *How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo* (2025-09-18): https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/
- NVIDIA, *Scaling Large MoE Models with Wide Expert Parallelism on NVL72 Rack-Scale Systems* (2025-12-18): https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/
- NVIDIA, *Enhancing Distributed Inference Performance with the NVIDIA Inference Transfer Library* (2026-03-09): https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/
- NVIDIA, *Deploying Disaggregated LLM Inference Workloads on Kubernetes* (2026-03-23): https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/
- NVIDIA, *Full-Stack Optimizations for Agentic Inference with NVIDIA Dynamo* (2026-04-17): https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/
- NVIDIA, *NVIDIA Vera CPU Delivers High Performance, Bandwidth, and Efficiency for AI Factories* (2026-03): https://developer.nvidia.com/blog/nvidia-vera-cpu-delivers-high-performance-bandwidth-and-efficiency-for-ai-factories/
- StorageReview, *NVIDIA Launches Vera Rubin Architecture at CES 2026* (2026-01): https://www.storagereview.com/news/nvidia-launches-vera-rubin-architecture-at-ces-2026-the-vr-nvl72-rack
- Astera Labs, *How CXL Transforms RAG and KV Cache Performance* (2025-11): https://www.asteralabs.com/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance/
- Kimi, *Kimi Introduces Agent Swarm: Let 100 AI Agents Work for You* (2026-04-11): https://www.kimi.com/blog/agent-swarm.html
- 火山引擎, *不止对话，更能执行！火山引擎 Mobile Use Agent 全新升级，解锁企业级移动 AI 执行力* (2026-04-29): https://developer.volcengine.com/articles/7628489608359395369
- OpenClaw, *openclaw/openclaw README* (release window referenced: 2026-04-14): https://github.com/openclaw/openclaw
- Anthropic, *Subagents* (current official documentation; publish date not exposed, used only as supplementary evidence): https://docs.anthropic.com/en/docs/claude-code/sub-agents
