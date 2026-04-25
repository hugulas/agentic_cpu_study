---
title: "Agentic AI 推理中机头 CPU 的系统作用：算子下发、KV 生命周期、MoE 编排与平台演化"
subtitle: "综述"
date: "2026-04-24"
lang: "zh-CN"
toc: true
toc-depth: 2
numbersections: true
---
# Agentic AI 推理中机头 CPU 的系统作用：算子下发、KV 生命周期、MoE 编排与平台演化

本文聚焦 `agentic LLM inference` 对机头 CPU 的影响，主体证据采用 `2025-07-01` 及之后公开发布的论文、厂商技术材料与产品文档；个别未公开精确发布日期的官方文档仅作补充说明。全文不将工具执行本身的 CPU 消耗纳入主结论，而将分析重点放在请求接入、阶段切换、状态分层、跨池传输、专家驻留和多代理并发控制等推理侧机制上。

## 摘要

Agentic AI 推理正在把系统优化的重心，从“让 GPU 算得更快”推进到“让整个推理控制链路不阻塞 GPU”。围绕这一变化，现有证据可以收敛成五条主线：`算子下发与状态驱动调度`、`KV cache 卸载与生命周期管理`、`MoE 的 host-side orchestration`、`真实 agentic workload 对传统 serving 假设的修正`、以及 `Vera / Rubin / BlueField-4 / CXL` 所代表的平台侧收敛。综合这些材料，较稳健的结论是：机头 CPU 的角色已经从传统 host 演化为推理系统的第一层 orchestrator，其核心任务不再是辅助 GPU，而是管理请求接入、阶段切换、KV 放置、跨池传输、专家驻留与多代理并发控制，避免 GPU 因调度、状态和传输失配而空等 [1][2][6][9][10][11][12][22][23]。  

**关键词：** agentic AI；LLM inference；host CPU；operator dispatch；KV cache offloading；MoE；prefill-decode disaggregation

\newpage

## 一、总论：为什么机头 CPU 成为 agentic 推理的关键变量

### 1.1 核心判断

如果把现有论文、厂商文档与产品工作负载放在一起看，机头 CPU 的重要性并不是因为它重新承担矩阵运算，而是因为 agentic inference 改写了系统瓶颈的分布方式：  

1. 请求不再是单轮、连续、稳定的 decode 流。  
2. 状态不再是一次性中间产物，而是需要保留、恢复、分叉和复用的长期对象。  
3. GPU 不再只依赖本地 HBM，而要与 host memory、CXL、网络存储和跨节点 worker 协同。  
4. 多代理、子会话和多模态 ingress 让 host 侧调度链路显著变长。  

因此，对 agentic inference 而言，真正稀缺的不只是 GPU FLOPS，而是 `host-side orchestration budget` [2][9][16][22][23]。

这里最值得讲透的“为什么”是：agentic inference 并没有削弱 GPU 的重要性，而是改变了 GPU 发挥价值的前提。过去，系统只要把一条相对稳定的 token 流持续送进 GPU，很多 host 侧动作都可以被隐藏在长 decode 窗口里；现在，请求更像一组会被暂停、恢复、分叉和汇聚的状态对象。只要状态对象的形状持续变化，系统就需要不断重新判断下一步应该在哪跑、从哪层恢复、和哪些对象并批、以及该把哪些数据提前迁到更近位置 [9][22][24][26]。

也就是说，agentic inference 的本质变化不是“请求更多了”，而是“每单位 GPU 计算对应的 host 决策更多了”。一旦每次推进前都需要更多控制动作，CPU 就不再只是为 GPU 提供外围服务，而开始决定 GPU 是否能够连续、高效地工作。因此，这一章的核心判断并不是“CPU 变快很重要”，而是“控制平面是否稳定，已经成为 GPU 价值能否兑现的前提”。

<img src="assets/cpu-centric-agentic-workflow.png" alt="Agentic inference 中 CPU 作为控制中枢的工作流示意" width="760">

**图1** Agentic 推理中的 CPU-centric workflow。该图从工作流层面展示了 CPU 如何介入请求接入、状态管理、传输触发和 GPU 执行协调，适合作为全文总图。来源：作者根据解耦 serving、agentic KV、subagent / swarm / mobile agent 等公开资料整理 [9][22][24][26][27]。

### 1.2 本文的分析框架

本文采用“总分总”结构，但更具体地说，可以划分为三组连续展开的内容。

1. `命题层`：第二章先给出中心命题与证据地图，明确全文到底要证明什么。  
2. `机制层`：第三至第五章分别讨论算子下发、KV 生命周期与 MoE 编排，解释 CPU 为什么会进入关键路径。  
3. `验证与落地层`：第六至第九章依次讨论真实工作负载、平台演化、工程部署与评测框架，说明这些机制判断为何不仅成立，而且足以影响平台设计与工程选型。  

最后，第十章以附录形式集中列出关键数据，第十一、十二章再回到证据边界与全文结论。

这种组织方式的目的，不只是为了排版清楚，而是为了把一条因果链完整展开：先明确中心命题，再解释问题在哪些机制层面形成，随后验证这些机制在真实工作负载、平台路线图和工程实践中如何被放大，最后再回到结论与边界。换句话说，本文不是把材料按来源堆在一起，而是试图证明一件事：机头 CPU 在 agentic inference 中的重要性，是由多条独立证据链共同推出来的，而不是由单一厂商、单篇论文或单类产品工作负载强行宣称出来的。

\newpage

## 二、核心命题与证据概览

本章的作用不是重复后文内容，而是先将全文最重要的判断、对应证据和后续展开位置放在同一处，帮助读者建立稳定的阅读主线。若将全文压缩为一个中心命题，则可以表述为：在 agentic inference 中，机头 CPU 的核心价值不在于补充 GPU 的算力，而在于承担请求接入、状态组织、分层恢复、跨池传输和多阶段切换所需的控制职责；正是这些控制职责决定了 GPU 能否持续处于高利用率状态。

围绕这一中心命题，现有材料大体收敛为五组相互印证的证据。算子下发相关工作说明，量化、小模型和多 GPU 同步会把 host 抖动放大为系统级停顿；KV 相关工作说明，状态对象已经从容量副产物转变为需要持续放置与恢复的生命周期对象；MoE 相关工作说明，稀疏激活并不会降低 host 的协调负担，反而会抬高路由、搬运和驻留管理的重要性；真实工作负载表明，agentic 请求形态会系统性放大 prefill、resume 和多会话并发压力；平台路线图则从硬件组织层面对“CPU 作为控制平面”给出了侧面确认 [2][9][11][13][22][24][28]。

为避免后文看起来像彼此独立的专题堆叠，表 1 先将核心命题、证据类型与章节位置对应起来。

| 命题 | 支撑证据 | 关键含义 | 详见章节 |
| --- | --- | --- | --- |
| CPU 已进入推理关键路径 | CPU slowdown、PD 分离、KV read/write ratio、MoE orchestration [2][9][11][22] | GPU 性能上限越来越取决于 host 控制链路质量 | 第三至第五章 |
| CPU 问题本质是控制链路过长 | persistent batch、CUDA graphs、KV transfer、decoupled residency [4][5][12][23] | 优化重点是减少同步控制动作 | 第三至第五章 |
| KV 已成为生命周期对象 | NOSA、ScoutAttention、CoMEM、coherent memory、CXL [6][7][8][10][15] | 状态分层、预取与恢复成为长期系统问题 | 第四章 |
| MoE 抬高了 host 编排价值 | 宽专家并行、预测预取、驻留解耦 [11][12][28][30] | 稀疏激活降低计算量，但增加路由与搬运复杂度 | 第五章 |
| 平台已围绕控制平面开始收敛 | Vera、Rubin、BlueField-4、CPU:GPU 配比变化 [13][14][16] | 平台设计开始显式为 host orchestration 让路 | 第七、八章 |

### 2.1 命题一：机头 CPU 已进入推理关键路径

这一命题由四类证据共同支撑：多 GPU 推理中的 CPU oversubscription 会放大为集群级等待 [2]；PD 分离与解耦 serving 让 host 职责扩展到跨阶段路由和状态协调 [22][23]；agentic inference 的 `11.7x` KV read/write ratio 表明系统越来越像“恢复旧状态”而不是“持续写新状态” [9]；MoE 推理则要求 host 持续参与 expert residency、预取和通信编排 [11][12][28]。这些证据虽然表象不同，但都指向同一点：GPU 已经不再独自定义推理性能上限。

### 2.2 命题二：CPU 问题的本质不是“算得慢”，而是“控制链路太长”

在 agentic inference 中，CPU 侧真正消耗预算的并非纯计算，而是 request ingress、prefill/decode 切换、batch 更新、KV 放置与恢复、跨节点传输以及 expert routing 等控制动作。也正因如此，`persistent batch`、`CUDA graphs`、`persistent kernels`、`KV transfer library` 和 `decoupled residency` 这些优化的共同目标不是提高 CPU 算力，而是缩短同步控制链路 [4][5][11][12][23]。问题的核心不是 host 算得慢，而是 GPU 每向前一步都越来越依赖 host 先把队列、状态和传输关系整理好。

### 2.3 命题三：KV 已从容量对象变成生命周期对象

传统推理里，KV cache 常被视为长上下文带来的显存扩展；而在 agentic inference 中，KV 更像需要被保留、检索、恢复和复用的长期状态对象 [6][7][9][10]。根本原因在于访问模式不再单调：状态会被暂停、复制到分支、等待外部结果，再在之后重新进入计算路径。只要这一点成立，KV 的系统意义就不再只是容量问题，而是生命周期问题。

### 2.4 命题四：MoE 节省了 GPU 计算，却抬高了 host 编排价值

MoE 减少的是每 token 的激活计算，而不是系统编排复杂度。专家数增大、驻留空间受限、跨设备 all-to-all 与异步预取叠加后，host-side orchestration 很容易成为新的吞吐上限 [11][12][28][30]。原因在于 MoE 把“哪些参数会被访问”从静态事实变成了运行时变量，因此系统必须持续维护一条低抖动的路由、驻留与搬运链路，而这正是 host 的职责边界。

### 2.5 命题五：平台路线图已经围绕“CPU 作为控制平面”开始收敛

Vera 的高带宽主机内存、NVLink-C2C、一体化平台组织方式，以及 BlueField-4 对网络、存储与安全路径的旁路，传递的不是单点产品宣传，而是平台层面对 `CPU as orchestration plane` 的确认 [13][14][17][22][28]。平台层面的动作成本很高，不会围绕短期叙事频繁摆动；因此，一旦厂商开始同时在主机内存带宽、近端互连、DPU 旁路和系统级耦合上重新组织平台，就说明 host orchestration 已被视为长期、可工程化的系统角色。

以上五条判断共同构成全文的主干。后文的组织方式因此可以理解为一条连续展开的因果链：第三至第五章讨论控制路径如何在算子下发、KV 生命周期和 MoE 编排三个机制层面具体形成；第六章说明真实工作负载为何会放大这些机制问题；第七、八章再回到平台与部署层面，讨论这些机制判断为何已经足以影响工程决策。

## 三、机制层之一：算子下发与控制路径瓶颈

本章聚焦全文因果链中的第一环，即为什么最早看似属于 runtime 细节的 launch overhead，会在 agentic inference 中演化为更高层次的控制路径问题。这里讨论的重点不是单个 kernel 的绝对开销，而是 host 在提交、组织、同步和恢复各类执行片段时所承担的持续性控制负担。

### 3.1 量化、小模型与高并发共同放大 host 固定成本

现有测量结果和系统论文给出了一条很清晰的因果链：  

- 模型被进一步量化之后，单次 GPU 计算粒度缩小。  
- 计算粒度缩小后，kernel launch、队列维护、runtime 调度等固定成本占比迅速升高。  
- 一旦 CPU oversubscription 或 service stack 抖动进入关键路径，GPU 便会开始空等 [2][3][4][5]。  

一个具有代表性的工程测量是：某量化小模型一次前向传播发射 301 个 kernel，单个 launch 开销约 2.5 微秒，仅纯下发税就累计到约 750 微秒；通过 kernel fusion 将发射数降到 181 个后，吞吐从 1255 tok/s 提升到 1508 tok/s，提升约 20% [3]。这个结果的含义不是“launch 很慢”，而是：当 GPU 计算本体被压缩后，host 侧固定税费就会立刻浮出水面。  

这里真正需要解释清楚的是，为什么“模型更小”反而会让 CPU 更容易成为问题。原因在于，GPU 的优势来自大粒度、长持续时间的并行计算；而量化、小模型和高并发把一次推理拆成了更多、更短、更碎的执行片段。只要每个片段都还需要 host 参与提交、排队或状态更新，那么固定的 host 成本就会按片段数线性累加，而 GPU 侧的有效计算时间却在缩短。结果不是 GPU 更忙，而是 GPU 更容易在片段之间等待下一次提交 [2][3][4]。

换一个角度看，所谓“调度墙”并不是单独由 CUDA runtime 决定的。真正累积起来的包括：用户态框架组织 batch、runtime 生成提交队列、driver 触发 doorbell、worker 线程维持执行顺序、必要时还要等待跨卡同步点。这些动作单独看都不算重，但当一个请求被拆成数百个微小执行片段时，它们就会从背景噪声变成主导成本。因此，量化、小模型与高并发并不是简单地“让推理更便宜”，而是把系统从 compute-bound 推向 control-bound [2][3][5]。

### 3.2 CPU oversubscription 会把微秒问题放大成毫秒问题

《Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference》进一步把这一点系统化了 [2]。文中显示：

- 在 H100 上运行的某些 vLLM 路径中，HTTP 服务约占 33% 的执行时间，调度与输入准备约占 29%，GPU 计算仅约占 38%。  
- 在 TP=4、5 req/s、100k token 输入的场景下，广播队列 dequeue 延迟可从 12 ms 恶化到 228 ms，放大约 19 倍。  

这里的关键并不是某个具体数字，而是说明：在多 GPU serving 中，一个被抢占的 host 线程足以拖慢整个同步链路。  

之所以会这样，是因为多 GPU serving 里的 host 线程并不是彼此独立的“后台辅助线程”，而是分布式执行图的一部分。只要一个 rank 上负责 dequeue、launch、广播或同步回调的线程被 OS 调度器抢走，其他 rank 上已经准备好的 GPU 也无法继续向前推进。换句话说，GPU 集群中的等待不是只发生在设备侧 barrier，也发生在 host 侧控制动作不能按时完成的时候 [2]。

这也是为什么 CPU oversubscription 的危害远大于“本地机器慢了一点”。在单 GPU 场景中，抖动通常只影响单个请求的局部延迟；但在多 GPU、张量并行或专家并行场景中，某一台主机上的 host 抖动会被同步机制放大成整个执行组的共同等待。因此，host 线程调度质量本身已经构成了分布式推理的性能边界，而不只是外围实现细节。

<img src="assets/extracted/cpu-slowdown-01.png" alt="CPU-induced slowdown 论文中的多GPU推理慢化图" width="760">

**图2** 多 GPU LLM inference 中的 CPU-induced slowdown 示例图。该类图的意义不在于单个模型或单台机器的绝对数值，而在于直观展示 host 抖动如何被放大为端到端慢化。来源：CPU slowdown 论文图页 [2]。

### 3.3 Agentic inference 为什么比普通 chat 更容易撞上调度墙

相比“单条上下文、长 decode、稳定批次”的传统假设，agentic inference 更常见的是：

1. 高频 prefill  
2. 短 decode  
3. 状态暂停与恢复  
4. 多会话并发  
5. 子代理 fan-out/fan-in  

这让 host 侧不再只是“发起一次前向传播”，而是要持续处理 stage transition、context placement 和 request resumption [22][24][26][27]。  

这里的“为什么”在于，agentic inference 的基本执行单元已经从“一个请求对应一段连续 decode”变成“一个任务对应多个相互打断的推理片段”。每当外部观察、新状态、分支会话或多代理汇聚重新进入模型，系统就需要重新做一次与推理相关的控制工作：这次该进 prefill 还是 decode，原有状态放在哪一层，是否要把某个会话重新 pin 到某个 worker，以及当前 batch 是否仍然成立。也就是说，agentic inference 增加的不是纯 token 数，而是 host 必须处理的阶段切换次数 [22][24][26]。

这也是它与传统 chat serving 的根本差别。传统 chat 的优化大多假设上下文沿着一条相对单调的路径增长，所以系统可以把很多控制动作摊平到长 decode 流中；而 agentic inference 的状态更像一组被不断恢复、分叉、合并的会话对象。只要状态对象在频繁变化，host 侧就需要不断重新确认“下一步应该怎么跑”，从而让控制路径密度明显上升。

<img src="assets/nvidia-k8s-disagg-serving-2026.webp" alt="解耦式 LLM inference 中的 ingress router、prefill worker 和 decode worker" width="760">

**图3** 解耦式 serving 拓扑。图中 ingress router、prefill worker、decode worker 的拆分说明，host 侧已从单机 launch 扩展为跨阶段路由和状态协调中枢。来源：NVIDIA, 2026 [22]。

### 3.4 现阶段最有效的缓解路线

从现有研究与工程材料看，针对算子下发的有效路线主要有四类：

- `kernel fusion / megakernel`：减少发射次数与跨 kernel 边界同步 [3][4]。  
- `persistent batch`：避免每步重建张量和 Python 数据结构 [5]。  
- `CUDA graph / piecewise capture`：用图执行降低重复 launch 成本 [5]。  
- `CPU isolation`：为 GPU worker 留出不被抢占的 host 核心，削弱 OS 调度抖动 [2]。  

这些路线之所以有效，不是因为它们都在“优化 CPU 算力”，而是因为它们在减少 host 必须同步参与的次数。Kernel fusion 和 megakernel 直接减少提交次数；persistent batch 减少每步都要重建的 host 数据结构；CUDA graph 把重复出现的控制路径提前固化；CPU isolation 则是在无法减少控制动作时，尽量确保这些动作不会被外部噪声打断。四条路线表面不同，本质上都在降低 `host touch frequency` 或 `host touch jitter`。

小结：算子下发已经不是一个局部 runtime 问题，而是 agentic inference 下 host 侧控制链路被拉长后的系统性问题。

### 3.5 从引擎优化回看 CPU 的真实职责

如果把 vLLM V1、Event Tensor、解耦 serving 和 CPU slowdown 这几类工作放在一起看，可以更具体地界定机头 CPU 在算子下发问题中的职责边界：

- 它要决定什么时候发起下一步计算。  
- 它要决定哪些请求可以并批，哪些请求必须拆开。  
- 它要决定某一步执行结束后，状态应该留在本地、发往远端还是进入 warm tier。  
- 它还要维护 transport、queue 和 worker 之间的一致节奏。  

这说明“launch overhead”只是表层现象，底层问题其实是 `control-path density` 在 agentic inference 中上升了：每单位 GPU 计算量所对应的 host 控制动作更多、更碎、更难被隐藏 [2][4][5][22][23]。因此，只要控制路径仍然被频繁打断，GPU 算得再快，也会在系统层面失去意义。

进一步说，这一章真正要成立的核心判断不是“CPU 也参与了推理”，而是“GPU 的有效利用率越来越取决于 host 能否维持一条低抖动的控制链路”。当一个系统持续出现 TTFT 抖动、burst 下吞吐掉头或多卡环境里局部慢节点拖垮全局时，首先要怀疑的往往已经不再是 GPU kernel 本身，而是 host 侧的 batch 组织、线程调度、状态迁移和提交节奏。也正因为如此，算子下发问题在 agentic inference 中才会从微观优化点上升为架构问题。

这一定义为下一章的 KV 讨论提供了过渡。既然 host 已经需要持续维护执行节奏，那么当状态对象本身也开始跨轮次保留和恢复时，CPU 的职责就会进一步从“提交计算”扩展为“管理状态”。

\newpage

## 四、机制层之二：KV 生命周期与状态分层

如果说上一章讨论的是 CPU 如何维持计算链路，本章讨论的则是 CPU 如何维持状态链路。对 agentic inference 而言，状态对象的存放、恢复和迁移已经不再是显存容量不足时的补救措施，而是决定系统尾延迟与恢复效率的基础机制。

### 4.1 Agentic inference 改写了 KV 的访问模式

关于 KV，最重要的证据不是“KV 很大”，而是 Dynamo agentic inference 材料给出的访问模式变化：高命中率与 `11.7x` 的 read/write ratio 表明，系统的主要压力不再是不断写入新 KV，而是频繁读取、路由、恢复和复用既有 KV [9]。  

换言之，在 agentic inference 中，KV 不再只是“上下文长度的副产品”，而是可跨轮次保留、可跨 worker 转移、可被多个子代理复用的长期状态。  

这里最关键的“为什么”在于，agentic inference 改变了状态的使用方式。传统单轮对话里，KV 主要沿着一条会话时间线单向增长，写入后很快就在后续 decode 中被消费；而 agentic inference 中的状态会被暂停、恢复、复制到子代理、在分支结束后重新汇聚，甚至在下一次工具结果回来时重新进入计算路径。于是，系统面对的不再只是“是否有足够显存容纳这些 KV”，而是“这些 KV 之后还会不会被再次访问、从哪一层访问、以多快的尾延迟访问” [9][22][24][26]。

一旦状态具有这种长期性，KV 的系统属性就发生了变化。它不再像临时中间结果，而更像缓存层级中的可迁移对象：不同状态有不同热度，不同恢复路径对尾延迟容忍度不同，不同 worker 对同一批状态的访问概率也不同。也就是说，问题已经从“能不能存下”转成“如何在多层内存里持续管理这些对象”。

<img src="assets/nvidia-dynamo-agentic-kv-readwrite-2026.webp" alt="Agentic inference 中累计 KV 读取远高于写入" width="760">

**图4** Agentic inference 的 KV 读写关系。该图显示累计读取显著高于累计写入，说明真正吃预算的是恢复、预取与重用，而非持续生成新状态。来源：NVIDIA, 2026 [9]。

### 4.2 CPU 角色已从“搬运工”变为 warm-tier manager

NOSA、ScoutAttention 和 CoMEM 代表了三类不同但相互补充的路线：

- NOSA 通过稀疏注意力减少 CPU-GPU KV 传输量，在多个模型上实现最高 `5.04x` 的解码吞吐提升 [6]。  
- ScoutAttention 让 CPU 提前一层参与 attention 相关预计算，在精度损失小于 2.4% 的情况下实现约 `2.1x` 加速 [7]。  
- CoMEM 则把长生命周期 agent memory 的压缩与主推理链路异步解耦，说明 CPU 侧不仅可以做传输和预取，也可以承担部分状态整理与压缩任务 [8]。  

这些工作共同说明，CPU 在 KV 卸载系统中的位置已经变化：

- 它不是简单的数据中转站。  
- 它正在执行 placement、prefetch 和 resume path control。  

之所以会出现这种角色变化，是因为 GPU 并不天然擅长管理跨层状态对象。GPU 能高效消费已经在本地、已经按正确顺序就位的数据，但它不负责决定哪些 KV 值得保留在近端、哪些应该被降级、以及何时应当提前拉回。这个决策过程需要结合访问历史、当前任务阶段、下游恢复概率和可用带宽，而这些都更接近 host 侧调度器和内存管理器的职责边界 [6][7][8]。

NOSA、ScoutAttention 和 CoMEM 分别从三个角度说明了这一点。NOSA 通过减少必须搬运的 KV 量来降低跨层访问成本；ScoutAttention 通过让 CPU 提前一层参与预计算，把原本会阻塞恢复路径的工作前移；CoMEM 则说明长期 memory 本身也需要被压缩和整理，而不是原样堆积。三类路线背后其实都在指向同一个事实：KV 问题已经演化成 `state orchestration problem`，而 CPU 正是这类 orchestration 的天然执行者。

<img src="assets/extracted/scoutattn-1.png" alt="ScoutAttention 或类似KV卸载论文中的系统示意图" width="760">

**图5** KV 卸载/预计算类方法的系统示意图。此类图揭示的核心不是具体实现细节，而是 CPU 已从被动存储代理变成主动参与预取、预计算和恢复控制的 warm-tier 协调者。来源：ScoutAttention 论文图页 [7]。

### 4.3 KV warm tier 已经进入分层经济学阶段

Grace/Blackwell coherent CPU-GPU memory 与 CXL 相关材料说明，现代 KV 分层已经不应被简化为“HBM vs DRAM” [10][15][29]。更现实的体系是：

1. HBM  
2. coherent CPU memory  
3. host DRAM  
4. CXL memory  
5. local / remote storage  

其中，CXL 的意义不只是扩容。Astera Labs 给出的生产建模显示，CXL 内存层可使 GPU 需求下降 87%，GPU 利用率提升 75%，每查询 CPU 利用率下降 40%，并支持约 2 倍并发实例 [15]。这些数字说明，KV 分层已经从“技术补丁”变成“容量-性能-成本”的架构选择题。  

这里背后的机制是，状态分层并不会自动带来收益，只有当系统能把“热对象尽量留近、冷对象尽量放远、即将恢复对象提前拉回”这三件事做对时，多层内存才有意义。否则，扩容只会把带宽瓶颈和恢复延迟推向更远的层级。CXL 之所以重要，并不是因为它替代了 HBM，而是因为它在 host DRAM 和更冷层之间提供了一个容量更大、延迟又低于纯存储层的中间选项，使系统可以更细粒度地匹配状态热度和成本预算 [10][15][29]。

换句话说，CXL 不是孤立的硬件卖点，而是让 KV lifecycle 管理从“两层内存策略”升级为“多层内存策略”的关键条件。一旦层数增多，placement 和 prefetch 的正确性就比“单层速度”更重要，而这再次把问题推回到 host 侧控制能力。

<img src="assets/cpu-gpu-unified-memory.webp" alt="CPU-GPU 统一地址空间与一致性互连示意" width="760">

**图6** CPU-GPU 统一地址空间与一致性互连。高带宽一致性互连降低了 host memory 参与恢复路径时的摩擦，使 CPU 内存更适合作为 KV warm tier。来源：NVIDIA, 2025 [10]。

### 4.4 对机头 CPU 选型的直接影响

一旦 KV 成为长期状态对象，CPU 的需求就不再只看核心数，而要看：

- 主机内存带宽  
- 一致性互连  
- NUMA 摩擦  
- 大容量内存挂接能力  
- 恢复路径尾延迟  

这些指标之所以重要，是因为它们分别对应 warm-tier 命中后的恢复成本、近端/远端分层摩擦以及大容量状态对象的放置效率；换句话说，机头 CPU 的选型已不再只是“多少核心更划算”，而是“哪种内存层级和互连方式更适合承担状态系统” [10][15][29]。  

这也是为什么 KV 章节最终会落到 CPU 选型问题上。只要状态恢复已经进入关键路径，那么 CPU 的价值就体现在三个方面：能否足够快地判断状态该放在哪，能否足够稳地把状态按时搬回来，以及能否在多层之间切换时不把尾延迟放大。如果一颗 CPU 拥有更多核心却没有足够内存带宽或一致性互连，那么它未必比一颗“更像状态控制器”的 CPU 更适合 agentic inference。

小结：KV 卸载的重心已经从“把放不下的 KV 挪出去”转向“如何以低尾延迟维护一套分层状态系统”，而 CPU 正是这一系统的主执行者。

### 4.5 KV lifecycle 可以拆成四个 CPU 控制动作

把这一章进一步压缩，可以把 KV lifecycle 理解成四个连续动作：

1. `keep`：哪些状态继续留在热层。  
2. `demote`：哪些状态降到 host DRAM / CXL / storage。  
3. `prefetch`：哪些状态需要在恢复前被提前拉回。  
4. `resume`：恢复时如何把尾延迟控制在可接受范围内。  

这四步中，真正由 GPU 决定的内容很少，真正由 CPU 决定的内容很多。因此，KV 卸载不是显存容量问题的附属解法，而是一个标准的 host-side state machine 问题 [6][7][9][10][15][29]。

把这一点讲透之后，可以更容易理解为什么“KV 已从容量对象变成生命周期对象”不是修辞，而是系统边界的变化。过去讨论 KV，重点是总量和压缩比；现在讨论 KV，重点是对象热度、恢复概率、跨层迁移时机以及恢复失败对端到端延迟的影响。这些问题一旦成立，CPU 就不再是被动接收卸载流量的宿主，而是整个状态系统的管理者。

\newpage

## 五、机制层之三：MoE 编排与 host-side orchestration 的前移

在全文逻辑里，MoE 对 CPU 的挑战并不是一条平行支线，而是对前两章结论的进一步放大。算子下发说明 host 需要维持执行节奏，KV 生命周期说明 host 需要维持状态层级，而 MoE 则进一步表明：当参数访问本身也变成运行时变量时，CPU 还必须维持路由、驻留与搬运的一致性。

### 5.1 MoE 的系统代价不止是通信

MoE 的常见叙事是“总参数大、激活参数小，所以更高效”。但现有材料更清楚地表明：一旦 expert 总量超出单节点 GPU 驻留能力，系统问题会迅速浮现 [11][12][28][30]。  

主要压力至少来自三处：

- expert routing  
- weight residency / prefetch  
- collective orchestration  

因此，MoE 真正节省的是 GPU 上的密集计算，不是系统编排成本。  

这里真正需要解释清楚的是，为什么 MoE 会天然把压力推回 host。稠密模型里，大部分权重在推理开始前就已按固定方式驻留，系统主要关心 token 如何流过一条相对稳定的计算路径；而 MoE 把这条路径改成了运行时选择题。每批 token 不仅要决定逻辑上该访问哪些 expert，还要面对一个物理问题：这些 expert 是否已经在合适的设备、合适的内存层、合适的时间窗口内就位 [11][12][28]。

一旦“逻辑选择”和“物理位置”分离，CPU 的工作就会显著变重。因为 GPU 擅长执行已经确定的矩阵运算，却不擅长跨时间窗口管理“下一步可能需要哪些专家、这些专家该提前搬到哪里、哪些专家值得继续常驻”。这意味着 MoE 的稀疏性虽然降低了局部计算量，却把大量原本被静态隐藏的问题暴露成了运行时 orchestration 问题。也正因如此，MoE 的系统代价不能被简化成“通信多一点”，而应被理解成“更多关键决策必须在 host 侧持续做出”。

### 5.2 当前主流突破口：预测、预取与驻留解耦

Speculating Experts 的核心价值，是把 CPU→GPU 的权重搬运从同步阻塞路径挪到异步预测路径；文中报告 TPOT 可降低约 14% [11]。FluxMoE 则通过 decoupled residency，将逻辑 expert 身份与物理驻留位置分离，降低系统对“预测完全准确”的依赖 [12]。  

这些工作共同说明：MoE 推理的关键不只是“算得稀疏”，而是能否在正确的时间把正确的 expert 放到正确的位置。这个问题本质上是 host-side orchestration 问题。  

Speculating Experts 和 FluxMoE 分别击中了这个问题的两端。前者解决的是“能否更早知道下一步需要哪个 expert”，从而把同步搬运改成异步预取；后者解决的是“即便无法完全预测，能否把逻辑 expert 与物理驻留解耦”，从而降低系统对完美预测的依赖 [11][12]。两条路线表面上针对不同瓶颈，底层却在处理同一件事：不要让 token 到达某个 expert 时，系统才发现这个 expert 还没被放到正确位置。

这正是为什么 MoE 会持续抬高 host-side orchestration 的价值。因为只要 expert 是否在位仍然是运行时变量，就必须有人维护一条低抖动的 `route -> place -> move` 链路，而这个“有人”在当前系统中主要就是机头 CPU 及其上的调度器，而不是单个 GPU kernel。

<img src="assets/nvidia-wide-ep-moe-2025.webp" alt="Wide expert parallelism 在机架级系统中的组织方式" width="760">

**图7** 宽专家并行拓扑。该图反映出 MoE 的性能瓶颈已经扩展到 expert 路由、放置与跨设备通信，因此 host 侧的协调能力直接影响系统吞吐。来源：NVIDIA, 2025 [28]。

### 5.3 Agentic workload 下，MoE 压力会与 KV 压力叠加

在 agentic inference 中，同一个系统往往同时面对：

- 多上下文长期保留  
- 高频恢复  
- 子代理并发  
- 稀疏专家访问  

这意味着 KV 生命周期管理与 expert residency 很可能共享同一套 host 侧预算。也就是说，MoE 在 agentic inference 里的意义，不是单独增加一个问题，而是把原本已经存在的 host 压力继续叠加：一边是状态对象的保留与恢复，另一边是稀疏专家的放置与预取，两者都会竞争 host memory、传输窗口和调度队列 [9][11][12][28]。  

这里的关键不是“两个问题碰巧同时出现”，而是它们竞争的是同一组 host 资源。KV lifecycle 需要 host memory 带宽、恢复判断和跨层传输窗口；expert residency 同样需要 host memory、带宽预算和调度优先级。一旦 agentic inference 同时具有长生命周期状态和稀疏专家访问，CPU 就必须在两类高优先级对象之间反复取舍：现在更该优先拉回 KV，还是优先预取 expert；是先保证恢复路径低尾延迟，还是先保证下一批 token 的路由连续性。

也正因为如此，MoE 在 agentic workload 中尤其敏感。它并不是孤立地增加一类开销，而是把原本已经存在的状态管理问题推得更尖锐：只要 host 预算有限，KV 与 expert 的竞争就会直接表现为预取错误、尾延迟上升和 GPU 等待时间增加。

小结：MoE 不是“GPU 算得更少，系统自然更轻”，而是“GPU 变稀疏之后，host-side orchestration 更容易成为上限”。

<img src="assets/extracted/spec-experts-01.png" alt="Speculating Experts 论文中的专家预测与预取示意图" width="760">

**图8** MoE 专家预测与预取机制示意图。该类图说明 MoE 推理的关键优化正在把同步搬运改成预测驱动的异步预取，CPU 因而需要更稳定地承担路由、搬运与调度三重职责。来源：Speculating Experts 论文图页 [11]。

### 5.4 从 MoE 论文可以抽出的三层 host 任务

沿着 Speculating Experts、FluxMoE、wide expert parallelism 这条线，机头 CPU 在 MoE 中的任务可以被拆成三层：

1. `logical routing`：决定 token 应送往哪些 expert。  
2. `physical residency`：决定 expert 现在住在哪里，下一步该住在哪里。  
3. `transfer scheduling`：决定何时搬运、如何与计算重叠、如何避免 all-to-all 的同步放大。  

这三层任务都不是单个 GPU kernel 能独立完成的，因此 MoE 越成功，host orchestration 的价值越高 [11][12][28][30]。

如果把这三层任务串成一条因果链，会更容易理解 CPU 为什么变得关键：先有逻辑路由，系统才知道哪些 expert 可能会被访问；只有知道访问目标，系统才谈得上安排驻留和搬运；而搬运如果不能与计算重叠，就会反过来压垮路由带来的收益。因此，MoE 的系统性能最终不是由“路由器聪不聪明”单独决定的，而是由整条 `route -> place -> move` 链路是否足够稳定共同决定的。由于这条链主要由 host 侧维护，MoE 越被广泛采用，机头 CPU 的系统价值就越难被边缘化。

沿着这条逻辑继续向前，下一章需要回答的就不再是“机制上会不会这样”，而是“真实产品形态是否真的会把这些机制问题放大到足以影响部署”。

\newpage

## 六、场景层验证：真实工作负载对传统 serving 假设的修正

前三章分别从算子、状态和稀疏参数三个机制层面说明了 CPU 为什么会进入关键路径。本章的任务是把这些机制判断投射到真实产品形态上，解释为什么 agentic workload 会系统性放大控制路径、状态恢复和多会话协调压力。

### 6.1 真实系统不再符合“单上下文、长 decode、平滑批次”的理想模型

引入 Claude Code subagents、Kimi Agent Swarm、OpenClaw、Mobile Use Agent 的价值，不在于把产品介绍塞进综述，而在于它们帮助识别了真实 agentic workload 的几个稳定特征 [24][25][26][27]：

- 高频 prefill  
- 多会话并存  
- 极宽 fan-out/fan-in  
- 多模态 ingress  

这些特征并不直接告诉我们底层实现细节，但足以说明：真实系统已经偏离了传统 serving 优化默认采用的负载模型。  

这里真正重要的“为什么”是：一旦负载模型改变，瓶颈位置也会跟着改变。传统 serving 论文通常默认请求相对独立、上下文单调增长、decode 可以持续一段较长时间、batch 也会在相对稳定的区间内波动。只要这些前提成立，很多 host 侧控制动作就可以被摊平、隐藏或者提前安排。真实 agentic workload 恰恰打破了这些前提：状态会被暂停后恢复、会话会并发分叉、视觉或工具结果会重新进入模型、任务阶段会在 prefill、decode、pause、resume 之间频繁切换 [22][24][25][26][27]。

因此，真实 workload 的意义并不是“再给出几个复杂产品例子”，而是说明原本被认为可以忽略的 host 控制动作已经不再拥有被平摊的条件。只要阶段切换更频繁、状态对象更多、并发形状更宽，CPU 侧的排队、恢复和调度就会比传统 chat serving 更早进入关键路径。

<img src="assets/vllm-disagg-prefill-overview.jpg" alt="Prefill-decode 解耦视角下的请求阶段与状态流转" width="760">

**图9** Prefill-decode 解耦视角下的请求阶段与状态流转。该图本身来自 serving 系统视角，但与 subagents、swarm、mobile agent 等产品形态结合后，可以更直观看到为何真实 agentic workload 会放大高频 prefill、多会话并存和状态恢复压力。来源：vLLM 路线图与解耦 serving 资料 [5][22]；工作负载解释参考 [24][26][27]。

### 6.2 四类关键工作负载修正

#### 6.2.1 `prefill-first` 压力的重要性

GUI agent、mobile agent 和多轮代码 agent 会不断把新的观察、状态或中间结果重新送入模型。这意味着 host 侧不应只按“长 decode”设计，而必须对高频 prefill 更敏感 [22][24][27]。  

原因在于，prefill 与 decode 对系统的要求并不相同。Decode 更像带宽受限的连续流，很多控制动作可以被摊平到较长时间窗口里；而 prefill 更像一个短时集中 burst，需要在较短窗口内重新整理输入、更新会话状态、组织 batch、选择 worker 并决定状态应该放在哪一层。只要产品形态不断把新观察或新状态送回模型，系统就会不断重复这套“前端重整”过程，于是 host 侧比 GPU 更容易先被打满。

#### 6.2.2 `session multiplicity` 的协调复杂度

Claude Code subagents 和 Kimi Swarm 共同指向的是：问题不只是“一个上下文很长”，而是“同时活跃的上下文太多”。这会直接抬高 per-session queue、admission control 和 placement policy 的复杂度 [24][26]。  

之所以如此，是因为多会话并存会把原本单队列的问题变成多队列协调问题。系统不再只决定“当前 batch 怎么跑”，还要决定“哪个会话先跑、哪个会话稍后跑、哪些状态继续 pin 在近端、哪些状态可以暂时降级”。只要活跃上下文数量增长，这些决策就不再是局部调优，而会直接影响公平性、尾延迟和恢复成本。

#### 6.2.3 `fan-out/fan-in burst` 引发的瞬时宽并发

多代理系统的压力点并不总在平均 QPS，而往往出现在某一轮任务分解后的并发爆发，再在汇聚阶段形成二次瓶颈。这对 host 侧 burst handling 和尾延迟控制极为不友好 [26]。  

这里需要强调的是，fan-out/fan-in 的难点并不是长期平均吞吐，而是瞬时形状变化。系统在大部分时间里也许并不繁忙，但一旦某个任务阶段触发十几个甚至几十个子代理同时进入推理，host 侧就必须在极短时间内完成 admission、batch 重整、状态分发和恢复安排。更糟的是，汇聚阶段还会再次形成一次集中压力，因为多个分支结果需要被重新组织进后续推理路径。也就是说，平均吞吐无法描述真实压力，burst 宽度更接近 host 侧的实际边界。

#### 6.2.4 `multimodal ingress` 对 host 内存与排队的放大作用

即便不把手机操作或工具执行本身算进 CPU 主结论，视觉输入重新进入推理链路，也会放大 prefill、状态切换和会话绑定压力 [25][27]。  

其原因并不复杂：多模态输入比纯文本更难被视作“连续 decode 流”的自然延伸。每次视觉状态重入，系统往往都需要重新做输入准备、会话关联和状态更新，而这些动作大多发生在 host 侧而非 GPU kernel 内部。因此，多模态 ingress 的系统含义并不是“模型更强”，而是“更多需要被 CPU 重整后才能进入计算路径的输入出现了”。

小结：真实 agentic workload 让机头 CPU 的问题从“服务单条请求”演化为“管理异构、可恢复、可分叉的状态集合”。

### 6.3 把真实 workload 重新翻译成系统语言

这一章如果继续压缩，可以把几类产品形态翻译成更底层的系统约束：

- Claude Code subagents 对应的是 `session multiplicity`。  
- Kimi Swarm 对应的是 `fan-out / fan-in burst width`。  
- Mobile agent 与 OpenClaw 对应的是 `multimodal prefill churn`。  
- 多轮 agent 对应的是 `resume-heavy execution`。  

这种翻译的意义在于，它把“产品看起来更复杂”变成了“host 需要管理更多状态机”。因此，真实 workload 不是补充案例，而是用来修正系统模型的核心证据 [24][25][26][27]。

一旦把产品形态翻译成状态机问题，就更容易理解为什么这一章会与前面的算子下发、KV、MoE 三章收敛。因为无论外部产品看起来是代码代理、手机代理还是 agent swarm，真正压到系统内部的都是相似的 host 任务：更多阶段切换、更多状态对象、更多恢复路径以及更多调度分支。产品表面不同，但 host 侧的痛点越来越相似。

### 6.4 案例比较：四类 agentic 产品形态的 CPU 压力来源

为了避免“真实 workload”停留在抽象标签层，表 1 将 `Claude Code`、`Kimi Swarm`、`OpenClaw` 和 `Mobile Use Agent` 并排压缩到同一个比较框架中。这里的比较不讨论工具执行本身的 CPU 消耗，而只讨论这些产品形态对 `agentic LLM inference` 所施加的 host-side 约束。

| 案例 | 公开形态特征 | 对应的推理侧 CPU 约束 | 最容易被低估的点 | 主要来源 |
| --- | --- | --- | --- | --- |
| `Claude Code subagents` | 子代理独立上下文、按任务拆分、需要额外上下文收集 | `session multiplicity`、per-session queue、resume 与 placement | 不是单上下文太长，而是同时活跃 context 太多 | [24] |
| `Kimi Swarm` | 多 agent 并行、任务分解后再聚合 | `fan-out/fan-in burst`、admission control、聚合阶段调度 | 不是平均吞吐，而是瞬时宽并发和汇聚尾延迟 | [26] |
| `OpenClaw` | 常驻、多入口、移动/视觉交互、持续观察环境 | `multimodal prefill churn`、状态切换、会话绑定 | 视觉/界面状态频繁重入模型会拉高 prefill 频率 | [25] |
| `Mobile Use Agent` | 手机 GUI 执行、多模态交互、任务链式推进 | 高频 prefill、短 decode、resume-heavy 执行 | 即便不算操作执行，光是视觉输入重入就会放大 host 排队和状态管理 | [27] |

表 1 的价值不在于证明某个产品内部到底如何实现，而在于提供一套更稳的 workload 翻译方法：当产品公开形态显示它具有 `子代理并行`、`常驻观察`、`移动多模态` 或 `任务分解后聚合` 这些特征时，就可以合理预期机头 CPU 将更容易在 `session scheduling`、`prefill orchestration`、`resume latency` 和 `burst handling` 上成为上限 [22][24][25][26][27]。

进一步看，这四个案例共同说明，agentic inference 中最难的往往不是某一次前向传播，而是如何在连续多轮、并发分支和多模态重入之间维持稳定的控制节奏。对 GPU 来说，这些系统只是“更多请求”；对 CPU 来说，它们是“更多需要被协调的状态机”。因此，案例化对比并不是装饰性部分，而是把产品形态翻译成系统约束的必要步骤。

这一点也是平台章节需要承接的原因。既然不同工作负载都在把系统推向相似的 host 约束，那么平台路线图是否已经围绕这些约束发生变化，就成为检验前文判断是否足够稳健的下一步证据。

\newpage

## 七、平台层验证：平台演化与产品信号

前文的讨论主要来自软件系统、状态管理和工作负载形态。本章转向平台层，目的不是加入新的旁证，而是检验一个更强的判断：如果 CPU 作为控制平面的角色确实正在上升，那么平台组织方式理应开始主动为这一角色重新配置带宽、互连与数据面职责。

### 7.1 Vera / Rubin / BlueField-4 所传递的架构信号

NVIDIA Vera CPU 相关资料给出的关键信号包括：88 个定制 Arm 核、`1.2 TB/s` 的 LPDDR5X 带宽、`1.8 TB/s` 的 NVLink-C2C 带宽，以及“AI factory 控制平面”的明确定位 [13][14][17]。这些参数之所以重要，不只是因为它们高，而是因为它们恰好对应了 agentic inference 中最吃 host 预算的环节：

- 高带宽主机内存，对应 KV warm tier 与高频状态访问。  
- 高速 CPU-GPU 一致性互连，对应恢复路径和近端分层。  
- 与 GPU/DPU/交换的系统级耦合，对应跨池编排与传输协调。  

这里最需要补出的“为什么”是：平台厂商不会无缘无故把这些能力堆到同一颗 CPU 周围。之所以出现高主机内存带宽、强 CPU-GPU 近端互连和更紧的一体化平台耦合，是因为系统瓶颈已经不再只来自“CPU 算得不够快”，而来自“状态、传输和调度是否能按时到位”。只要推理系统越来越依赖 warm-tier 恢复、跨池传输和多阶段切换，那么主机内存带宽与近端一致性互连的重要性就会迅速上升 [10][13][22]。

换句话说，Vera 这类信号的真正含义不是“CPU 又成了主角”，而是平台开始承认一个新的现实：如果控制平面会反复进入推理关键路径，那么控制平面就不能再由一颗传统意义上的通用配套 CPU 顺带承担，而必须被当成值得单独优化的系统角色。

<img src="assets/nvidia-vera-cpu-architecture.png" alt="NVIDIA Vera CPU 架构与关键规格" width="760">

**图10** Vera CPU 架构与关键规格。高内存带宽和高 CPU-GPU 互连带宽说明其设计目标并非传统通用服务器 CPU，而是更接近 AI factory 控制平面处理器。来源：NVIDIA, 2026 [13]。

### 7.2 一体化平台说明“CPU 控制平面”假设已经下沉到系统设计

Rubin、NVLink Switch、BlueField-4 与网络交换的一体化平台表明，平台设计已经不再把 CPU、GPU、DPU 看成松耦合部件，而是按一个完整的推理系统来组织 [14][22][28]。  

这背后的因果链也需要讲透。只要推理过程越来越依赖“状态在多层之间流动”“请求在多个阶段之间流动”“数据在多个节点之间流动”，平台就不可能继续用松耦合方式拼装。因为松耦合意味着每一次状态迁移、每一次跨设备传输、每一次阶段切换都要支付额外摩擦成本。平台一体化的本质，就是在减少这些摩擦，让 CPU、GPU、DPU 和交换结构更像一台系统，而不是几类部件的简单拼接 [14][22]。

因此，图11 不只是一个更复杂的机柜示意图，而是平台厂商对控制路径问题的回应：既然控制路径已经足够重要到会限制 GPU 利用率，那么就必须把网络、存储、安全和计算组织成一条更低抖动的整体链路。

<img src="assets/nvidia-vera-rubin-6chips.png" alt="Vera Rubin 六芯片协同平台" width="760">

**图11** Vera、Rubin、BlueField-4 与交换组件的协同平台。该图说明机头 CPU 的位置已前移为整个平台的控制平面，而非传统意义上的外围 host。来源：StorageReview, 2026 [14]。

### 7.3 CPU:GPU 配比变化是系统需求变化的外在表现

TrendForce 以及多份产业评论都判断，AI 数据中心的 CPU:GPU 配比正在从传统的 `1:4–1:8` 向 `1:1–1:2` 演进 [16][18][19][20][21]。即便这些数字需要继续验证，它们至少说明产业已经在按“更多 CPU 预算是合理的”来规划未来部署。  

这里不能把配比变化简单理解成“多配点 CPU 以防万一”。更准确的解释是，当 GPU 单位算力继续快速上升，而 host 侧承担的状态管理、调度与恢复任务没有同步消失时，系统就会自然出现“GPU 进步快于控制平面进步”的失衡。配比往 CPU 侧回摆，反映的正是这种失衡：不是 GPU 不重要了，而是如果控制平面预算跟不上，GPU 的增量价值就无法完整兑现 [16][18][19]。

因此，CPU:GPU 配比变化更像一种系统补偿机制。只要 agentic inference 持续强化多阶段推理、状态保留和多代理并发，部署者就会发现仅靠增加 GPU 数量并不能线性换来端到端效率提升，于是预算开始重新分配到 CPU、内存层级和旁路数据面能力上。

小结：平台路线图并非对论文结论的简单附会，而是在硬件组织方式上确认了 host orchestration 的价值正在上升。

### 7.4 平台层的真正变化不是“CPU 更强”，而是“CPU 更靠前”

把 Vera、Rubin、BlueField-4、CXL 这些信号放在一起，最值得注意的不是单个参数，而是职责位置的变化：

- CPU 更靠近 memory tiering。  
- CPU 更靠近 GPU 协调。  
- DPU / SuperNIC 更主动替 CPU 卸下数据面任务。  
- 平台开始默认控制平面值得被单独设计。  

这说明 host CPU 的问题已经不是“传统服务器 CPU 能不能顺便做一点推理工作”，而是“AI inference control plane 应该长什么样”。这也是为什么平台讨论必须成为综述正文的一部分，而不能只放在附录里 [13][14][15][16][17][22][28]。

从这个意义上说，平台演化并不是对前文结论的附会，而是在更上游层面给出验证：如果硬件路线图开始围绕 memory tiering、近端一致性互连、DPU 旁路和平台级耦合重新组织，就说明 host orchestration 的重要性已经从研究判断转变为平台设计需求。

在这一点上，平台结论自然会导向部署问题。既然系统级硬件已经开始围绕 host orchestration 调整，那么工程实现就不应继续沿用统一节点规格和单层状态放置的旧思路。

\newpage

## 八、工程落地：选型与部署

本章并不引入新的基础判断，而是把前文得到的机制结论转写为工程语言。这样安排的目的，是避免部署建议看起来像经验清单：每一条部署原则都应能够回溯到前文关于控制路径、状态分层、MoE 编排和工作负载修正的具体机制。

### 8.1 机头 CPU 不应只按“品牌/代际”选，而应按节点角色选

从前文四条主线可以看出，同样是“机头 CPU”，不同部署角色的关注点其实不同：

1. `co-located GPU node`：更看重近端一致性互连、主机内存带宽与恢复路径延迟。  
2. `prefill-heavy ingress node`：更看重高频 prefill 下的调度稳定性、tokenization 与输入处理能力。  
3. `capacity-oriented state node`：更看重大容量 host memory、CXL 挂接与分层成本。  
4. `coordination-heavy swarm node`：更看重多会话调度、burst handling 与尾延迟隔离。  

这也是为什么本文一直强调，agentic inference 中的 CPU 选型不应再被理解为“给 GPU 配一颗合适的 host CPU”，而应被理解为“给不同节点角色配置不同控制平面预算” [10][15][16][22][23]。

这里的“为什么”在于，不同节点进入关键路径的方式并不相同。某些节点主要负责近端恢复和与 GPU 协同，因此更怕内存带宽不足和一致性互连摩擦；某些节点主要负责高频 prefill，因此更怕 burst 下调度不稳；还有一些节点主要承担状态分层和远端恢复，因此更怕容量不足和跨层迁移代价过高。既然瓶颈出现的位置不同，CPU 选型自然不能再按“统一主机规格”来做。

换句话说，角色分层不是管理上的便利，而是系统边界变化后的必然结果。前文已经说明，agentic inference 的问题不是单一算力瓶颈，而是控制路径、状态路径和传输路径的耦合瓶颈；一旦如此，不同节点承担的耦合方式不同，硬件配置也就必须跟着分化。

### 8.2 部署分层框架

如果把现有材料压成工程语言，可以得到一个相对稳健的部署分层框架：

- `热层`：GPU HBM，承载当前 step 必须参与计算的状态与权重。  
- `温层`：coherent CPU memory / host DRAM，承载高概率恢复对象、近端 overflow 与快速复用状态。  
- `暖层扩展`：CXL memory，承载容量优先但仍要求可控恢复延迟的对象。  
- `冷层`：local / remote storage，承载低热度状态与更远期恢复对象。  

<img src="assets/agentic-kv-memory-hierarchy.svg" alt="Agentic inference 的 KV / 状态分层内存示意图" width="760">

**图12** Agentic inference 的状态分层内存示意图。该图将 HBM、CPU memory、CXL 和更远层存储放在同一个层次视图里，更适合用来解释部署侧的 tiering 决策。来源：本目录整理图，依据 KV offload、coherent memory 与 CXL 资料 [9][10][15][29]。

这一分层最直接的意义，是把“CPU 配置”从单一算力参数改写成一组系统参数：

- host memory 带宽够不够支撑 warm-tier 命中  
- CXL 扩展是否真的降低总成本  
- 恢复路径是否会把尾延迟推高到不可接受  
- 不同层之间的 demote / prefetch / resume 是否能被稳定编排  

之所以需要这样一套分层框架，是因为状态对象的热度分布天然不均匀。总会有一小部分状态需要在极短时间内被恢复，另一部分状态虽然未来还会再访问但不值得长期占据最热层，还有更大一部分状态只需要被低成本保留以备未来使用。如果没有清晰的层级划分，系统就会在两个坏结果之间摇摆：要么把太多状态留在近端，浪费最贵资源；要么把太多状态过早放远，导致恢复路径尾延迟失控 [9][10][15][29]。

因此，部署分层框架不是“把现有硬件名词排个层次”而已，而是在回答一个更根本的问题：系统如何让不同热度、不同恢复时效要求的状态对象，占据不同成本和不同延迟的资源层。只要这个问题存在，CPU 就不再只是承载层，而是层级决策与迁移执行的中心。

### 8.3 DPU / SuperNIC 的意义在于给 CPU 让路

BlueField-4 这类组件在本文中的意义，不是单独讨论 NIC 或 DPU，而是说明：当网络、存储和安全路径可以被旁路时，机头 CPU 就能把更多预算留给推理控制面 [14][22]。

为什么“给 CPU 让路”会变得重要？因为在 agentic inference 中，CPU 最稀缺的往往不是总周期数，而是低抖动的关键路径预算。只要网络协议处理、存储协议栈、安全与隔离逻辑继续与推理控制动作争抢同一批 host 资源，那么哪怕 CPU 总体利用率看起来不高，也可能在关键时刻拖慢状态恢复、batch 组织或传输提交。DPU / SuperNIC 的真正价值就在于把这些相对可旁路的数据面职责剥离出去，使 CPU 能更稳定地服务推理控制链路。

从工程角度看，这意味着“是否需要 DPU”也不再只是网络规模问题，而开始与推理控制平面压力直接相关：当系统越来越依赖跨池传输、远端恢复和多租户隔离时，旁路数据面就等于间接提升了 CPU 用于 orchestration 的可用预算。

<img src="assets/nvidia-bluefield4.png" alt="BlueField-4 DPU 与网络卸载架构" width="760">

**图13** BlueField-4 DPU 与网络/存储卸载架构。该图的关键不是 DPU 自身，而是它说明平台开始主动把数据面职责从 CPU 上拿走，从而给推理控制链路让出预算。来源：StorageReview / NVIDIA 平台资料 [14][22]。

### 8.4 面向 agentic inference 的 CPU 选型要点

对于真正的工程部署，CPU 选型至少应回答以下问题：

- 是否主要面对 `prefill-heavy`、`resume-heavy` 还是 `decode-heavy` 负载？  
- 是否需要把 host memory 当作 warm tier，而不是仅仅当作 spill 层？  
- 是否存在大量 `session multiplicity` 或 `fan-out/fan-in burst`？  
- 是否需要与 CXL、DPU、远端存储组成多层状态系统？  
- 是否要求严格控制恢复路径和跨池传输的尾延迟？  

如果这些问题里大多数回答都是“是”，那么该节点对 CPU 的需求就不应再按传统 serving 假设来配置，而应按 orchestration node 来配置 [2][9][10][15][22][23]。

这份检查表背后的逻辑，是把前文结论重新投影到工程选择上：你的节点到底更像“执行密集型节点”，还是“协调密集型节点”。如果答案偏向后者，那么 CPU 的价值主要就体现在低抖动控制、状态层级管理和恢复路径保障，而不是传统意义上的通用服务器吞吐。部署建议因此不是经验清单，而是由前文机制链直接推导出的工程约束。

如果说部署章节回答的是“应该如何配”，那么下一章则进一步回答“应该如何测”。这一步是必要的，因为如果评测框架仍停留在传统 decode-centric 指标上，前文的很多判断就难以被稳定验证。

\newpage

## 九、评测与研究议程

到这里，全文已经从机制、工作负载、平台和部署四个层面说明了 CPU 角色变化的原因。本章的任务是补上验证闭环：如果缺少合适的 benchmark 和研究设计，那么前文许多判断即使方向正确，也很难被系统比较、复现实验或用于稳定选型。

### 9.1 现有 benchmark 的局限

当前公开材料已经足以说明 CPU 重要，但仍缺一个被普遍接受的 `agentic inference host benchmark`。原因并不复杂：传统 benchmark 大多只覆盖吞吐、TPOT、TTFT 或显存占用，却很少把以下因素放进同一个测量框架：

- 高频 prefill  
- session multiplicity  
- fan-out / fan-in burst  
- KV lifecycle efficiency  
- MoE residency / prefetch quality  
- multimodal ingress sensitivity  

因此，很多系统虽然在单请求或稳定 batch 上表现良好，却未必能在真实 agentic workload 下保持低抖动 [2][9][24][26][27]。

这里最重要的“为什么”是：benchmark 其实在隐含定义“什么算性能”。如果一个 benchmark 只测单请求吞吐、稳定 batch 下的 TTFT，或者只测长 decode 流中的 TPOT，那么它实际上默认系统的难点在于“持续推进一条相对平滑的计算路径”。但前文已经反复说明，agentic inference 的难点往往并不发生在这种平滑阶段，而发生在状态切换、恢复、并发分叉和多模态重入时刻 [2][9][22][24][26]。

也就是说，现有 benchmark 不够，并不是因为它们测得不准，而是因为它们测的是另一类系统假设。只要 workload 的真实形状更像一组需要不断协调的状态机，而不是一条单调的 decode 流，那么只测吞吐和平均延迟就会系统性低估 CPU 侧控制链路的重要性。结果就是：一些在传统 benchmark 上看起来表现优异的系统，到了真实 agentic workload 下却可能因为恢复路径抖动或 burst 调度失控而大幅退化。

### 9.2 面向 host 的 benchmark 应至少覆盖五类指标

如果要设计一套面向机头 CPU 的 benchmark，本文认为至少要覆盖五类指标：

1. `dispatch latency`：从请求进入到 GPU 有效开算之间的 host 延迟。  
2. `resume latency`：状态对象从 warm/cold tier 恢复回计算路径的延迟。  
3. `session concurrency quality`：多会话并存时的公平性与尾延迟。  
4. `burst robustness`：fan-out / fan-in 宽并发下的退化曲线。  
5. `tiering efficiency`：KV / expert / context 在多层内存中的放置与迁移效率。  

这些指标之所以重要，是因为它们分别对应本文四条主线的实际工程落点：算子下发、KV 生命周期、MoE orchestration、真实 workload 修正。

如果把这五类指标再往下拆，会更清楚它们为什么必须同时出现。`dispatch latency` 对应的是控制动作本身是否足够轻；`resume latency` 对应的是状态系统是否真的可恢复；`session concurrency quality` 对应的是多会话并存时 host 是否还能维持公平与稳定；`burst robustness` 对应的是系统在瞬时形状变化下是否会崩；`tiering efficiency` 则回答多层状态系统究竟有没有真的创造收益。缺掉任何一类，benchmark 都会重新退化成只测“静态好不好”，而无法回答“动态稳不稳”。

换句话说，这五类指标并不是随意罗列出来的清单，而是由本文前面几章的机制链直接推导出来的。只要你承认瓶颈来自控制路径、状态生命周期、稀疏专家编排和真实 workload 形状，那么 benchmark 就必须覆盖这五个面向，否则它测出来的“性能”就不是 agentic inference 真正在意的性能。

### 9.3 现有研究还有三条明显断层

即便材料已经相当丰富，仍有三条断层值得单独指出：

- `产品到机制的断层`：真实产品公开的内部指标仍太少，导致很多 workload 结论只能做形态推断。  
- `平台到软件的断层`：Vera / Rubin / BlueField-4 给出了清晰方向，但软件栈与实际部署经验还不够丰富。  
- `局部优化到系统目标的断层`：许多论文能优化某一段链路，却缺少覆盖整条 control path 的统一评估框架。  

这三条断层之所以严重，是因为它们分别切断了“从证据到结论”的关键路径。产品到机制的断层，使我们很难判断某个公开产品形态究竟对应多大的 host 压力；平台到软件的断层，使硬件路线图给出的信号还无法完全转化成稳定的软件收益；局部优化到系统目标的断层，则意味着很多论文虽然把某一个点做得更快，却不能证明整条控制链路因此真的更稳 [2][9][13][14][22]。

也就是说，当前研究的主要问题并不只是“资料还不够多”，而是不同证据层之间还没有被打通。只要这些断层存在，我们就很容易在两个极端之间摇摆：要么只相信底层论文，低估真实 workload 的复杂度；要么只看产品和平台信号，却无法证明底层机制真的支撑这些判断。补齐这些断层，本身就是后续研究议程的一部分。

### 9.4 后续研究议程

如果把本文归纳出的空白继续往前推，可以得到一个相对清晰的研究议程：

- 建立 `agentic inference host benchmark`，而不是继续只用传统 serving 指标。  
- 研究 `state-centric scheduling`，把 KV / context / expert 一起视作长期状态系统。  
- 研究 `control-plane-aware serving`，把 host orchestration 视作第一等优化目标。  
- 研究 `platform-software co-design`，让 CPU、GPU、DPU、CXL 和 transport 栈在同一框架下评估。  

这几条议程之所以重要，是因为 agentic inference 的核心难题已经不再只是“如何让单次前向传播更快”，而是“如何让一个长期、可恢复、可分叉的推理系统稳定运行” [2][9][11][15][22][28]。

更进一步说，这一研究议程的价值，在于它把“如何优化一个模型”改写成“如何优化一个推理系统”。前者关注局部 kernel、带宽和算子，后者则必须同时考虑状态流动、控制路径抖动以及平台与软件如何共同组织预算。因此，本章不是附带讨论，而是在为前文所有结论提供验证闭环。

\newpage

## 十、附录：关键数据与论据总表

本表不承担新的论证任务，仅用于将正文中分散出现的关键数值集中列出，便于交叉查阅。正文负责建立因果关系，本表负责提高检索效率。

| 主题 | 关键数据点 | 含义 | 主要来源 |
| --- | --- | --- | --- |
| 算子下发 | 301 次 kernel launch、约 750 微秒纯下发税、fusion 后吞吐约提升 20% | 小模型与量化场景下，host 固定成本会迅速成为上限 | [3] |
| CPU 抖动 | dequeue 延迟从 12 ms 恶化到 228 ms，约 19x | CPU oversubscription 可放大为集群级停顿 | [2] |
| 推理引擎优化 | vLLM V1 吞吐相对 V0 最高约 `1.7x` | 大量收益来自减少 CPU 侧调度和 Python 开销 | [5] |
| KV 访问模式 | KV read/write ratio `11.7x` | agentic inference 更像状态恢复问题，而非单纯写入问题 | [9] |
| KV 卸载 | NOSA 最高 `5.04x` 解码吞吐提升 | 减少 CPU-GPU KV 传输量是核心收益来源 | [6] |
| 协同式卸载 | ScoutAttention 约 `2.1x` 加速，精度损失 < 2.4% | CPU 可以从搬运者变成协同计算者 | [7] |
| CXL 分层 | GPU 需求下降 87%，GPU 利用率提升 75%，CPU 利用率下降 40% | KV 分层已经进入容量-性能-成本联合优化阶段 | [15] |
| MoE 预取 | TPOT 下降约 14% | expert 预取能把同步搬运改成异步路径 | [11] |
| 平台规格 | Vera 内存带宽 `1.2 TB/s`，NVLink-C2C `1.8 TB/s` | 平台已按 CPU 控制平面需求组织 | [13] |
| 产业配比 | CPU:GPU 由 `1:4–1:8` 向 `1:1–1:2` 演进 | 部署预算开始显式向 host 侧回流 | [16] |

\newpage

## 十一、讨论：证据优势与未决问题

经过前文的逐层展开，全文的结构可以被重新理解为三段：第二章提出中心命题，第三至第九章依次提供机制、工作负载、平台与部署证据，第十章给出可检索的数据附录。本章的任务因此不是重复结论，而是明确这组证据为何足以支撑当前判断，以及它们的有效边界在哪里。

### 11.1 当前证据的稳健性来源

这批材料最强的地方，是四条证据链能够相互印证：

- 机制论文解释了为什么 host 侧会成为瓶颈 [2][6][7][11][12]。  
- 厂商系统文档解释了这些瓶颈在生产栈中如何出现 [9][10][22][23][28]。  
- 产品 workload 解释了为什么传统 serving 假设会失效 [24][25][26][27]。  
- 平台路线图解释了为什么硬件开始围绕这一判断收敛 [13][14][15][16][17]。  

这也是为什么本文的主要结论有一定稳健性。单看其中任何一条证据链，都仍可能被质疑为“只覆盖了某一类场景”；但当机制论文、厂商系统文档、产品 workload 和平台路线图开始在同一个方向上汇聚时，结论的性质就从“某类材料的局部观察”提升为“跨层次的系统判断”。换句话说，本文最强的地方并不是引用多，而是这些引用之间能够相互解释，而不是彼此孤立。

### 11.2 仍待补足的关键证据

但仍有三类缺口没有真正闭合：

1. 缺少统一的 `agentic inference host benchmark`。  
2. 真实产品形态与底层机制之间，仍存在一层推断。  
3. Vera / Rubin / CXL 等平台信号很强，但长期普及度和跨平台可迁移性仍需更多独立部署证据。  

这三类缺口之所以必须被保留下来，是因为它们决定了本文结论的边界。我们可以较有把握地说“机头 CPU 已进入关键路径”，但还不能用同样强度去断言“某一种 CPU 架构已经 definitively 最优”，也不能仅凭产品形态就精确推回其内部 serving 参数。也就是说，讨论章节的作用不是削弱前文，而是在告诉读者：哪些判断已经足够稳，哪些判断仍然应被视为有条件成立。

\newpage

## 十二、结论

将本文归纳为一句话，较为准确的表述不是“CPU 又重要了”，而是：

**agentic AI 推理正在把原本隐藏在 host 侧的调度、状态、传输和编排链路重新暴露出来，而机头 CPU 就是这条链路的第一执行者。**

这句话之所以能作为全文总结，不是因为它足够醒目，而是因为前文各章实际上都在证明同一件事：算子下发说明控制路径被放大了，KV 生命周期说明状态管理被放大了，MoE 说明运行时放置问题被放大了，真实 workload 说明这些问题不能再被平均吞吐掩盖，平台与部署章节则说明这种变化已经开始反映到硬件组织方式和工程决策里。换句话说，这个结论并不是一个抽象口号，而是各章机制链共同收束后的最简表达。

因此，机头 CPU 在 agentic inference 中的角色不应再被理解为“GPU 旁边那颗普通服务器 CPU”，而应被理解为：

**推理系统的 orchestration layer in silicon。**

把机头 CPU 理解成 `orchestration layer in silicon` 的价值，在于它改变了我们看待系统优化的方式。过去我们更容易把 host 视为“GPU 外围的必要宿主”；现在更准确的理解是：它负责把请求、状态、层级内存、跨池传输和多代理执行组织成一条低抖动的控制链。如果这条链不稳定，GPU 侧的优化就很难完整兑现；如果这条链足够稳定，很多原本看起来互相冲突的目标，例如长上下文、低尾延迟和宽并发，也才有可能被同时逼近。

对未来系统设计而言，这一判断至少带来三点直接启示：

1. 不能再只按 CPU 核心数或主频做选型，而要按节点角色、内存分层与恢复路径来设计 [10][15][22]。  
2. 不能再把 KV 看作显存扩展问题，而要把它当作长期状态系统来处理 [6][7][9][10]。  
3. 不能再把 host 视为背景常量，而应把 host orchestration 纳入一等公民级别的建模、测量与优化 [2][9][11][22][28]。  

这三点启示之所以值得被单独保留下来，是因为它们分别对应设计、实现和评估三个层面。第一点要求工程上重新定义选型对象；第二点要求系统上重新定义状态对象；第三点则要求研究上重新定义性能度量。只有这三层同时变化，机头 CPU 在 agentic inference 中的新位置才不会停留在综述判断，而能真正落到平台、软件和 benchmark 的共同演进上。

\newpage

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
