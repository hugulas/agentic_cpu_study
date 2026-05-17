# Agent Team / Swarm 对 AI 机头的影响：深度洞察报告

> **研究状态**: audited（已审计）  
> **日期**: 2026-05-17  
> **核心发现**: Agent Team/Swarm 将 AI 机头 CPU 的负载从"单会话调度器"推向"多租户并发编排 + 跨 Agent KV 共享协调 + 动态预取决策"的三维控制平面。Kimi Swarm 的 100 并行 sub-agents、Claude Code Agent Teams 的 7× token 消耗、以及 Hive 揭示的 >70% token 集中在少数核心 Agent——三者共同证明：AI 机头的瓶颈不再是单条上下文的长度，而是"同时管理多少条独立上下文"和"如何让它们共享状态而不重复计算"。

---

## 目录

1. [摘要](#1-摘要)
2. [Agent Swarm/Team 的工作负载特征](#2-agent-swarmteam-的工作负载特征)
3. [对 KV Cache 访问模式的颠覆](#3-对-kv-cache-访问模式的颠覆)
4. [对 AI 机头 CPU 的多维影响](#4-对-ai-机头-cpu-的多维影响)
5. [系统级优化方向：从单轮到多 Agent](#5-系统级优化方向从单轮到多-agent)
6. [综合洞察：AI 机头的新边界](#6-综合洞察ai-机头的新边界)
7. [关键数据速查表](#7-关键数据速查表)

---

## 1. 摘要

本研究系统分析了 Agent Team / Swarm（多智能体协作/群体）工作负载对 AI 机头（推理前端/Host CPU）的深远影响。基于 8 份一手论文/报告的深度阅读——包括 Hive（arXiv:2604.17353）、RelayCaching（arXiv:2603.13289）、AMPD（arXiv:2602.14516）、KAIROS（arXiv:2604.16682）、Dive into Claude Code（arXiv:2604.14228）、Kimi Agent Swarm 官方博客、AMD Agentic AI 技术博客、PolyKV（arXiv:2604.24971）——我们提炼出以下核心洞察：

**Agent Swarm 的工作负载特征与传统单轮对话存在本质差异**：
- **极端并发宽度**：Kimi Swarm 支持 **100 并行 sub-agents**，单次任务触发 **>1,500 次工具调用**
- **会话多重性（Session Multiplicity）**：Claude Code 的 Agent Teams 消耗 **7×** 标准会话的 token，每个 sub-agent 拥有独立上下文窗口
- **Agent 异质性**：Hive 对 R3A 多 Agent 系统的剖析显示，**>70%** 的 token 消耗和调用频率集中在少数核心 Agent（Decision、Patcher、Viewer）
- **长程状态化**：KAIROS 测得平均并发 **17 个 Agent**，平均 **37 轮**交互，最长可达 **2,518 轮**

**对 KV Cache 访问模式的颠覆**：
- **从 O(N) 到 O(M²)**：多 Agent 流水线中，每个下游 Agent 必须重新预填充所有上游累积历史，累积成本呈二次方增长
- **从独享到共享**：RelayCaching 证明，上游 Agent 的 decoding KV cache 可直接重用于下游 prefill，重用率 **>80%**，TTFT 降低 **up to 4.7×**
- **从请求级到 Agent 级**：Hive 的 Agent-Aware Scheduling 将 KV Cache 驱逐从"请求平等"转向"按 Agent 贡献优先级驻留"，热点 miss rate 降低 **33%–51%**

**对 AI 机头 CPU 的多维影响**：
- **调度维度**：从单会话 batching 到 100 路并发的瞬时 burst 调度，CPU dequeue 延迟可被放大 **19×**
- **KV 管理维度**：从"单条长上下文"到"同时管理 100 条独立上下文 + 跨 Agent 共享"，CPU 成为 KV placement 和 eviction 的决策中枢
- **网络编排维度**：AMPD 证明 **13.9%–31.7%** 的增量 prefill 应留在 decode worker 本地执行，避免跨节点 KV 传输——这一决策必须由 CPU 侧的负载感知调度器实时做出
- **功耗维度**：KAIROS 发现 Agentic inference 功耗比单轮高 **2–3 个数量级**，CPU 侧的并发控制和频率调节成为能效管理的关键

**关键结论**：AI 机头 CPU 的角色正从"GPU 附属调度器"进化为"多 Agent 编排控制平面"。AMD 官方预测 CPU:GPU 配比从 1:4–1:8 转向 **1:1**，某些场景 CPU 侧负载甚至更高。Claude Code 的架构分析更揭示了一个惊人比例：**AI 决策逻辑仅占 1.6%，运营基础设施占 98.4%**——这意味着 AI 机头的优化空间不在模型本身，而在 CPU 侧的编排、缓存、压缩和恢复逻辑。

---

## 2. Agent Swarm/Team 的工作负载特征

### 2.1 三种产品形态与负载模式

当前主流的 Agent Team/Swarm 产品可分为三种形态，每种对 AI 机头产生不同的负载特征：

**形态 1：Kimi 式 Swarm（大规模并行发现/输出）**
- **并发宽度**：up to **100 sub-agents** 并行（K2.5），K2.6 扩展至 **300 sub-agents / 4,000 协调步骤**
- **工具调用密度**：单次任务 **>1,500 次工具调用**
- **加速效果**：相比串行执行快 **4.5×**
- **对机头的负载**：瞬时 prefill burst（100 条独立上下文同时进入 prefill）、大规模工具调用结果的并发返回处理、动态并行宽度控制

**形态 2：Claude Code 式 Subagent Teams（分层委派）**
- **并发宽度**：通常 3–10 个 sub-agents 同时活跃
- **上下文隔离**：每个 sub-agent 拥有 **独立上下文窗口**（200K–1M tokens），仅返回 summary（500–2,000 tokens）给父代理
- **Token 消耗**：Agent Teams 模式消耗约 **7×** 标准会话的 token
- **对机头的负载**：上下文隔离带来的多份独立 KV Cache 管理、sub-agent 生命周期（创建/销毁/恢复）、五层 context compaction pipeline 的持续执行

**形态 3：Hive/R3A 式多 Agent 工作流（角色专业化协作）**
- **并发宽度**：5–20 个角色化 Agent（Decision、Patcher、Viewer、Summary、Searcher）
- **异质性**：>70% 的 token 消耗集中在少数核心 Agent
- **对机头的负载**：Agent-Aware Scheduling 需要在线计算每个 Agent 的贡献分（Shapley 风格近似）、不同角色的 KV Cache 占用和复用模式差异巨大

### 2.2 与传统 Chat/单 Agent 的本质差异

| 维度 | 传统 Chat | 单 Agent | Agent Team/Swarm |
|------|----------|---------|-----------------|
| **并发会话数** | 1 | 1 | 10–300 |
| **上下文模式** | 单条线性增长 | 单条 + 工具插入 | 多条独立 + 级联依赖 |
| **KV Cache 管理** | 一条 LRU | 一条 + prefix cache | 多条 + 跨 Agent 共享 |
| **调度对象** | 单个请求 | 单个请求 + 工具间隙 | 多个 Agent + 父子依赖 |
| **瓶颈类型** | 单条长度 | 工具调用延迟 | **并发宽度 + 状态协调** |
| **CPU 负载峰值** | 低 | 中 | **极高（burst）** |

### 2.3 KAIROS 揭示的 Agentic 负载量化特征

KAIROS 论文（arXiv:2604.16682）基于 H100 实测，给出了 Agentic inference 的精确负载画像：

- **并发 Agent 数**：平均 **17 个**（steady state），请求率 0.08 agents/s
- **对话轮次**：平均 **37 轮**，范围 1–**2,518 轮**
- **功耗**：比传统单轮 LLM serving 高 **2–3 个数量级**
- **Thrashing 阈值**：GPU 频率低于 **900 MHz** 时，上下文累积速度超过释放速度，系统进入 thrashing 状态

---

## 3. 对 KV Cache 访问模式的颠覆

### 3.1 从 O(N) 到 O(M²)：级联冗余 prefill 问题

在多 Agent 流水线中，上游 Agent 的输出成为下游 Agent 的输入。标准 serving 系统必须从头重新计算 KV cache，导致累积预填充成本随交互轮次呈**二次方增长**：

```
总 prefill 成本 = Σ(i=1 to M) i·L = O(M²)
```

RelayCaching 论文（arXiv:2603.13289）量化了这一瓶颈：
- 2-Agent 流水线：Full Prefill TTFT = 85.2 ms
- 5-Agent 流水线：Full Prefill TTFT = **493.9 ms**（5.8× 增长）
- 三 Agent 系统长上下文（512–12,288 tokens）：Full Prefill 达 **3.89 s**

### 3.2 RelayCaching：>80% KV Cache 直接重用

RelayCaching 的核心突破是**将上游 Agent decoding 阶段的 KV cache 直接重用于下游 Agent 的 prefill 阶段**，通过选择性修正（selective rectification）处理 prefix variation：

- **重用率**：大多数多 Agent 设置中超过 **80%**
- **TTFT 加速**：Agent 2 → 2.10×，Agent 5 → **4.71×**
- **长上下文加速**：vs Full Prefill **9.2×**，vs CacheBlend **2.5×**
- **准确率保持**：GSM8K 84.84–85.50%（与 Full Prefill 相当）
- **成本扩展**：从 O(M²) 降至**次线性**

**对 AI 机头的启示**：CPU 需要维护一个**跨 Agent 的 KV Cache 引用图**，追踪哪些 KV 块可被下游重用、哪些需要修正、以及修正的范围（layer range + token set）。这比传统 prefix cache 的字符串匹配复杂得多。

### 3.3 PolyKV：从 O(N) 到 O(1) 的内存压缩

当 N 个 Agent 处理相同文档上下文时，PolyKV 提出只计算一次 compressed KV state：
- **内存复杂度**：O(N) → **O(1)**
- 每个 Agent 独立注入解压后的 KV tensor

**对 AI 机头的启示**：CPU 需要维护**共享 KV Pool**，管理压缩/解压的生命周期，并处理并发 Agent 对同一 KV 块的读写隔离。

### 3.4 Hive Agent-Aware Scheduling：从请求平等到 Agent 优先级

Hive 揭示了一个被忽视的问题：在多 Agent 系统中，>70% 的有效工作由少数核心 Agent 完成。传统 LRU 驱逐可能因"最近访问"而误驱逐核心 Agent 的高复用 KV 状态。

**Agent-Aware Scheduling**：
- 为每个 Agent 计算综合贡献分 `Score(a) = 0.4·Ūa + 0.6·C̄a`（内在效用 + 协作效用）
- KV Cache 驱逐时**优先保留高贡献 Agent 的状态**
- **效果**：hotspot miss rate 降低 **33%–51%**，被驱逐 KV token 总量减少 **19.2%–30.2%**

**对 AI 机头的启示**：CPU 需要运行**轻量级的 Agent 贡献度分析**（滑动窗口统计、Shapley 风格近似），并将优先级指令实时下发到 GPU 侧的 KV Cache 管理器。

### 3.5 Hive Logits Cache：消除跨路径冗余

在 Tree-of-Thoughts 等多分支 TTS 算法中，不同分支从同一中间状态重采样时会产生大量重叠输出 token。

**Logits Cache**：
- 将 decoding 过程中的中间 logits 序列缓存到**主存（CPU DRAM）**
- 通过重放采样（replay sampling）复用，避免重新执行模型前向传播
- Hotspot Sampling 平均加速 **1.76×**，Logits Cache 命中率 **30.4%**

**对 AI 机头的启示**：CPU 主存成为**算法级缓存层**，存储的不再是原始 KV 而是 logits 分布。这进一步推高了对 CPU 内存带宽的需求。

---

## 4. 对 AI 机头 CPU 的多维影响

### 4.1 调度维度：从单会话到 100 路并发的 Burst 调度

**核心变化**：
- Kimi Swarm 的 100 sub-agents 会在**极短时间内同时进入 prefill 或 decode**
- 传统 batching 假设请求均匀到达，但 swarm 的 fan-out 产生**阶跃式负载激增**
- CPU oversubscription 可使 dequeue 延迟放大 **19×**（CPU-Induced Slowdowns 论文）

**机头 CPU 的新职责**：
- **Admission Control**：在 burst 到达时快速决定是否接受、延迟或拒绝请求
- **动态 Batch 重整**：在 sub-agent 生命周期变化时（创建/完成/暂停）实时重组 batch
- **公平性保障**：防止某条 Agent 链独占资源导致其他链饿死

### 4.2 KV 管理维度：从"单条长度"到"并发条目数"

**核心变化**：
- Claude Code 的每个 sub-agent 拥有独立上下文窗口 → 系统同时维护 **N 份独立 KV Cache**
- Kimi Swarm 的 100 sub-agents → 峰值同时存在 **100 条活跃 KV Cache 会话**
- PolyKV 的共享池 → CPU 需要管理**压缩 KV 的引用计数和生命周期**

**机头 CPU 的新职责**：
- **Session-Level Placement**：决定每条 Agent 的 KV Cache 放在哪层存储（HBM / DRAM / SSD）
- **跨 Agent 共享发现**：识别哪些 Agent 可以共享前缀或文档 KV（如 PolyKV 的 SharedKVPool）
- **Agent-Aware Eviction**：按 Agent 贡献度而非请求时间决定驱逐优先级（Hive）
- **Resume Path 管理**：Claude Code 的 `/resume` 需要 CPU 从 transcript 重建完整对话状态和 KV Cache

### 4.3 网络编排维度：PD 分离的自适应调度

**核心变化**：
- 标准 PD 分离针对单轮设计，但 Agentic 工作负载产生大量**增量 prefill（append-prefill）**
- 远程 prefill worker 需要传输历史 KV cache，而增量 KV 往往很小
- AMPD 证明：将 **13.9%–31.7%** 的 prefill 留在 decode worker **本地执行**，可避免跨节点传输开销

**机头 CPU 的新职责**：
- **实时负载感知**：基于窗口化 TTFT/ITL 统计（AMPD 使用 10 秒滑动窗口）动态决策 prefill 路由
- **成本估计**：估计本地执行 vs 远程执行的总成本（含 KV 传输、排队时间）
- **Reordering 决策**：在 prefill queue 头部维护 lookahead window（w=3），枚举排序以最大化 SLO 满足率

### 4.4 功耗与能效维度：从算力优化到状态管理优化

**核心变化**：
- KAIROS 发现 Agentic inference 功耗比单轮高 **2–3 个数量级**
- 根本原因是：单次推理被替换为多次工具交错的、有状态的 LLM 调用 + 历史 token 保留
- GPU 频率低于 **900 MHz** 时系统进入 thrashing，反而增加能耗

**机头 CPU 的新职责**：
- **并发控制**：调节同时服务的 Agent 数量，防止聚合上下文超出 GPU 内存
- **频率协调**：根据当前负载动态建议 GPU 频率（避免过低频率导致 thrashing）
- **状态压缩**：在内存压力时触发 context compaction（Claude Code 的五层管道）

### 4.5 运营基础设施维度：98.4% 的代码在 CPU 侧

Claude Code 的架构分析揭示了一个惊人比例：
- **AI 决策逻辑**：~**1.6%**
- **运营基础设施**：~**98.4%**

这 98.4% 包括：
- 权限门控（7 层独立安全机制）
- 工具路由与 MCP lazy loading
- 五层 context compaction pipeline
- 对话恢复（resume/fork）逻辑
- Sub-agent 生命周期管理
- 文件锁定协调（替代消息代理）

**对 AI 机头的启示**：在 Agent Team/Swarm 场景中，**系统性能的上限由 CPU 侧的运营基础设施决定，而非 GPU 侧的模型推理能力**。优化 AI 机头的投资回报率远高于单纯升级 GPU。

---

## 5. 系统级优化方向：从单轮到多 Agent

### 5.1 优化方向矩阵

| 优化层级 | 单轮/单 Agent 时代 | Agent Team/Swarm 时代 | 代表工作 |
|---------|------------------|----------------------|---------|
| **KV Cache 压缩** | 减少单条长度 | 跨 Agent 共享 + 级联重用 | RelayCaching, PolyKV |
| **KV Cache 调度** | 请求级 LRU | Agent 级贡献感知 | Hive Agent-Aware Scheduling |
| **Prefill 调度** | 远程 prefill worker | 本地/远程自适应路由 | AMPD |
| **Decode 优化** | 减少单次生成延迟 | 跨分支 logits 复用 | Hive Logits Cache |
| **上下文管理** | 单条 compaction | 多层级 + 懒加载 | Claude Code 5-layer pipeline |
| **功耗管理** | 固定频率 | 并发感知动态调频 | KAIROS |
| **网络流量** | 计算网络为主 | 存储网络与计算网络联合均衡 | DualPath |

### 5.2 关键优化数据汇总

| 优化技术 | 效果 | 来源 |
|---------|------|------|
| RelayCaching KV 重用 | >80% 重用率，TTFT 4.7× 降低 | arXiv:2603.13289 |
| Hive Agent-Aware Scheduling | Miss rate 降低 33–51% | arXiv:2604.17353 |
| Hive Logits Cache | 1.76× 加速（Hotspot Sampling） | arXiv:2604.17353 |
| AMPD 自适应路由 | SLO 提升 67%（平均）/ 967%（最高） | arXiv:2602.14516 |
| AMPD prefill reordering | 额外 13–15% SLO 提升 | arXiv:2602.14516 |
| KAIROS 功耗优化 | 27% 降低（单实例）/ 46%（多实例） | arXiv:2604.16682 |
| PolyKV 内存压缩 | O(N) → O(1) | arXiv:2604.24971 |
| Claude Code context compaction | 五层渐进式压缩 | arXiv:2604.14228 |
| SideQuest 模型驱动驱逐 | 峰值 token 减少 56–65%，吞吐 +83.9% | arXiv:2602.22603 |
| DualPath 双路径加载 | 在线吞吐 1.96× | arXiv:2602.21548 |

---

## 6. 综合洞察：AI 机头的新边界

### 洞察 1：从"上下文长度"到"并发宽度"——瓶颈的转移

传统长上下文优化的目标是"单条上下文能撑多长"。Agent Swarm 彻底改变了这一目标：
- **Kimi Swarm**：100–300 条独立上下文同时活跃
- **Claude Code**：3–10 条上下文通过 sub-agent 机制并发
- **Hive R3A**：5 个角色化 Agent，但 >70% 工作集中在 3 个核心

**瓶颈从"单条长度"转向"并发条目数"和"跨 Agent 状态协调复杂度"**。AI 机头需要管理的不只是"一条很长的 KV Cache"，而是"100 条中等长度的 KV Cache + 它们之间的共享关系图"。

### 洞察 2：从"请求平等"到"Agent 优先级"——调度哲学的转变

传统推理引擎对请求"一视同仁"，LRU 驱逐是最优策略。多 Agent 系统改变了这一假设：
- 核心 Agent（Decision/Patcher）的 KV 被反复访问，应**常驻 GPU HBM**
- 辅助 Agent（Searcher）的 KV 调用稀疏但可能引入瞬态内存尖峰，应**快速卸载**
- 父 Agent 的 KV 在 sub-agent 执行期间处于"暂停但不死"状态，需要**特殊保留策略**

**AI 机头需要引入 Agent 语义感知**：理解 Agent 的角色、依赖关系和生命周期，才能做出最优的调度决策。

### 洞察 3：从"计算密集"到"状态管理密集"——CPU 负载的本质

Claude Code 的 1.6%/98.4% 比例是一个强烈的信号：
- GPU 负责的是那 1.6% 的模型推理
- CPU 负责的是 98.4% 的**编排、缓存、压缩、恢复、权限、路由**

在 Agent Swarm 场景中，CPU 的负载进一步分化：
- **轻量计算**：Agent 贡献分统计、Shapley 近似、负载预测
- **重量 I/O**：KV Cache 分层搬运、跨节点 RDMA 传输、SSD 读写
- **复杂状态机**：sub-agent 生命周期、resume/fork 路径、context compaction 触发

### 洞察 4：从"GPU 附属"到"控制平面"——硬件配比的结构性变化

多条独立证据指向同一趋势：
- **AMD 官方博客**：CPU:GPU 从 1:4–1:8 转向 **1:1**
- **TrendForce 预测**：每 GW 数据中心 CPU 核心 **4×** 增长
- **NVIDIA Vera CPU**：1.2 TB/s LPDDR5X，定位为 AI Factory "控制平面"
- **BlueField-4**：专用 DPU 将机头负载从主机 CPU 剥离

在 Agent Swarm 场景中，这一趋势被进一步放大：当 100 个 sub-agents 同时竞争 CPU 调度、内存带宽和网络资源时，机头 CPU 的选型从"够发 kernel 就行"变成"需要专用推理编排处理器"。

### 洞察 5：从"单点优化"到"全栈协同"——系统设计的范式转移

Agent Swarm 的优化不能孤立地在某个层面完成，需要全栈协同：

```
前端产品层: Kimi Swarm / Claude Code / Hive
    ↓ 定义 Agent 角色、依赖关系、并发策略
编排调度层: Agent-Aware Scheduling / AMPD / KAIROS
    ↓ 实时决策 prefill 路由、KV placement、频率调节
缓存管理层: RelayCaching / PolyKV / Hive Logits Cache
    ↓ 跨 Agent KV 共享、级联重用、压缩解压
基础设施层: Vera CPU / BlueField-4 / CXL / NVLink-C2C
    ↓ 内存带宽、一致性互连、DPU 卸载
```

**AI 机头处于这个栈的中心位置**：它接收前端的产品语义（Agent 角色、工具调用图），执行实时的编排决策，管理缓存层的生命周期，并驱动基础设施层的资源分配。

---

## 7. 关键数据速查表

### Agent Swarm 工作负载特征

| 指标 | 数值 | 来源 |
|------|------|------|
| Kimi Swarm 并行 sub-agents | 100（K2.5）/ 300（K2.6） | Kimi 官方博客 |
| Kimi Swarm 单次任务工具调用 | >1,500 | Kimi 官方博客 |
| Kimi Swarm 加速比 | 4.5× vs 串行 | Kimi 官方博客 |
| Claude Code Agent Teams token 消耗 | 7× 标准会话 | arXiv:2604.14228 |
| Claude Code AI/基础设施代码占比 | 1.6% / 98.4% | arXiv:2604.14228 |
| KAIROS 平均并发 Agent 数 | 17 | arXiv:2604.16682 |
| KAIROS 平均对话轮次 | 37（范围 1–2,518） | arXiv:2604.16682 |
| KAIROS 功耗 vs 单轮 | 高 2–3 个数量级 | arXiv:2604.16682 |
| Hive 核心 Agent token 集中度 | >70% | arXiv:2604.17353 |

### KV Cache 与调度优化

| 指标 | 数值 | 来源 |
|------|------|------|
| RelayCaching KV 重用率 | >80% | arXiv:2603.13289 |
| RelayCaching TTFT 加速（Agent 5） | 4.71× | arXiv:2603.13289 |
| RelayCaching 长上下文加速 | 9.2× vs Full Prefill | arXiv:2603.13289 |
| PolyKV 内存复杂度 | O(N) → O(1) | arXiv:2604.24971 |
| Hive miss rate 降低 | 33–51% | arXiv:2604.17353 |
| Hive Logits Cache 加速 | 1.76× | arXiv:2604.17353 |
| AMPD vs Dynamo SLO 提升 | 67.29%（平均）/ 967.54%（最高） | arXiv:2602.14516 |
| AMPD prefill 本地执行比例 | 13.9–31.7% | arXiv:2602.14516 |
| KAIROS 功耗降低 | 27%（单实例）/ 46.3%（多实例） | arXiv:2604.16682 |
| KAIROS thrashing 频率阈值 | 900 MHz | arXiv:2604.16682 |
| CPU dequeue 延迟放大 | 19× | arXiv:2603.22774 |
| GPU 计算占比（多 GPU） | 38% | arXiv:2603.22774 |
| AMD CPU:GPU 配比趋势 | 1:4-8 → 1:1 | AMD 官方博客 |
