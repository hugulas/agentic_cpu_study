# 阅读日志：Engram 与 DeepSeek V4 对 KV Cache 及 AI 机头的影响

## 阅读记录

### SRC-01: Engram 论文（Conditional Memory via Scalable Lookup）
- **source id**: SRC-01
- **title**: Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
- **url**: https://arxiv.org/abs/2601.07372
- **local path**: deep-research-engram-dsv4/engram-paper.pdf
- **date**: 2026-01-12
- **source type**: arXiv 论文（DeepSeek-AI + 北京大学）
- **direction id**: D1
- **disposition**: kept
- **note**: 一手来源，明确 Engram 不是 KV Cache，是静态参数表。确定性寻址支持异步预取和多级卸载。100B 参数卸载到 Host 仅 <3% 吞吐损失。U 型缩放律证明 Engram 与 MoE 互补。
- **key claims**: 
  - 每 token 每 layer 检索 16 个离散 embedding（5KB）
  - 100B Engram 表 Host 卸载吞吐损失 1.9%（4B Dense）/ 2.8%（8B Dense）
  - 最优稀疏分配 ρ≈75-80% 给 MoE，20-25% 给 Engram
  - Multi-Query NIAH 从 84.2 提升至 97.0
- **scope tag**: direct

### SRC-02: CXL-Engram 论文
- **source id**: SRC-02
- **title**: Pooling Engram Conditional Memory in Large Language Models using CXL
- **url**: https://arxiv.org/abs/2603.10087
- **local path**: deep-research-engram-dsv4/cxl-engram.pdf
- **date**: 2026-03-10
- **source type**: arXiv 论文
- **direction id**: D2
- **disposition**: kept
- **note**: 首次验证 CXL 内存池化支持 Engram。CXL 延迟接近本地 DRAM，端到端与 DRAM 差距 <1.5%。400B Engram + 16 节点可节省 $166,040。指出 Engram 与 KV Cache 在 CXL 池中共存是开放挑战。
- **key claims**:
  - 每 token 每 layer 仅检索 5KB（16 个 320B embedding）
  - CXL 读取延迟与本地 DRAM 相当
  - Qwen3-4B + Engram：DRAM 5683.7 tok/s vs CXL 5614.4 tok/s（差距 1.2%）
  - 所需带宽仅约 0.7 GB/s
- **scope tag**: direct

### SRC-03: DeepSeek V4 技术报告
- **source id**: SRC-03
- **title**: DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
- **url**: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf
- **local path**: deep-research-engram-dsv4/deepseek-v4-technical-report.pdf
- **date**: 2026-04-24（推断）
- **source type**: 官方技术报告
- **direction id**: D3, D4, D11
- **disposition**: kept
- **note**: 最权威的 V4 一手来源。**重要发现：全文未提及 Engram**。V4 放弃 MLA，使用 high-rank MQA + CSA/HCA。CSA 4x 压缩 + Lightning Indexer top-k，HCA 128x 压缩 + dense attention。1M 上下文 KV Cache 为 V3.2 的 10%（Pro）/ 7%（Flash）。
- **key claims**:
  - V4-Pro: 1.6T 总参 / 49B 激活 / 61 层 / hidden 7168 / 128 query heads / head dim 512
  - V4-Flash: 284B 总参 / 13B 激活 / 43 层 / hidden 4096 / 64 query heads
  - CSA 压缩率 m=4，HCA 压缩率 m'=128
  - Lightning Indexer 内部 FP4，Index Score BF16（2x speedup, 99.7% recall）
  - 相比标准 Transformer BF16 GQA8，1M 上下文 KV Cache 降至约 2%
  - MegaMoE2 加速 1.50-1.73x（一般推理），最高 1.96x（延迟敏感）
- **scope tag**: direct

### SRC-04: StreamIndex 论文
- **source id**: SRC-04
- **title**: StreamIndex: Memory-Bounded Compressed Sparse Attention via Streaming Top-k
- **url**: https://arxiv.org/abs/2605.02568
- **local path**: deep-research-engram-dsv4/streamindex.pdf
- **date**: 2026-05
- **source type**: arXiv 论文
- **direction id**: D5
- **disposition**: kept
- **note**: 揭示 V4 Lightning Indexer 的内存瓶颈：物化中间张量在 S=65K 时 256GB，S=262K 时 4TB。StreamIndex 通过流式 top-k 将峰值 HBM 降到 6.21GB（S=1M），32x 范围扩展。
- **key claims**:
  - V4-Flash 物化路径 S=65,536 即 OOM
  - 分块路径 S=1M 峰值 HBM 仅 6.21GB（V4-Flash）/ 12.27GB（V4-Pro）
  - 分块路径时间 S=1M 约 30,900 ms（单 H200）
  - 精度：位精确集合匹配，召回率 1.0000
- **scope tag**: direct

### SRC-05: HISA 论文
- **source id**: SRC-05
- **title**: (未完整阅读，基于引用信息)
- **url**: https://arxiv.org/abs/2603.28458
- **local path**: deep-research-engram-dsv4/hisa.pdf
- **date**: 2026-03
- **source type**: arXiv 论文
- **direction id**: D5
- **disposition**: maybe
- **note**: 分层索引稀疏注意力，block-level 粗筛 + token-level 精修。在 V3.2 上替换 indexer，32K 加速 2x，128K 加速 4x。与 DSA token 选择 IoU > 99%。
- **key claims**:
  - 32K 加速 2x，128K 加速 4x
  - 与 DSA IoU > 99%
- **scope tag**: adjacent

### SRC-06: DualPath 论文
- **source id**: SRC-06
- **title**: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference
- **url**: https://arxiv.org/abs/2602.21548
- **local path**: deep-research-engram-dsv4/agentic-kv-bottleneck.pdf
- **date**: 2026-02
- **source type**: arXiv 论文
- **direction id**: D6
- **disposition**: kept
- **note**: Agentic 推理的核心系统论文。98.7% KV Cache 命中率，I/O 密集型。PD 分离下 prefill SNIC 饱和、decode SNIC 闲置。DualPath 通过双路径加载将在线吞吐提升 1.96x。
- **key claims**:
  - Agentic 平均 157 轮交互，32.7K 上下文，每轮追加 429 tokens
  - KV-Cache 命中率 98.7%
  - Cache-compute ratio: DS-V3.2 约 22 GB/PFLOP，Qwen2.5-32B 高达 117-267 GB/PFLOP
  - 在线服务平均吞吐提升 1.96x（DS 660B: 2.25x，DS 27B: 1.67x）
  - 瓶颈自由 P/D 比例范围：17 ≤ P/D ≤ 72
- **scope tag**: direct

### SRC-07: SideQuest 论文
- **source id**: SRC-07
- **title**: SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning
- **url**: https://arxiv.org/abs/2602.22603
- **local path**: deep-research-engram-dsv4/model-driven-kv.pdf
- **date**: 2026-02
- **source type**: arXiv 论文（NVIDIA）
- **direction id**: D7
- **disposition**: kept
- **note**: KV Cache 管理从启发式向模型驱动演进。并行辅助线程让模型自己决定哪些工具输出已过时。峰值 token 使用减少 56-65%，吞吐提升 83.9%。
- **key claims**:
  - 峰值 token 使用减少 56-65%
  - KV Cache 内存读取减少 53-71%
  - H100 上吞吐从 828 tok/s 提升至 1523 tok/s（+83.9%）
  - peak batch size 从 24 提升到 36
- **scope tag**: direct

### SRC-08: HeadInfer 论文
- **source id**: SRC-08
- **title**: HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading
- **url**: https://arxiv.org/abs/2502.12574
- **local path**: deep-research-engram-dsv4/headinfer.pdf
- **date**: 2025-02
- **source type**: arXiv 论文
- **direction id**: D8
- **disposition**: kept
- **note**: Head-wise 卸载将粒度从 layer 推进到 head，Llama-3-8B 在 RTX-4090 上支持 4M tokens。KV cache 从 128GB 降至 1GB（减少 92%）。数学等价性保证。
- **key claims**:
  - Llama-3-8B @ 1M tokens：GPU 内存从 207GB 降至 17GB（prefill）/ 16.4GB（decoding）
  - RTX-4090 上支持 4M tokens（标准方法 25K）
  - Prefill 20K 时吞吐 7,210 tok/s（标准 7,235 tok/s，几乎无损）
  - Decoding 1M 时约 6.51 s/token
- **scope tag**: direct

### SRC-09: NVIDIA Dynamo Agentic Inference
- **source id**: SRC-09
- **title**: NVIDIA Dynamo: Agentic Inference
- **url**: https://www.nvidia.com/en-us/data-center/dynamo/
- **local path**: cited-materials/nvidia-agentic-inference-dynamo-2026-04-17.pdf
- **date**: 2026-04-17
- **source type**: 厂商技术博客
- **direction id**: D6
- **disposition**: kept
- **note**: Agentic 推理 WORM 式 KV 访问，85-97% cache hit，11.7x read/write ratio。NVIDIA 官方数据，是 agentic workload 特征的重要证据。
- **key claims**:
  - WORM KV 访问模式
  - 85-97% cache hit（早期层），97.2% aggregate hit
  - 11.7x read/write ratio
- **scope tag**: direct

### SRC-10: CPU-Induced Slowdowns 论文
- **source id**: SRC-10
- **title**: Quantifying and Mitigating CPU-Induced Slowdowns in Multi-GPU LLM Inference
- **url**: https://arxiv.org/abs/2603.22774
- **local path**: cited-materials/cpu-induced-slowdowns-multigpu-llm-2603.22774.pdf
- **date**: 2026-03
- **source type**: arXiv 论文
- **direction id**: D9
- **disposition**: kept
- **note**: 系统量化 CPU 竞争对多 GPU LLM 推理的影响。HTTP 服务占 33%、调度占 29%、GPU 计算仅 38%。dequeue 延迟放大 19x。是理解 AI 机头 CPU 瓶颈的核心论文。
- **key claims**:
  - GPU 计算仅占 38%，HTTP+调度占 62%
  - dequeue 延迟放大 19x
  - CPU oversubscription 放大效应
- **scope tag**: direct

### SRC-11: DeepSeek V4 Model Card
- **source id**: SRC-11
- **title**: DeepSeek-V4 Model Card
- **url**: https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf
- **local path**: deep-research-engram-dsv4/deepseek-v4-model-card.pdf
- **date**: 2026-04
- **source type**: 官方模型卡
- **direction id**: D3
- **disposition**: kept
- **note**: 官方参数确认。MIT 开源，1M 上下文，MoE，Muon 优化器。
- **key claims**:
  - 1M token 上下文
  - MoE 架构
  - Muon Optimizer
- **scope tag**: direct

### SRC-12: 项目既有综述（扩展版）
- **source id**: SRC-12
- **title**: Agentic AI Head CPU Comprehensive Review (Expanded)
- **url**: (本地)
- **local path**: agentic-ai-head-cpu-comprehensive-review-expanded.md
- **date**: 2026-04
- **source type**: 项目综述
- **direction id**: D9, D10, D11
- **disposition**: maybe
- **note**: 项目既有材料，包含 Engram 和 V4 的详细分析。但 V4/Engram 部分基于间接引用，需要与一手来源交叉验证。已发现一处关键偏差：该材料声称 V4 集成 Engram，但技术报告全文未提及。
- **key claims**:
  - 声称 V4 "将模型拆分为静态知识检索模块（Engram，CPU RAM）与动态推理模块（GPU）"
  - CPU RAM 成本比 GPU HBM 低 10-20 倍
  - 1M token 默认上下文
- **scope tag**: out_of_scope_but_suggestive（作为偏差案例记录）

### SRC-13: 中文权威采访（晚点 LatePost）
- **source id**: SRC-13
- **title**: 详解 DeepSeek V4：Infra 巨鲸"四连击"
- **url**: https://www.sohu.com/a/1019108891_211762
- **date**: 2026-05-02/07
- **source type**: 权威中文媒体采访
- **direction id**: D4
- **disposition**: kept
- **note**: DeepSeek 员工刘益枫在 ICLR 现场表示 V4 放弃了 MLA，重回 MQA。与英文博客声称 V4 仍使用 MLA 的信息冲突，本来源可信度更高。
- **key claims**:
  - V4 放弃 MLA，重回 MQA
  - Kimi K2.6、GLM-5.1 等仍采用 MLA
- **scope tag**: direct

### SRC-14: Chimere 开源运行时
- **source id**: SRC-14
- **title**: Chimere: Rust-native MoE inference runtime
- **url**: https://github.com/AIdevsmartdata/chimere
- **local path**: deep-research-engram-dsv4/chimere-repo.html
- **date**: 2026-04-25
- **source type**: 开源代码
- **direction id**: D2
- **disposition**: maybe
- **note**: 支持 multi-tier Engram memory 和 DART speculative drafter using engram n-grams。证明 Engram 的工程化落地已开始，但非官方实现。
- **key claims**:
  - 支持 CHIMERE_ENGRAM_DIR 配置
  - 支持 NEST 自适应
- **scope tag**: adjacent

## === Agent Team/Swarm 补充研究 ===

### SRC-15: Hive 多智能体基础设施
- **source id**: SRC-15
- **title**: Hive: A Multi-Agent Infrastructure for Algorithm- and Task-Level Scaling
- **url**: https://arxiv.org/abs/2604.17353
- **local path**: deep-research-engram-dsv4/hive.pdf
- **date**: 2026-04
- **source type**: arXiv 论文
- **direction id**: D6
- **disposition**: kept
- **note**: 多智能体推理基础设施。Logits Cache 消除跨路径冗余，Agent-Aware Scheduling 按贡献分配 KV Cache。>70% token 消耗集中在少数核心 Agent。
- **key claims**:
  - Logits Cache Hotspot Sampling 平均加速 1.76x
  - Agent-Aware Scheduling 降低 hotspot miss rate 33-51%
  - 驱逐 token 总量减少 19.2-30.2%
  - R3A 剖析：>70% token 消耗和调用频率集中在少数核心 Agent
- **scope tag**: direct

### SRC-16: RelayCaching
- **source id**: SRC-16
- **title**: RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse
- **url**: https://arxiv.org/abs/2603.13289
- **local path**: deep-research-engram-dsv4/relaycaching.pdf
- **date**: 2026-03
- **source type**: arXiv 论文
- **direction id**: D6
- **disposition**: kept
- **note**: 多 Agent 流水线的级联冗余 prefill 问题。直接将 decoding KV cache 重用于下游 prefill，选择性修正处理 prefix variation。累积 prefill 成本从 O(M²) 降至次线性。
- **key claims**:
  - KV Cache 重用率 >80%
  - TTFT 降低 up to 4.71x（Agent 5）
  - 三 Agent 系统长上下文加速 9.2x vs Full Prefill
  - 准确率与 Full Prefill 相当（GSM8K 84.84-85.50%）
- **scope tag**: direct

### SRC-17: AMPD
- **source id**: SRC-17
- **title**: AMPD: Efficient Multi-round LLM Inference over Disaggregated Serving
- **url**: https://arxiv.org/abs/2602.14516
- **local path**: deep-research-engram-dsv4/ampd.pdf
- **date**: 2026-02
- **source type**: arXiv 论文
- **direction id**: D6
- **disposition**: kept
- **note**: 多轮/Agentic 工作负载中 PD 分离的自适应调度。13.9-31.7% prefill 任务路由到 decode worker 本地执行，避免远程 KV 传输。
- **key claims**:
  - vs Dynamo SLO attainment 提升 67.29%（平均）/ 967.54%（最高）
  - vs vLLM 提升 339.74%（平均）/ 3435.1%（最高）
  - 13.9-31.7% prefill 本地执行
  - 自适应路由单独带来 27.37-350% 提升
- **scope tag**: direct

### SRC-18: KAIROS
- **source id**: SRC-18
- **title**: KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving
- **url**: https://arxiv.org/abs/2604.16682
- **local path**: deep-research-engram-dsv4/kairos.pdf
- **date**: 2026-04
- **source type**: arXiv 论文
- **direction id**: D6
- **disposition**: kept
- **note**: Agentic inference 功耗比单轮高 2-3 个数量级。GPU 频率低于 900MHz 会进入 thrashing 状态。平均并发 17 个 Agent，平均 37 轮，最多 2518 轮。
- **key claims**:
  - Agentic inference 功耗比单轮高 2-3 orders of magnitude
  - GPU 频率 900MHz 是 thrashing 阈值
  - 单实例功耗降低 27%（最高 39.8%）
  - 多实例功耗降低 46.3%
  - 平均并发 17 agents，平均 37 turns，最多 2518 turns
- **scope tag**: direct

### SRC-19: Dive into Claude Code
- **source id**: SRC-19
- **title**: Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems
- **url**: https://arxiv.org/abs/2604.14228
- **local path**: deep-research-engram-dsv4/claude-code-analysis.pdf
- **date**: 2026-04
- **source type**: arXiv 论文
- **direction id**: D6
- **disposition**: kept
- **note**: Claude Code 源代码级架构分析。1.6% AI 决策逻辑 / 98.4% 运营基础设施。Agent teams 7x token 消耗。五层 context compaction pipeline。上下文窗口是绑定资源约束。
- **key claims**:
  - AI 决策逻辑 1.6%，运营基础设施 98.4%
  - Agent teams 消耗约 7x 标准会话 token
  - 三种隔离模式：Worktree / Remote / In-process
  - 五层压缩管道：Budget → Snip → Microcompact → Collapse → Auto-compact
- **scope tag**: direct

### SRC-20: Kimi Agent Swarm 博客
- **source id**: SRC-20
- **title**: Kimi Agent Swarm
- **url**: https://www.moonshot.cn/blog/kimi-agent-swarm
- **local path**: cited-materials/kimi-agent-swarm-2026-04-11.pdf
- **date**: 2026-04-11
- **source type**: 官方技术博客
- **direction id**: D6
- **disposition**: kept
- **note**: Kimi 官方博客。100 并行 sub-agents，1,500+ 工具调用，4.5x 加速。自组织结构（CEO 雇佣研究员/分析师）。突破单 Agent 上下文天花板。
- **key claims**:
  - 100 并行 sub-agents
  - 单次任务 >1,500 工具调用
  - 比串行执行快 4.5x
  - 三个场景：Discovery at Scale / Output at Scale / Perspective at Scale
- **scope tag**: direct

### SRC-21: AMD Agentic AI 博客
- **source id**: SRC-21
- **title**: Agentic AI Changes the CPU/GPU Equation
- **url**: https://www.amd.com/en/blogs/2026/agentic-ai-changes-the-cpu-gpu-equation.html
- **date**: 2026
- **source type**: 厂商技术博客
- **direction id**: D9
- **disposition**: kept
- **note**: AMD 官方博客。Agentic AI 时代 CPU:GPU 比例从聊天机器人的 1:4-8 转向 1:1，某些场景 CPU 侧更高。CPU 负责编排、Agent 执行、工具调用、策略和安全。
- **key claims**:
  - CPU:GPU 从 1:4-8 转向 1:1
  - CPU 负责编排、Agent 执行、工具调用、策略、安全
- **scope tag**: direct

### SRC-22: PolyKV
- **source id**: SRC-22
- **title**: PolyKV: Shared Compressed KV Pool for Multi-Agent
- **url**: https://arxiv.org/abs/2604.24971
- **local path**: deep-research-engram-dsv4/polykv.pdf
- **date**: 2026-04
- **source type**: arXiv 论文
- **direction id**: D6
- **disposition**: kept
- **note**: N 个 Agent 处理相同文档上下文时，只计算一次 compressed KV state，内存复杂度从 O(N) 降到 O(1)。
- **key claims**:
  - SharedKVPool 和 PooledAgent 抽象
  - 内存复杂度 O(N) → O(1)
- **scope tag**: direct

## 拒绝记录

### REJ-01: 声称 V4 仍使用 MLA 的英文博客
- **title**: DeepSeek V4 Performance and Pricing Analysis (n1n.ai)
- **reason**: 信息过时，基于 V3 架构惯性推断，与权威中文采访矛盾
- **date**: 2026-04-24

### REJ-02: 浅层媒体报道
- **title**: DeepSeek unveils Engram technology (digitaltoday.co.kr)
- **reason**: 科技媒体报道，无技术实质，仅重复已知信息
- **date**: 2026-01-15

### REJ-03: 综合分析博客（anycap.ai）
- **title**: DeepSeek V4 Engram Explained
- **reason**: 声称 V4 使用 Engram，但无官方来源支撑，与技术报告矛盾
- **date**: 2026-04-24
