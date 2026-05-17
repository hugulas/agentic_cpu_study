# Subchapter Support Plan

## 更新目标

基于 2026-05-17 完成的深度研究（Engram/DSV4 + Agent Swarm），对 review-expansion-workspace 的 11 个章节进行刷新，核心任务：

1. **修正关键事实错误**：DeepSeek V4 未集成 Engram，两者是并行研究线
2. **补充 Engram 独立研究**：添加 Engram 论文和 CXL-Engram 论文的具体数据
3. **补充 V4 技术细节**：放弃 MLA 重回 MQA、StreamIndex Indexer 瓶颈
4. **新增 Agent Swarm 系统研究**：Hive、RelayCaching、AMPD、KAIROS、Claude Code 架构分析、PolyKV
5. **刷新平台信号**：AMD CPU:GPU 1:1 趋势

---

## 逐章计划

### Ch01 摘要
- **main judgment**: 核心判断不变（CPU 成为 inference orchestration layer），但需修正 Engram-V4 关系
- **source ids**: SRC-01, SRC-03, SRC-15, SRC-16, SRC-17, SRC-18, SRC-19, SRC-20, SRC-21
- **key numbers**: 
  - Engram 100B Host 卸载 <3% 损失
  - V4 放弃 MLA 重回 MQA
  - V4 1M 上下文 KV 为 V3.2 的 10%
  - Agentic 98.7% KV 命中率 / 11.7x r/w
  - RelayCaching >80% KV 重用 / 4.7x TTFT
  - Hive miss rate 降低 33-51%
  - AMPD SLO 提升 67-967%
  - KAIROS 功耗高 2-3 数量级
  - Claude Code 1.6%/98.4%
  - Kimi 100 sub-agents / 1,500 tool calls
  - AMD CPU:GPU 1:1
- **figures**: 无需新增图
- **boundary**: 保持简洁，数字点到为止

### Ch02 引言
- **main judgment**: 瓶颈外溢到 host 侧，但修正 Engram 作为 V4 组件的说法
- **source ids**: SRC-01, SRC-03, SRC-13, SRC-15, SRC-20
- **key numbers**: 
  - Engram 是独立研究线（arXiv:2601.07372），V4 未提及
  - V4 三大创新：CSA+HCA, mHC, Muon
  - Kimi Swarm 100 sub-agents
- **figures**: 无需新增
- **boundary**: 与 Ch04 的 Engram 详细分析形成呼应

### Ch04 主线二：KV 卸载
- **main judgment**: 需重大重构 3.2 节，从"V4 的 Engram"改为"Engram 与 V4：两条互补路线"
- **source ids**: SRC-01, SRC-02, SRC-03, SRC-04, SRC-13, SRC-15, SRC-16, SRC-22
- **key numbers**:
  - Engram: 100B Host 卸载 <3%，CXL 差距 <1.5%，每 token 5KB，带宽 ~0.7 GB/s
  - V4: CSA 4x + HCA 128x，KV 降至 V3.2 的 10%，~2% 标准 Transformer
  - V4 放弃 MLA 重回 MQA（晚点采访确认）
  - StreamIndex: Indexer 64K 即 OOM（256GB），S=1M 峰值 HBM 6.21GB
  - Hive: Agent-Aware miss rate 降低 33-51%
  - RelayCaching: >80% 重用，TTFT 4.7x，O(M²)→次线性
  - PolyKV: O(N)→O(1)
- **figures**: 
  - 可引用 assets/extracted/nosa-01.png（NOSA）已存在
  - 可引用 assets/extracted/scoutattn-1.png（ScoutAttention）已存在
  - 新增概念：Engram vs V4 双轨对比表
- **boundary**: 3.2 节需重写，其他节可局部刷新

### Ch07 真实工作负载
- **main judgment**: 从"三项遗漏"扩展为包含 Agent Swarm 系统级研究的四维分析
- **source ids**: SRC-15, SRC-16, SRC-17, SRC-18, SRC-19, SRC-20, SRC-21, SRC-22
- **key numbers**:
  - Claude Code: 1.6%/98.4%，7x token，五层压缩管道
  - Kimi Swarm: 100 sub-agents，1,500 tool calls，4.5x 加速
  - Hive: >70% token 集中在核心 Agent
  - RelayCaching: >80% KV 重用
  - AMPD: 13.9-31.7% prefill 本地执行
  - KAIROS: 平均 17 agents，37 turns，功耗高 2-3 数量级
- **figures**: 无需新增图，但需补充表格
- **boundary**: 可新增 6.5/6.6 节覆盖系统级研究，保持与前面主线章节的呼应

### Ch08 平台信号
- **main judgment**: 补充 AMD CPU:GPU 1:1 官方信号
- **source ids**: SRC-21
- **key numbers**: CPU:GPU 从 1:4-8 转向 1:1
- **figures**: 无需新增
- **boundary**: 局部刷新 7.3 节

### Ch09 讨论
- **main judgment**: 修正 Engram-V4 关系，补充 Agent Swarm 研究空白
- **source ids**: SRC-01, SRC-03, SRC-15, SRC-18
- **key numbers**: 同 Ch01
- **figures**: 无需
- **boundary**: 修正 8.3 节，新增 8.4 关于 Agent Swarm 的空白

### Ch10 结论
- **main judgment**: 修正 Engram 相关表述，补充 Agent Swarm 作为第五驱动力
- **source ids**: SRC-01, SRC-03, SRC-20
- **key numbers**: 同 Ch01
- **figures**: 无需
- **boundary**: 9.1/9.2/9.3 均需刷新

### Ch11 参考文献
- **main judgment**: 新增 Engram、CXL-Engram、StreamIndex、Hive、RelayCaching、AMPD、KAIROS、Claude Code、AMD 博客、PolyKV 的引用
- **source ids**: 全部新增来源
- **boundary**: 补充 10+ 条新引用
