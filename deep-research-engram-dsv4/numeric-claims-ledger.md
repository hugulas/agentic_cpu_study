# 数字声明账本：Engram 与 DeepSeek V4 对 KV Cache 及 AI 机头的影响

## 账本说明

- **direct evidence**: 来自一手论文/官方报告的原始数据
- **inferred implication**: 基于直接证据的合理推断
- **in report**: 是否已纳入最终报告

---

## 声明列表

### N1: Engram Host 卸载吞吐损失
- **claim id**: N1
- **number**: 1.9% (4B Dense) / 2.8% (8B Dense)
- **what it measures**: 100B 参数 Engram 表完全卸载到 Host DRAM 时的推理吞吐量损失
- **source id**: SRC-01
- **evidence type**: direct evidence
- **in report**: no
- **note**: 保守基线，强制所有检索走 PCIe，未利用 HBM 热缓存

### N2: Engram 最优稀疏分配
- **claim id**: N2
- **number**: ρ≈75-80% MoE, 20-25% Engram
- **what it measures**: 在 10B 规模下，最优稀疏参数分配比例
- **source id**: SRC-01
- **evidence type**: direct evidence
- **in report**: no
- **note**: U 型缩放律，纯 MoE 验证 loss 1.7248，最优 1.7109

### N3: Engram NIAH 准确率提升
- **claim id**: N3
- **number**: 84.2 → 97.0 (Multi-Query NIAH)
- **what it measures**: Engram-27B vs MoE-27B 在 32k 上下文下的 Needle-in-Haystack 准确率
- **source id**: SRC-01
- **evidence type**: direct evidence
- **in report**: no

### N4: Engram CXL 池化性能差距
- **claim id**: N4
- **number**: <1.5% (Qwen3-4B) / <0.3% (Qwen3-8B)
- **what it measures**: CXL 池化 vs 本地 DRAM 的端到端推理吞吐差距
- **source id**: SRC-02
- **evidence type**: direct evidence
- **in report**: no
- **note**: Qwen3-4B: DRAM 5683.7 vs CXL 5614.4 tok/s；Qwen3-8B: DRAM 3909.7 vs CXL 3895.0 tok/s

### N5: Engram CXL 成本节省
- **claim id**: N5
- **number**: $166,040 (400B Engram + 16 节点)
- **what it measures**: CXL 池化相比本地 DRAM 的总成本节省
- **source id**: SRC-02
- **evidence type**: direct evidence
- **in report**: no

### N6: V4-Pro KV Cache 压缩比
- **claim id**: N6
- **number**: 10% (vs V3.2) / ~2% (vs 标准 Transformer BF16 GQA8)
- **what it measures**: 1M token 上下文下 V4-Pro 的 KV Cache 内存占用比例
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no
- **note**: V3.2 1M 上下文约 83.88 GiB → V4-Pro 约 9.62 GiB；标准 Transformer 约 480 GiB → V4 约 9.62 GiB

### N7: V4-Flash KV Cache 压缩比
- **claim id**: N7
- **number**: 7% (vs V3.2)
- **what it measures**: 1M token 上下文下 V4-Flash 的 KV Cache 内存占用比例
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N8: V4 单 token FLOPs 降低
- **claim id**: N8
- **number**: 27% (Pro) / 10% (Flash) (vs V3.2)
- **what it measures**: 1M token 上下文下单 token 推理 FLOPs 比例
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N9: V4 CSA 压缩率
- **claim id**: N9
- **number**: 4x (m=4)
- **what it measures**: CSA 沿序列维度的 KV 压缩率
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N10: V4 HCA 压缩率
- **claim id**: N10
- **number**: 128x (m'=128)
- **what it measures**: HCA 沿序列维度的 KV 压缩率
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N11: V4 Lightning Indexer 内存瓶颈
- **claim id**: N11
- **number**: 256 GB (S=65K) / 4 TB (S=262K)
- **what it measures**: 物化中间张量的大小
- **source id**: SRC-04
- **evidence type**: direct evidence
- **in report**: no

### N12: StreamIndex 峰值 HBM 降低
- **claim id**: N12
- **number**: 6.21 GB (V4-Flash S=1M) / 12.27 GB (V4-Pro S=1M)
- **what it measures**: 流式 top-k 分块路径的峰值 HBM 占用
- **source id**: SRC-04
- **evidence type**: direct evidence
- **in report**: no

### N13: StreamIndex 范围扩展
- **claim id**: N13
- **number**: 32x (从 32K 扩展到 1M)
- **what it measures**: 可运行序列长度范围扩展倍数
- **source id**: SRC-04
- **evidence type**: direct evidence
- **in report**: no

### N14: Agentic KV Cache 命中率
- **claim id**: N14
- **number**: 98.7%
- **what it measures**: Agentic 工作负载的 KV-Cache 命中率
- **source id**: SRC-06
- **evidence type**: direct evidence
- **in report**: no

### N15: Agentic 交互特征
- **claim id**: N15
- **number**: 157 轮 / 32.7K 上下文 / 429 tokens/轮追加
- **what it measures**: Agentic 工作负载的平均交互轮数、上下文长度、每轮追加 token 数
- **source id**: SRC-06
- **evidence type**: direct evidence
- **in report**: no

### N16: Cache-Compute Ratio
- **claim id**: N16
- **number**: 22 GB/PFLOP (DS-V3.2) / 117-267 GB/PFLOP (Qwen2.5-32B)
- **what it measures**: KV-Cache 加载量与计算量的比值
- **source id**: SRC-06
- **evidence type**: direct evidence
- **in report**: no

### N17: DualPath 吞吐提升
- **claim id**: N17
- **number**: 1.96x (在线平均) / 1.87x (离线最高)
- **what it measures**: DualPath 双路径加载相比基础方案的吞吐提升
- **source id**: SRC-06
- **evidence type**: direct evidence
- **in report**: no

### N18: SideQuest 内存节省
- **claim id**: N18
- **number**: 56-65% 峰值 token 减少 / 53-71% KV Cache 内存读取减少
- **what it measures**: 模型驱动 KV Cache 管理的内存节省幅度
- **source id**: SRC-07
- **evidence type**: direct evidence
- **in report**: no

### N19: SideQuest 吞吐提升
- **claim id**: N19
- **number**: 83.9%
- **what it measures**: H100 上峰值系统吞吐提升
- **source id**: SRC-07
- **evidence type**: direct evidence
- **in report**: no

### N20: HeadInfer 内存节省
- **claim id**: N20
- **number**: 92% (KV cache 从 128GB → 1GB)
- **what it measures**: Head-wise 卸载在 Llama-3-8B @ 1M tokens 下的 GPU 内存节省
- **source id**: SRC-08
- **evidence type**: direct evidence
- **in report**: no

### N21: HeadInfer 上下文扩展
- **claim id**: N21
- **number**: 4M tokens (Llama-3-8B on RTX-4090)
- **what it measures**: Head-wise 卸载支持的最大上下文长度
- **source id**: SRC-08
- **evidence type**: direct evidence
- **in report**: no

### N22: NVIDIA Dynamo KV 读写比
- **claim id**: N22
- **number**: 11.7x
- **what it measures**: Agentic 推理中 KV Cache 的累计读写比
- **source id**: SRC-09
- **evidence type**: direct evidence
- **in report**: no

### N23: NVIDIA Dynamo Cache Hit
- **claim id**: N23
- **number**: 85-97% (早期层) / 97.2% (aggregate)
- **what it measures**: Agentic 推理 KV Cache 命中率
- **source id**: SRC-09
- **evidence type**: direct evidence
- **in report**: no

### N24: CPU 竞争对 GPU 推理的影响
- **claim id**: N24
- **number**: GPU 计算仅占 38% / dequeue 延迟放大 19x
- **what it measures**: CPU 竞争对多 GPU LLM 推理的时间分配和延迟放大
- **source id**: SRC-10
- **evidence type**: direct evidence
- **in report**: no

### N25: V4 模型规模
- **claim id**: N25
- **number**: 1.6T 总参 / 49B 激活 (Pro) / 284B 总参 / 13B 激活 (Flash)
- **what it measures**: DeepSeek V4 的参数规模
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N26: V4 注意力配置
- **claim id**: N26
- **number**: 128 query heads / head dim 512 / top-k 1024 (Pro)
- **what it measures**: V4-Pro 的 attention 配置参数
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N27: V4 FP4 Index Score 加速
- **claim id**: N27
- **number**: 2x speedup / 99.7% recall
- **what it measures**: Index Score 从 FP32 量化到 BF16 的加速效果和召回率
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N28: V4 MegaMoE2 加速
- **claim id**: N28
- **number**: 1.50-1.73x (一般推理) / 最高 1.96x (延迟敏感)
- **what it measures**: MegaMoE2 对 V4 推理的端到端加速
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N29: Engram 每 token 检索量
- **claim id**: N29
- **number**: 5KB / token / layer
- **what it measures**: Engram 每 token 每 layer 的检索数据量
- **source id**: SRC-02
- **evidence type**: direct evidence
- **in report**: no

### N30: Engram 所需带宽
- **claim id**: N30
- **number**: ~0.7 GB/s
- **what it measures**: Engram 推理所需的内存带宽
- **source id**: SRC-02
- **evidence type**: direct evidence
- **in report**: no
- **note**: 极低带宽需求使其非常适合 CXL/Host 内存

### N31: I/O-Compute Ratio 下降趋势
- **claim id**: N31
- **number**: 14.4x 下降 (Ampere → Blackwell)
- **what it measures**: GPU 算力提升与 I/O 带宽提升的错配比例
- **source id**: SRC-06
- **evidence type**: direct evidence
- **in report**: no

### N32: V4 训练 token 数
- **claim id**: N32
- **number**: 32T (Flash) / 33T (Pro)
- **what it measures**: V4 预训练 token 数
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N33: V4 训练序列长度渐进
- **claim id**: N33
- **number**: 4K → 16K → 64K → 1M
- **what it measures**: V4 训练时的序列长度渐进策略
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no

### N34: HeadInfer Prefill 吞吐
- **claim id**: N34
- **number**: 516 tok/s (1M) / 7,210 tok/s (20K)
- **what it measures**: HeadInfer 在 Llama-3-8B 上的 prefill 吞吐
- **source id**: SRC-08
- **evidence type**: direct evidence
- **in report**: no

### N35: V4 推理 On-Disk KV 策略
- **claim id**: N35
- **number**: SWA KV 体积约为压缩 KV 的 8 倍
- **what it measures**: Sliding Window Attention KV 与压缩 KV 的体积比
- **source id**: SRC-03
- **evidence type**: direct evidence
- **in report**: no
