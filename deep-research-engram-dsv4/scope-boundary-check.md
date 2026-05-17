# 范围边界检查：Engram 与 DeepSeek V4 对 KV Cache 及 AI 机头的影响

## 边界检查清单

### 1. Engram vs KV Cache
- **证据族**: Engram 论文、CXL-Engram 论文
- **风险**: 将 Engram 与 KV Cache 混为一谈，或认为 Engram "替代"了 KV Cache
- **判定**: in-scope（核心区分对象）
- **使用方式**: 明确指出 Engram 是静态参数表，KV Cache 是动态中间激活；两者正交互补

### 2. Engram vs DeepSeek V4 集成关系
- **证据族**: 项目既有综述（扩展版）、V4 技术报告、各种分析博客
- **风险**: 项目既有材料声称 V4 "集成 Engram"，但技术报告全文未提及
- **判定**: adjacent but usable
- **使用方式**: 
  - 报告需明确指出：V4 技术报告（55 页）全文未出现 "Engram" 一词
  - Engram 是独立研究线（arXiv:2601.07372，GitHub: deepseek-ai/Engram）
  - 两者可视为同一研究方向的"并行探索"：V4 走 CSA/HCA 序列压缩路线，Engram 走条件记忆静态检索路线
  - 未来模型（如 V5）可能同时采用两者

### 3. 训练 vs 推理
- **证据族**: V4 技术报告（含训练细节）、Engram 论文（含训练细节）
- **风险**: 训练阶段的 offload 策略（如 Teacher Weights Offload）与推理阶段混淆
- **判定**: 训练细节 → exclude / 推理细节 → in-scope
- **使用方式**: 仅引用推理相关的 offload 描述（On-Disk KV Cache Storage、Host Codegen 优化）

### 4. 权重卸载 vs KV Cache 卸载
- **证据族**: MoE 专家卸载、KV Cache offload、Engram 卸载
- **风险**: 将权重卸载的瓶颈与 KV Cache 卸载混为一谈
- **判定**: in-scope（作为关联背景，但需明确区分）
- **使用方式**: 
  - KV Cache / Engram 卸载：核心主题
  - MoE 专家权重卸载：作为 AI 机头 CPU 的关联负载提及
  - 两者对 CPU 内存带宽和 PCIe 的需求不同

### 5. 产品营销 vs 技术证据
- **证据族**: NVIDIA Dynamo 博客、Astera Labs CXL 报告、TrendForce 分析
- **风险**: 厂商数据可能存在建模乐观偏差
- **判定**: adjacent but usable
- **使用方式**: 
  - NVIDIA Dynamo 数据（85-97% hit, 11.7x r/w）标注为 vendor data，需交叉验证
  - Astera Labs CXL 数据（GPU 需求降低 87%）标注为 vendor modeling
  - 优先使用 arXiv 论文的独立实验数据

### 6. 架构信号 vs 直接服务证据
- **证据族**: Vera CPU、BlueField-4、CXL Switch 等硬件路线图
- **风险**: 硬件信号不等于实际部署证据
- **判定**: adjacent but usable
- **使用方式**: 作为"平台信号"章节，说明硬件厂商正在围绕 CPU 控制平面收敛，但不将其等同于已验证的部署方案

### 7. 旧术语 vs 新术语
- **证据族**: MLA（V2/V3）vs MQA（V4）vs CSA/HCA（V4）
- **风险**: 部分英文博客仍用 MLA 描述 V4，造成混淆
- **判定**: in-scope（需要澄清）
- **使用方式**: 明确指出 V4 放弃 MLA，重回 MQA；CSA/HCA 是序列维度压缩，与 MLA 的 head 维度压缩不同

### 8. Agentic 推理 vs 普通长上下文推理
- **证据族**: DualPath、SideQuest、NVIDIA Dynamo
- **风险**: 将 agentic 特有的 WORM 访问模式与普通长上下文混为一谈
- **判定**: in-scope（agentic 是 AI 机头负载变化的关键驱动力）
- **使用方式**: 强调 agentic 工作负载 98.7% KV Cache 命中率和 11.7x 读写比是理解系统瓶颈的关键
