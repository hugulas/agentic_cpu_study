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
