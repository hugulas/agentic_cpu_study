# PDF 图表提取结果索引

本次共处理 17 个 PDF 文件，提取了页面截图和内嵌图片。

## 提取结果总览

| PDF 文件名 | 页面截图 | 内嵌图片 | 备注 |
|-----------|---------|---------|------|
| ai-rs-memory-wall-llm-inference-2026-03 | 8 页 | 1 张 | 网页文章PDF |
| anthropic-claude-code-subagents-current | 10 页 | 0 张 | 无内嵌图片 |
| comem-openreview-2025 | 2 页 | 0 张 | OpenReview评论 |
| diligence-stack-secret-agent-cpu-2026-03 | 6 页 | 0 张 | 无内嵌图片 |
| kimi-agent-swarm-2026-04-11 | 8 页 | 5 张 | 含高清宣传图 |
| netapp-kv-cache-offloading-2025-11 | 1 页 | 0 张 | 单页文档 |
| nvidia-cpu-gpu-memory-sharing-2025-09-05 | 10 页 | 2 张 | 技术博客 |
| nvidia-disaggregated-llm-k8s-2026-03-23 | 10 页 | 2 张 | 技术博客 |
| nvidia-inference-transfer-library-2026-03-09 | 10 页 | 2 张 | 技术博客 |
| nvidia-kv-bottlenecks-dynamo-2025-09-18 | 10 页 | 2 张 | 技术博客 |
| nvidia-wide-expert-parallelism-2025-12-18 | 10 页 | 2 张 | 技术博客 |
| patsnap-moe-inference-patents-2026-04 | 10 页 | 1 张 | 专利分析 |
| raj-agentic-ai-cpu-centric-2511.00739 | 10 页 | 18 张 | **学术论文，图表丰富** |
| rmmod-agentic-cpu-bottleneck-2026-03 | 10 页 | 0 张 | 无内嵌图片 |
| uncoveralpha-agentic-cpu-bottleneck-2026-02 | 10 页 | 0 张 | 无内嵌图片 |
| volcengine-mobile-use-agent-2026-04-29 | 5 页 | 8 张 | 技术博客 |
| zylos-ai-inference-optimization-2026-01 | 10 页 | 17 张 | 多为重复小图 |

## 目录结构

- `*-pages/` - PDF 页面渲染截图（PNG, 200 DPI）
- `*-images/` - 从 PDF 中提取的内嵌原始图片

## 精选图表

较大尺寸（>80KB）的内嵌图片已复制到 `../extracted-figures/` 目录，共 31 张。
