# LLMHello 模型发现平台

## 当前交付

保留 Astro 静态站架构与现有文章 URL。首页主线调整为「发现 → 筛选 → 对比 → 费用估算 → 官方平台」。

- `/`：任务选型入口、模型列表、常用对比与内容入口。
- `/models/`、`/models/[id]/`：统一模型目录与详情。
- `/rankings/`、`/rankings/[category]/`：按场景过滤，按参考成本或规格排序。
- `/compare/`：2–3 个模型对比；URL 参数 `models` 支持分享。
- `/compare/gpt-vs-claude/` 等：明确具体模型版本的常用对比页。
- `/finder/`：本地规则匹配任务，按场景、长度、预算过滤候选。
- `/pricing/`：每次输入/输出 × 请求量、输入缓存比例、长上下文和 DeepSeek 非高峰处理。
- `/benchmarks/`：来源、排序方法、费用边界与独立评测链接。
- `/learn/`：保留上一版教程首页；文章、地图、实验与配方仍可访问。
- `/rankings/legacy/`：旧版未经重验的榜单归档，设置 noindex 并从 sitemap 排除。

## 数据与维护

`src/data/models.ts` 是模型资料的单一来源。2026-09-07 核验 6 个模型、4 家厂商，价格与规格来源见各条目。

更新时需同时检查模型别名、价格条件、缓存费用、上下文与输出上限，并更新 checkedAt。Gemini 当前优惠在 2026-12-31 结束；不应只改日期而不核验价格。新数据先通过下列检查再发布。

排序序号不是模型能力排名。没有统一独立质量/速度数据，不渲染虚构分数。图片与视频生成暂不纳入文本计费表，筛选会提示目录范围。联网能力字段为本次已核验能力，不代表未核验条目不支持该功能。

选型为本地可解释规则，不调用远程 LLM。预算是给定批次的 Token 预算；不含 OCR、Embedding、检索、搜索工具、缓存写入/存储和税费。相同文本在不同模型中的 Token 数可能不同。

## 验证

- `node --experimental-strip-types --test scripts/model-discovery.test.mjs`（Node 24）
- `ASTRO_TELEMETRY_DISABLED=1` 后运行 Astro build/check。
- `node node_modules/pagefind/lib/runner/bin.cjs --site dist`

现有环境的 pnpm/Corepack 目录受限时，可用 `node node_modules/astro/astro.js build` 与 Pagefind 命令执行同等构建步骤，不必修改锁文件。

## 后续产品阶段

1. 引入有授权、可追溯的质量与速度数据，保留版本、样本量和测试环境。
2. 扩展图片/视频专用模型及其计费单位。
3. 用用户真实任务建立可重复评测，而非用总榜直接推断任务质量。
4. 在明确预算和服务边界后接入真实试用 API；密钥留在服务端。
5. 联盟链接、赞助位或付费服务上线时明确披露；目前官方入口没有附加联盟跟踪或实际模型调用。

未迁移托管服务，未修改域名配置，未推送或发布生产站点。
