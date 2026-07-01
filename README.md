# LLM, Hello!

`LLM, Hello!` 是部署在 [llmhello.com](https://llmhello.com) 的个人技术博客与 LLM 实验型知识站点。

这个项目最初基于 Fuwari / Astro 博客主题搭建，目前正在从“文章型技术博客”升级为围绕大模型、Agent、Prompt、RAG 与人机协作的个人实验室。

## 网站定位

当前目标不是做一个复杂 SaaS，而是先把静态博客扩展成一个更丰富、可持续生长的 AI 技术站点：

- 技术文章：记录 LLM、RAG、Agent、应用开发和部署实践。
- 学习地图：把零散文章组织成一条可持续更新的学习路线。
- 实验室：承载轻量前端工具、流程演示和模型行为观察。
- Prompt 配方：沉淀可复用提示词模板和使用经验。
- 项目日志：记录真实项目从想法、原型、踩坑到上线的过程。

详细规划见：[docs/llmhello-site-plan.md](docs/llmhello-site-plan.md)。

## 当前站点结构

主要页面：

- `/`：首页，包含站点入口和最新文章。
- `/labs/`：实验室，目前包含 Token 成本估算器。
- `/map/`：LLM 学习地图。
- `/prompts/`：Prompt 配方。
- `/projects/`：项目日志。
- `/archive/`：文章归档。
- `/about/`：关于页面。

主要源码目录：

```text
src/
  components/        Astro / Svelte 组件
  content/posts/     博客文章
  content/spec/      特殊页面内容
  layouts/           页面布局
  pages/             Astro 路由页面
  styles/            全局样式
  utils/             内容、日期、URL 等工具函数
docs/
  llmhello-site-plan.md  网站规划文档
public/
  CNAME              GitHub Pages 自定义域名
```

## 技术栈

- [Astro](https://astro.build/)：静态站点框架
- [Svelte](https://svelte.dev/)：交互组件
- [Tailwind CSS](https://tailwindcss.com/)：样式工具
- [Pagefind](https://pagefind.app/)：静态搜索
- [Expressive Code](https://expressive-code.com/)：代码块增强
- [Astro Icon](https://www.astroicon.dev/) / Iconify：图标
- GitHub Pages：站点托管

## 本地运行

建议使用 Node.js LTS 和 pnpm。

检查环境：

```powershell
node -v
npm -v
pnpm -v
```

如果还没有 pnpm，可以用 Corepack 启用：

```powershell
corepack enable
corepack prepare pnpm@9.14.4 --activate
```

安装依赖：

```powershell
pnpm install
```

启动开发服务器：

```powershell
pnpm dev
```

默认访问：

```text
http://localhost:4321/
```

构建生产版本：

```powershell
pnpm build
```

本地预览生产构建：

```powershell
pnpm preview
```

## 常用命令

| 命令 | 说明 |
| --- | --- |
| `pnpm install` | 安装依赖 |
| `pnpm dev` | 启动本地开发服务器 |
| `pnpm build` | 构建生产站点，并生成 Pagefind 搜索索引 |
| `pnpm preview` | 本地预览生产构建 |
| `pnpm check` | 运行 Astro 检查 |
| `pnpm type-check` | 运行 TypeScript 类型检查 |
| `pnpm format` | 使用 Biome 格式化 `src` |
| `pnpm lint` | 使用 Biome 检查并修复 `src` |
| `pnpm new-post <filename>` | 创建新文章 |

## 新建文章

推荐使用脚本：

```powershell
pnpm new-post my-post
```

文章位于：

```text
src/content/posts/
```

文章 frontmatter 示例：

```yaml
---
title: 我的第一篇文章
published: 2026-01-01
description: 文章摘要
image: ./cover.jpg
tags: [LLM, RAG]
category: AI 工程
draft: false
lang: zh_CN
---
```

字段说明：

- `title`：文章标题
- `published`：发布时间
- `updated`：更新时间，可选
- `description`：文章摘要
- `image`：封面图，可选
- `tags`：标签
- `category`：分类
- `draft`：是否草稿
- `lang`：文章语言，可选

## Markdown 能力

项目保留了 Fuwari 主题中的 Markdown 增强能力：

- GitHub Flavored Markdown
- Admonitions 提示块
- GitHub 仓库卡片
- Expressive Code 代码高亮
- 数学公式
- 文章目录
- 图片预览

## 部署

当前站点托管在 GitHub Pages，并通过 `public/CNAME` 绑定：

```text
llmhello.com
```

构建产物输出到：

```text
dist/
```

当前阶段保持静态站点架构。后续如果接入真实 LLM API，建议新增独立后端服务，例如：

```text
api.llmhello.com
```

如果未来出现用户登录、文档上传、私有知识库、数据库或复杂 Agent 后台任务，再考虑新增独立应用：

```text
app.llmhello.com
```

## 当前状态

已完成：

- 修正站点名称、中文副标题和个人简介。
- 首页已从文章列表页改为正式门户页。
- 新增实验室、学习地图、Prompt 配方、项目日志页面。
- 新增 Token 成本估算器。
- 扩展顶部导航，并优化中小屏菜单行为。
- 支持首页使用无侧边栏宽版布局。
- 新增网站规划文档。
- 已通过 `pnpm build` 构建验证。

待完成：

- 把已有文章挂到学习地图。
- 实现 Prompt 对比器。
- 实现 RAG 流程演示。
- 实现 Agent 工作流演示。
- 补充 Prompt 配方库。
- 为网站升级写第一篇项目日志。

## 许可

当前项目继承原主题的 MIT License。站点内容的版权和授权可在后续按个人博客策略单独补充说明。
