---
title: 个人 RAG 知识库：从文档到可追溯回答
published: 2026-07-09
description: 从文档整理、切块、Embedding、向量检索、引用回答到失败排查，完成一个个人知识库助手的最小闭环。
tags: [RAG, 知识库, Embedding, 实战案例]
category: 实战案例
lang: zh_CN
draft: false
---

# 个人 RAG 知识库：从文档到可追溯回答

RAG 的价值不是让模型“凭空更聪明”，而是让模型在回答前先查你提供的资料。这个案例的目标，是做出一个最小可复现的个人知识库助手。

## 最小闭环

一个可用的 RAG 知识库至少包含六步：

1. 准备文档。
2. 清洗文本。
3. 文档切块。
4. 生成 Embedding。
5. 向量检索相关片段。
6. 基于片段生成带引用的回答。

## 项目结构

```text
personal-rag/
├─ docs/
│  └─ notes.md
├─ src/
│  ├─ load.ts
│  ├─ chunk.ts
│  ├─ embed.ts
│  ├─ retrieve.ts
│  └─ answer.ts
└─ .env
```

## 第一步：准备文档

先从最简单的 Markdown 文档开始。不要一上来处理 PDF、网页、表格和图片，否则会把问题混在一起。

示例文档：

```markdown
# RAG 学习笔记

RAG 是 Retrieval-Augmented Generation 的缩写。
它通过外部检索补充模型上下文，适合企业知识库、个人笔记和文档问答。
```

## 第二步：文档切块

切块是 RAG 的第一个关键点。块太短会丢语义，块太长会影响召回精度。

入门阶段可以先用简单规则：

```ts
function chunkText(text: string, size = 500, overlap = 80) {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + size, text.length);
    chunks.push(text.slice(start, end));
    start = end - overlap;
    if (start < 0) start = 0;
    if (end === text.length) break;
  }

  return chunks;
}
```

后续再升级为按标题、段落、语义边界切块。

## 第三步：生成 Embedding

Embedding 的作用是把文本转换成向量，方便做语义相似度检索。

你可以先把每个 chunk 变成：

```ts
type ChunkRecord = {
  id: string;
  content: string;
  source: string;
  embedding: number[];
};
```

入门阶段不必纠结向量数据库，先用内存数组模拟也可以。

## 第四步：检索相关片段

用户问题也要生成 Embedding，然后和文档块计算相似度。

```ts
function cosineSimilarity(a: number[], b: number[]) {
  const dot = a.reduce((sum, value, index) => sum + value * b[index], 0);
  const normA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
  const normB = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
  return dot / (normA * normB);
}
```

检索时取相似度最高的前几个片段：

```ts
const topChunks = chunks
  .map((chunk) => ({
    ...chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }))
  .sort((a, b) => b.score - a.score)
  .slice(0, 3);
```

## 第五步：生成带引用回答

RAG Prompt 要非常明确：只能基于资料回答。

```text
你是一个基于资料回答问题的助手。

用户问题：
{question}

资料片段：
[1] {chunk_1}
[2] {chunk_2}
[3] {chunk_3}

要求：
1. 只基于资料片段回答。
2. 关键结论后标注来源编号。
3. 如果资料不足，请说明缺少什么。
4. 不要编造资料中没有的信息。
```

## 第六步：失败排查

RAG 出错时，先不要怪模型。按链路排查：

| 现象 | 优先排查 |
|---|---|
| 回答跑偏 | 检索片段是否相关 |
| 回答缺信息 | 文档是否覆盖、top_k 是否太小 |
| 引用错误 | 片段编号是否和 Prompt 对齐 |
| 回答编造 | Prompt 是否限制“只基于资料” |
| 成本太高 | chunk 是否太长、召回是否太多 |

## 最小验收标准

完成后，你的知识库助手应该做到：

- 能加载一份 Markdown 文档。
- 能切成多个片段。
- 能根据问题找出相关片段。
- 能生成带引用回答。
- 能在资料不足时说明“不知道”。

## 下一步

把这个最小闭环接到真实向量数据库，例如 Chroma、Milvus、pgvector 或云端向量服务。等链路稳定后，再考虑查询重写、混合检索和 rerank。

