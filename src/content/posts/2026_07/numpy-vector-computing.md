---
title: Numpy 向量计算
published: 2026-07-09
description: 用 Numpy 理解向量、相似度和基础矩阵计算，为 Embedding 和 RAG 检索打基础。
tags: [Numpy, 向量计算, Embedding, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# Numpy 向量计算

Embedding 的本质是把文本变成向量。理解一点 Numpy 向量计算，会让你更容易看懂 RAG、相似度检索和模型评测。

## 向量是什么

向量可以理解为一串数字：

```python
import numpy as np

a = np.array([0.2, 0.4, 0.8])
b = np.array([0.1, 0.5, 0.7])
```

在 RAG 中，一段文本会被 Embedding 模型转成更长的向量。

## 向量长度

```python
norm = np.linalg.norm(a)
print(norm)
```

长度本身不是最重要的，RAG 中更常用的是方向相似度。

## 余弦相似度

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

score = cosine_similarity(a, b)
print(score)
```

分数越接近 1，说明两个向量方向越接近。

## 模拟检索

```python
query = np.array([0.2, 0.4, 0.8])
docs = [
    ("doc1", np.array([0.1, 0.5, 0.7])),
    ("doc2", np.array([0.9, 0.1, 0.2])),
]

ranked = sorted(
    docs,
    key=lambda item: cosine_similarity(query, item[1]),
    reverse=True,
)

print(ranked[0][0])
```

这就是向量检索的极简版本。

## 常见用途

- 计算文本相似度。
- 找最相关文档。
- 对模型输出做聚类。
- 做简单排序和过滤。

## 最小练习

手动创建 3 个“文档向量”和 1 个“问题向量”，计算余弦相似度，找出最相似的文档。

