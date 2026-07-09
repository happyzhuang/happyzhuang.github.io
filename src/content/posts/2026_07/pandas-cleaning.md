---
title: Pandas 清洗
published: 2026-07-09
description: 学会用 Pandas 读取、清洗和筛选表格数据，为模型评测、日志分析和数据整理做准备。
tags: [Pandas, 数据清洗, 评测, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# Pandas 清洗

很多 LLM 项目都会产生表格数据：评测集、用户反馈、调用日志、模型对比结果。Pandas 是处理这些数据的常用工具。

## 读取 CSV

```python
import pandas as pd

df = pd.read_csv("logs.csv")
print(df.head())
```

## 常见清洗动作

### 删除空值

```python
df = df.dropna(subset=["question", "answer"])
```

### 填充默认值

```python
df["status"] = df["status"].fillna("unknown")
```

### 筛选数据

```python
failed = df[df["status"] == "failed"]
```

### 新增列

```python
df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
```

## LLM 日志分析示例

假设有调用日志：

```csv
model,input_tokens,output_tokens,status
model-a,300,120,success
model-b,800,200,failed
```

可以统计平均 token：

```python
summary = df.groupby("model")["total_tokens"].mean()
print(summary)
```

## 常见错误

- CSV 编码不一致。
- 列名有空格。
- 数字列被读成字符串。
- 没有保留原始数据备份。
- 清洗过程不可复现。

## 最小练习

创建一个 `logs.csv`，包含 `model`、`input_tokens`、`output_tokens`、`status` 四列。用 Pandas 统计每个模型的平均总 token。

