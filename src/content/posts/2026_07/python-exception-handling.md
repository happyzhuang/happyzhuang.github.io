---
title: 异常处理
published: 2026-07-09
description: 理解 try/except、错误分类和失败恢复，让 LLM 应用在 API 调用失败时不至于直接崩溃。
tags: [Python, 异常处理, API, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# 异常处理

大模型应用经常会失败：网络超时、API Key 错误、额度不足、模型返回格式不对、文件不存在。异常处理的目标不是掩盖错误，而是让程序知道“哪里失败了，下一步怎么办”。

## 基本写法

```python
try:
    result = risky_operation()
except Exception as error:
    print(f"执行失败：{error}")
```

这能捕获错误，但太粗糙。更好的做法是按错误类型处理。

## 文件读取示例

```python
from pathlib import Path

def read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在：{path}")
    except UnicodeDecodeError:
        raise ValueError(f"文件编码不是 UTF-8：{path}")
```

这里区分了“文件不存在”和“编码错误”，排查会清楚很多。

## API 调用中的异常

模型 API 常见错误包括：

| 错误 | 处理 |
|---|---|
| 401 | 检查 API Key |
| 429 | 降低请求频率或重试 |
| 400 | 检查请求参数 |
| 超时 | 设置 timeout 并重试 |
| 返回格式错误 | 做 JSON 校验和兜底 |

## 自定义异常

当项目变大时，可以定义自己的异常：

```python
class ModelCallError(Exception):
    pass

def call_model(prompt: str) -> str:
    try:
        # 调用模型
        return "模型回答"
    except TimeoutError as error:
        raise ModelCallError("模型调用超时") from error
```

这样上层代码只需要处理 `ModelCallError`。

## 不要吞掉错误

不推荐：

```python
try:
    call_model(prompt)
except Exception:
    pass
```

这会让失败悄悄发生，后面更难排查。

## 最小练习

写一个 `safe_read_text(path)`：

- 文件存在时返回内容。
- 文件不存在时返回友好的错误信息。
- 编码错误时提示用户检查文件格式。

这是后续 RAG 文档读取的基础。

