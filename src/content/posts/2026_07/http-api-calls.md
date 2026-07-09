---
title: HTTP API 调用
published: 2026-07-09
description: 理解 GET、POST、Header、JSON Body 和状态码，为调用模型 API 做准备。
tags: [HTTP, API, Python, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# HTTP API 调用

模型 API 本质上也是 HTTP API。你把请求发给服务器，服务器返回 JSON 或流式数据。理解 HTTP，是写 LLM 应用的基础。

## GET 和 POST

- GET：获取资源。
- POST：提交数据。

模型调用通常使用 POST，因为你要提交 Prompt、参数和上下文。

## Header

Header 用来放认证和内容类型：

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
```

## JSON Body

```python
payload = {
    "model": "demo-model",
    "messages": [
        {"role": "user", "content": "解释 RAG"}
    ],
}
```

## 使用 requests

```python
import requests

response = requests.post(
    "https://api.example.com/v1/chat/completions",
    headers=headers,
    json=payload,
    timeout=30,
)

response.raise_for_status()
data = response.json()
```

## 状态码

| 状态码 | 含义 |
|---|---|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 认证失败 |
| 429 | 限流或额度不足 |
| 500 | 服务端错误 |

## 最小练习

用 `requests` 调用一个公开测试 API，打印状态码和 JSON 返回。理解这个流程后，再换成模型 API。

