---
title: 日志与重试
published: 2026-07-09
description: 学会记录关键日志和设计简单重试，让模型 API 调用更稳定、更容易排查。
tags: [日志, 重试, API, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# 日志与重试

当模型调用失败时，你需要知道失败发生在哪里。日志负责记录事实，重试负责处理短暂故障。

## 基础日志

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("开始调用模型")
logger.warning("请求耗时较长")
logger.error("模型调用失败")
```

## LLM 项目应该记录什么

- 使用的模型。
- 请求开始时间。
- 请求耗时。
- 输入和输出 token。
- 错误状态码。
- 是否重试。

注意：不要把 API Key 和敏感用户数据写进日志。

## 简单重试

```python
import time

def retry(fn, times=3, delay=1):
    last_error = None
    for attempt in range(times):
        try:
            return fn()
        except Exception as error:
            last_error = error
            time.sleep(delay * (attempt + 1))
    raise last_error
```

重试适合处理临时网络错误、服务端 500、短暂限流。不要对 401 这类认证错误反复重试。

## 指数退避

```python
sleep_seconds = min(2 ** attempt, 30)
```

请求失败越多，等待越久，避免继续冲击服务。

## 常见错误

- 什么都不记录。
- 日志里打印完整密钥。
- 所有错误都无限重试。
- 没有记录请求耗时。
- 失败后只告诉用户“出错了”。

## 最小练习

写一个函数，模拟随机失败的 API 调用。给它加日志和最多 3 次重试，观察每次尝试的输出。

