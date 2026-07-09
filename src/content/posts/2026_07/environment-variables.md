---
title: 环境变量
published: 2026-07-09
description: 学会用环境变量管理 API Key、模型名称和配置，避免密钥泄露。
tags: [环境变量, API Key, 配置, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# 环境变量

环境变量用于保存不同环境下的配置，比如 API Key、模型名称、Base URL、数据库地址。它的核心价值是：配置和代码分离。

## 为什么不要写死配置

不推荐：

```python
api_key = "sk-xxxx"
```

原因：

- 容易提交到 Git。
- 不方便切换开发和生产环境。
- 多人协作不安全。

## 使用 .env

`.env` 文件：

```text
MODEL_API_KEY=你的密钥
MODEL_NAME=demo-model
MODEL_BASE_URL=https://api.example.com/v1
```

安装：

```bash
pip install python-dotenv
```

读取：

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MODEL_API_KEY")
model = os.getenv("MODEL_NAME")
```

## 必须加入 .gitignore

```text
.env
```

可以提交 `.env.example`：

```text
MODEL_API_KEY=
MODEL_NAME=
MODEL_BASE_URL=
```

## 常见错误

- `.env` 被提交到仓库。
- 环境变量名拼错。
- 读取后没有检查是否为空。
- 前端项目暴露了服务端密钥。

## 最小练习

创建 `.env` 和 `.env.example`，写一个 Python 脚本读取 `MODEL_NAME` 并打印。如果变量不存在，给出明确错误。

