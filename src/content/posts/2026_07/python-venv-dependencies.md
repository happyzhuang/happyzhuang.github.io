---
title: 虚拟环境与依赖
published: 2026-07-09
description: 学会用虚拟环境隔离 Python 项目依赖，避免 RAG、Agent 项目出现版本混乱。
tags: [Python, 虚拟环境, 依赖管理, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# 虚拟环境与依赖

不同项目需要不同依赖。RAG 可能需要向量库，Agent 可能需要工具框架，数据处理可能需要 Pandas。如果所有包都装到全局环境，迟早会版本冲突。

虚拟环境的作用，就是给每个项目一个独立的 Python 包空间。

## 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux 激活：

```bash
source .venv/bin/activate
```

激活后，命令行前面通常会出现 `(.venv)`。

## 安装依赖

```bash
pip install requests python-dotenv
```

保存依赖：

```bash
pip freeze > requirements.txt
```

别人拿到项目后可以安装：

```bash
pip install -r requirements.txt
```

## LLM 项目常见依赖

| 依赖 | 用途 |
|---|---|
| requests/httpx | HTTP API 调用 |
| python-dotenv | 读取 `.env` |
| pandas | 数据清洗 |
| numpy | 向量计算 |
| pydantic | 数据结构校验 |

## .gitignore

不要提交虚拟环境：

```text
.venv/
__pycache__/
.env
```

## 常见错误

- 忘记激活虚拟环境就安装依赖。
- 把 `.venv` 提交到 Git。
- 不保存 `requirements.txt`。
- 多个项目共用全局依赖导致版本冲突。

## 最小练习

创建一个 `hello-python` 项目：

1. 创建 `.venv`。
2. 安装 `python-dotenv`。
3. 写一个读取 `.env` 的脚本。
4. 生成 `requirements.txt`。

