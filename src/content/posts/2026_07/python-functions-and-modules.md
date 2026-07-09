---
title: 函数与模块
published: 2026-07-09
description: 学会用函数和模块组织 Python 代码，为后续 LLM API、RAG 和 Agent 项目打基础。
tags: [Python, 函数, 模块, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# 函数与模块

写大模型应用时，很快就会遇到重复逻辑：读取配置、调用模型、切分文本、记录日志、解析结果。如果所有代码都堆在一个文件里，项目会很快失控。

函数和模块的作用，就是把代码拆成可理解、可复用、可测试的小单元。

## 函数解决什么问题

函数适合封装一个清晰动作：

- 读取环境变量。
- 调用模型 API。
- 清洗一段文本。
- 统计 token 数量。
- 构造 Prompt。
- 解析模型返回。

好的函数应该做到：输入明确、输出明确、副作用尽量少。

## 一个最小示例

```python
def build_prompt(topic: str, audience: str) -> str:
    return f"""
请解释：{topic}

目标读者：{audience}
要求：
1. 先给一句话定义。
2. 再解释核心机制。
3. 最后给一个最小示例。
"""

prompt = build_prompt("RAG", "刚开始学习大模型应用的开发者")
print(prompt)
```

这里 `build_prompt` 只负责构造 Prompt，不负责调用模型。职责越单一，后面越容易维护。

## 模块是什么

模块就是一个 Python 文件。比如：

```text
hello-llm/
├─ main.py
├─ prompts.py
└─ model_client.py
```

`prompts.py` 放 Prompt 构造逻辑：

```python
def build_explain_prompt(topic: str) -> str:
    return f"请用三句话解释：{topic}"
```

`main.py` 中导入使用：

```python
from prompts import build_explain_prompt

prompt = build_explain_prompt("Embedding")
```

## LLM 项目中的推荐拆法

```text
src/
├─ config.py        # 环境变量和配置
├─ prompts.py       # Prompt 模板
├─ model_client.py  # 模型调用
├─ documents.py     # 文档读取和切块
└─ main.py          # 入口
```

这个结构足够支撑 Hello LLM、RAG 和简单 Agent。

## 常见错误

- 一个函数做太多事情。
- 函数名模糊，比如 `handle()`、`process()`。
- 模块之间互相导入，形成循环依赖。
- 把 API Key 写在函数里。
- 所有逻辑都放在 `main.py`。

## 最小练习

把下面三个动作拆成三个函数：

1. 构造解释概念的 Prompt。
2. 模拟调用模型并返回字符串。
3. 把回答保存到文件。

目标不是写复杂代码，而是练习“每个函数只做一件事”。

