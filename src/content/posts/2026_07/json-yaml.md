---
title: JSON / YAML
published: 2026-07-09
description: 理解 JSON 和 YAML 在配置、结构化输出、模型调用和项目管理中的作用。
tags: [JSON, YAML, 配置, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# JSON / YAML

LLM 应用里，JSON 和 YAML 出现得非常频繁：API 请求、模型结构化输出、配置文件、评测数据、工具定义都会用到它们。

## JSON 适合什么

JSON 适合程序交换数据：

```json
{
  "model": "demo-model",
  "temperature": 0.3,
  "messages": [
    {
      "role": "user",
      "content": "解释 RAG"
    }
  ]
}
```

Python 读取：

```python
import json

data = json.loads('{"name": "LLM"}')
print(data["name"])
```

## YAML 适合什么

YAML 更适合人写配置：

```yaml
model:
  name: demo-model
  temperature: 0.3
rag:
  top_k: 3
  chunk_size: 500
```

读取 YAML 需要安装依赖：

```bash
pip install pyyaml
```

```python
import yaml

with open("config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
```

## 在 LLM 项目中的用途

| 格式 | 常见用途 |
|---|---|
| JSON | API 请求、结构化输出、工具参数 |
| YAML | 项目配置、模型配置、实验参数 |

## 注意事项

- JSON 字符串必须用双引号。
- YAML 对缩进敏感。
- 不要用 YAML 保存密钥，密钥放 `.env` 更好。
- 模型输出 JSON 后，程序要校验格式。

## 最小练习

写一个 `config.yaml`，包含模型名称、temperature、top_k。用 Python 读取后打印出来。

