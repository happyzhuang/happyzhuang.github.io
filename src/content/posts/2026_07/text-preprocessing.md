---
title: 文本预处理
published: 2026-07-09
description: 学会清理文本噪声、规范空白和保留结构，为 Prompt、RAG 和数据处理打基础。
tags: [文本预处理, RAG, Prompt, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# 文本预处理

大模型应用很依赖输入质量。文档里如果有大量噪声、重复空白、无关导航、乱码和残缺表格，模型输出也会受到影响。

文本预处理的目标是：保留有用信息，去掉干扰信息。

## 常见清洗动作

### 去掉多余空白

```python
import re

def normalize_space(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
```

### 去掉重复行

```python
def remove_duplicate_lines(text: str) -> str:
    seen = set()
    lines = []
    for line in text.splitlines():
        if line not in seen:
            lines.append(line)
            seen.add(line)
    return "\n".join(lines)
```

### 保留标题结构

Markdown 标题很重要，不要随便删：

```markdown
# 一级标题
## 二级标题
正文内容
```

标题能帮助后续切块和引用。

## RAG 中的预处理原则

- 删除页眉页脚。
- 删除导航和版权噪声。
- 保留标题、列表和表格。
- 保留文档来源。
- 不要过度清洗导致语义丢失。

## Prompt 中的预处理

把输入材料包起来：

```text
材料：
"""
{cleaned_text}
"""
```

这样模型更容易区分任务说明和材料本身。

## 最小练习

找一段带大量空行和重复内容的文本，写一个清洗函数：

1. 合并多余空行。
2. 去掉首尾空白。
3. 保留 Markdown 标题。
4. 输出清洗前后长度。

