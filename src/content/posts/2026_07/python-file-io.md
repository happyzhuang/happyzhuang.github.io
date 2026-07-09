---
title: 文件读写
published: 2026-07-09
description: 掌握 Python 文件读取、写入、路径处理和编码习惯，为文档处理和 RAG 做准备。
tags: [Python, 文件读写, RAG, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# 文件读写

RAG、Prompt 工具箱、项目日志整理都离不开文件读写。你需要能稳定读取 Markdown、保存模型输出、遍历文档目录，并处理编码问题。

## 使用 pathlib

推荐使用 `pathlib`：

```python
from pathlib import Path

path = Path("docs/note.md")
text = path.read_text(encoding="utf-8")
print(text)
```

相比字符串路径，`Path` 更适合跨平台处理。

## 写入文件

```python
from pathlib import Path

output = Path("outputs/answer.md")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("# 模型回答\n\n这里是回答内容。", encoding="utf-8")
```

注意先创建目录，否则写入会失败。

## 遍历目录

```python
from pathlib import Path

docs_dir = Path("docs")

for path in docs_dir.glob("*.md"):
    text = path.read_text(encoding="utf-8")
    print(path.name, len(text))
```

如果要递归读取子目录：

```python
for path in docs_dir.rglob("*.md"):
    print(path)
```

## RAG 中的文件记录

读取文档时要保留来源：

```python
def load_documents(folder: str):
    records = []
    for path in Path(folder).rglob("*.md"):
        records.append({
            "source": str(path),
            "content": path.read_text(encoding="utf-8"),
        })
    return records
```

后续生成引用时，`source` 很重要。

## 常见错误

- 忘记指定 `encoding="utf-8"`。
- 用相对路径时不知道程序从哪里运行。
- 写入文件前没有创建目录。
- 读取二进制文件时当文本处理。
- 没保留文档来源，导致 RAG 无法引用。

## 最小练习

创建一个 `docs` 目录，放两篇 Markdown，然后写脚本：

1. 遍历所有 `.md` 文件。
2. 读取内容。
3. 输出文件名和字数。
4. 把结果保存到 `outputs/report.md`。

