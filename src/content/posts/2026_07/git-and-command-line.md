---
title: Git 与命令行
published: 2026-07-09
description: 掌握最常用的命令行和 Git 操作，让 LLM 项目可以被记录、回滚和部署。
tags: [Git, 命令行, 工程基础, 学习地图]
category: Python 与数据基础
lang: zh_CN
draft: false
---

# Git 与命令行

LLM 项目也需要工程化。命令行让你运行脚本、安装依赖、查看日志；Git 让你记录变更、回滚错误、协作开发。

## 常用命令行操作

```bash
pwd          # 查看当前目录
ls           # 查看文件
cd folder    # 进入目录
mkdir demo   # 创建目录
```

Windows PowerShell 中：

```powershell
Get-ChildItem
Set-Location .\demo
```

## Git 最小流程

```bash
git status
git add .
git commit -m "Add hello llm script"
git push
```

每次完成一个小功能就提交，不要等到所有东西都写完。

## 查看改动

```bash
git diff
git diff --stat
```

提交前先看改了什么，是很好的习惯。

## LLM 项目特别注意

不要提交：

- `.env`
- API Key
- 用户私有数据
- 大型临时文件
- 向量数据库缓存

`.gitignore` 示例：

```text
.env
.venv/
__pycache__/
data/private/
vector_store/
```

## 常见错误

- 不看 `git status` 就提交。
- 把密钥提交到 GitHub。
- 一个提交包含太多不相关改动。
- 不写有意义的 commit message。

## 最小练习

创建一个小项目，写一个 `README.md`，然后完成一次：

1. `git status`
2. `git add`
3. `git commit`
4. `git log --oneline`

