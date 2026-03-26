---
title: OpenClaw 系列（二）——手把手部署，从零到第一个自动化任务
published: 2026-03-15
description: 详细的 OpenClaw 安装部署教程，覆盖 Windows / macOS / Linux 三平台，从环境准备、模型配置到跑通第一个自动化任务，全程图文说明。
tags: [大模型, Agent, OpenClaw, 部署教程, AI自动化]
category: OpenClaw
lang: zh_CN
draft: false
---

# OpenClaw 系列（二）——手把手部署，从零到第一个自动化任务

上一篇我们聊清楚了 OpenClaw 是什么、它的工作原理是什么。这一篇直接进入实战：**把它安装起来，配置好，跑通第一个真实任务**。

全文覆盖 Windows、macOS、Linux 三个平台，你用哪个照着哪个来就行。

## 一、部署前的准备：搞清楚你需要什么

在动手之前，先梳理清楚两个核心问题：

### 1.1 选择模型来源：云端 API 还是本地部署？

OpenClaw 本身不包含大模型，它只是一个框架——需要你提供一个能"思考"的大脑。目前有两种选择：

| 方式 | 代表服务 | 优点 | 缺点 |
|------|----------|------|------|
| **云端 API** | Claude、GPT-4、Gemini、Moonshot、DeepSeek | 无需硬件、效果好、开箱即用 | 按 Token 付费，有数据隐私顾虑 |
| **本地部署** | Ollama + Llama3 / Qwen2.5 / Mistral | 完全私密、一次投入无后续费用 | 需要较好的 GPU，配置略复杂 |

**推荐策略**：初次上手，建议先用云端 API（成本低、效果好）。如果你有隐私需求或长期重度使用，再考虑本地模型。

国内用户友好的云端 API 选项：
- **DeepSeek**：性价比极高，支持 OpenAI 兼容接口
- **月之暗面（Moonshot / Kimi）**：中文理解强
- **智谱 AI（GLM-4）**：同样支持 OpenAI 兼容接口
- **阿里云 DashScope（Qwen）**：文档能力强

### 1.2 系统环境要求

| 要求 | 说明 |
|------|------|
| 操作系统 | Windows 10/11、macOS 12+、Ubuntu 20.04+ |
| Node.js | **v20 或更高版本**（必须） |
| 内存 | 最低 4GB，推荐 8GB+ |
| 网络 | 稳定网络（调用云端 API 需要）|

## 二、安装 Node.js

OpenClaw 基于 Node.js 运行，这是安装的第一步。

### macOS / Linux

推荐使用 `nvm`（Node 版本管理器）安装，可以避免权限问题：

```bash
# 安装 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 重启终端后，安装 Node.js 20
nvm install 20
nvm use 20

# 验证安装
node --version   # 应输出 v20.x.x
npm --version
```

### Windows

前往 [Node.js 官网](https://nodejs.org) 下载 LTS 版本安装包（选 v20.x），一路下一步安装完成。

安装后打开 PowerShell 验证：

```powershell
node --version
npm --version
```

## 三、安装 OpenClaw

Node.js 就绪后，安装 OpenClaw 只需一行命令：

```bash
npm install -g openclaw@latest
```

安装完成后验证：

```bash
openclaw --version
```

如果输出了版本号（如 `3.8.x`），说明安装成功。

> **Windows 用户注意**：如果提示权限错误，请以管理员身份运行 PowerShell，或参考 [npm 官方文档](https://docs.npmjs.com/resolving-eacces-permissions-errors-when-installing-packages-globally) 修复全局安装权限问题。

## 四、初始化 OpenClaw

安装完成后，运行初始化向导：

```bash
openclaw onboard --install-daemon
```

这个命令会引导你完成以下步骤：

1. **初始化配置文件**：在 `~/.openclaw/` 目录下创建 `openclaw.json`
2. **安装后台守护进程**：让 OpenClaw Gateway 开机自动运行（可选）
3. **基础环境检查**：检测 Node.js 版本、依赖完整性等

初始化成功后，你会看到类似这样的输出：

```
✓ 配置目录已创建：~/.openclaw/
✓ 守护进程已注册
✓ OpenClaw 初始化完成，使用 openclaw gateway 启动服务
```

## 五、配置大模型 API

打开配置文件：

```bash
# macOS / Linux
nano ~/.openclaw/openclaw.json

# Windows（用记事本）
notepad %USERPROFILE%\.openclaw\openclaw.json
```

配置文件的基本结构如下：

```json
{
  "model": {
    "provider": "openai",
    "apiKey": "你的API密钥",
    "baseURL": "https://api.deepseek.com/v1",
    "model": "deepseek-chat"
  },
  "channels": {
    "telegram": {
      "botToken": "你的Telegram Bot Token"
    }
  }
}
```

### 5.1 使用 DeepSeek（国内推荐）

DeepSeek 兼容 OpenAI 接口，配置非常简单。前往 [platform.deepseek.com](https://platform.deepseek.com) 注册并获取 API Key，然后填入配置：

```json
{
  "model": {
    "provider": "openai",
    "apiKey": "sk-xxxxxxxxxxxxxxxx",
    "baseURL": "https://api.deepseek.com/v1",
    "model": "deepseek-chat"
  }
}
```

### 5.2 使用 Claude（效果最佳）

OpenClaw 原生支持 Anthropic Claude，任务执行效果目前综合评价最优：

```json
{
  "model": {
    "provider": "anthropic",
    "apiKey": "sk-ant-xxxxxxxxxxxxxxxx",
    "model": "claude-sonnet-4-5"
  }
}
```

### 5.3 使用本地 Ollama 模型（完全离线）

先安装 [Ollama](https://ollama.com) 并下载模型：

```bash
# 下载 Qwen2.5 7B（推荐，中文支持好）
ollama pull qwen2.5:7b
```

然后在 OpenClaw 配置中指向本地 Ollama：

```json
{
  "model": {
    "provider": "ollama",
    "baseURL": "http://localhost:11434",
    "model": "qwen2.5:7b"
  }
}
```

> **注意**：本地小参数模型（7B 以下）在复杂多步任务中的表现明显弱于云端大模型，推荐 14B 或以上规格用于生产环境。

## 六、配置消息渠道（以 Telegram 为例）

OpenClaw 需要一个消息渠道来接收你的指令。Telegram 是最容易配置的选择，我们以它为例。

### 步骤 1：创建 Telegram Bot

1. 在 Telegram 中搜索 `@BotFather`，发送 `/newbot`
2. 按提示给 Bot 取名（如 `MyClaw Bot`）和用户名（如 `myclawbot`）
3. BotFather 会给你一个 **Bot Token**，类似：`7123456789:AAHxxxxxxxxxxxxxxx`

### 步骤 2：填入配置

```json
{
  "model": {
    "provider": "openai",
    "apiKey": "你的DeepSeek API Key",
    "baseURL": "https://api.deepseek.com/v1",
    "model": "deepseek-chat"
  },
  "channels": {
    "telegram": {
      "botToken": "7123456789:AAHxxxxxxxxxxxxxxx",
      "allowFrom": ["你的Telegram用户ID"]
    }
  }
}
```

> **如何查找自己的 Telegram 用户 ID**：搜索 `@userinfobot`，发送任意消息，它会回复你的数字 ID。

`allowFrom` 字段是访问控制白名单——**强烈建议填写**，防止任何人都能向你的 Agent 下达指令。

## 七、启动 OpenClaw Gateway

配置完成后，启动网关服务：

```bash
openclaw gateway --port 18789
```

看到以下输出说明启动成功：

```
🦞 OpenClaw Gateway v3.8.x
✓ 配置加载完成
✓ Telegram 渠道已连接
✓ Agent 就绪
🚀 Web 控制台：http://127.0.0.1:18789/
```

现在打开浏览器访问 `http://127.0.0.1:18789/`，你会看到 OpenClaw 的 Web 控制台界面，可以在这里查看会话历史、管理配置、监控任务状态。

## 八、跑通第一个任务

万事俱备。打开 Telegram，找到你刚创建的 Bot，发送第一条指令：

```
在桌面创建一个名为"测试文件夹"的目录，然后在里面新建一个 hello.txt 文件，写入"Hello from OpenClaw!"
```

OpenClaw 收到指令后，会开始执行：

1. 理解任务：需要创建目录 + 创建文件 + 写入内容
2. 调用 `exec` 工具执行 `mkdir` 命令
3. 调用 `write` 工具写入文件内容
4. 反馈执行结果

成功后，你会在 Telegram 中收到一条确认消息，并在桌面看到新创建的文件夹和文件。

**恭喜，你的第一个 AI Agent 任务跑通了！**

## 九、常见问题排查

**Q：启动后 Telegram 没有响应怎么办？**
检查 Bot Token 是否正确，以及网络是否能访问 Telegram 服务器。国内用户可能需要配置代理。

**Q：模型返回错误 "API Key invalid"**
仔细检查 API Key 是否有多余的空格或换行，`baseURL` 是否以 `/v1` 结尾。

**Q：执行文件操作时报权限错误**
在 macOS 上，需要在「系统设置 → 隐私与安全性 → 文件和文件夹」中给予终端应用相应权限。Windows 用户确保以管理员权限运行。

**Q：执行结果和预期不符**
尝试把指令写得更具体、更明确。例如把"整理文件"改成"把桌面上所有 .pdf 文件移动到 ~/Documents/PDFs 文件夹"。

---

## 小结

到这里，你已经完成了：
- Node.js 环境准备
- OpenClaw 安装与初始化
- 大模型 API 配置（以 DeepSeek 为例）
- Telegram 渠道接入
- 跑通第一个文件操作任务

下一篇，我们进入进阶内容：**如何安装和管理 Skills 扩展、如何为不同任务配置不同的模型、以及如何做好安全加固，让你的"小龙虾"既好用又安全**。

> 本文是「OpenClaw 系列」的第二篇：
> - 第一篇：[OpenClaw 系列（一）——它不只是聊天，它是你的数字员工](/posts/openclaw-intro)
> - 第三篇：OpenClaw 进阶玩法——Skills 扩展、模型配置与安全加固
