---
title: OpenClaw 系列（三）——进阶玩法：Skills 扩展、多模型配置与安全加固
published: 2026-03-16
description: 深入 OpenClaw 的 Skills 扩展生态、多模型灵活配置策略、上下文管理机制，以及让 Agent 安全运行的关键配置实践。
tags: [大模型, Agent, OpenClaw, Skills, AI安全, 进阶配置]
category: OpenClaw
lang: zh_CN
draft: false
---

# OpenClaw 系列（三）——进阶玩法：Skills 扩展、多模型配置与安全加固

经过前两篇的铺垫，你的 OpenClaw 应该已经跑起来了。这一篇我们进入进阶领域，聊三件事：**怎么让它更能干（Skills 扩展）、怎么让它更省钱（多模型配置）、怎么让它更安全（安全加固）**。

## 一、Skills 系统：给 Agent 装上各种技能

如果说 OpenClaw 的 Agent Loop 是躯体，那么 **Skills（技能）** 就是它的职业技能包。Skills 本质上是一套预设的工具集合和指令模板，告诉 Agent：面对这类任务，你有哪些特殊工具可用、该遵循什么操作规范。

### 1.1 内置 Skills 一览

OpenClaw 3.8 自带 52+ 个内置 Skills，覆盖了日常使用的主要场景：

| 类别 | 代表 Skills | 功能 |
|------|-------------|------|
| 开发工具 | `github`、`git`、`npm` | Git 操作、GitHub API 调用、包管理 |
| 系统操作 | `filesystem`、`terminal` | 文件读写、命令执行 |
| 网络搜索 | `web-search`、`brave-search` | 联网检索、获取实时信息 |
| 办公自动化 | `calendar`、`email` | 日历管理、邮件收发 |
| 媒体处理 | `image`、`pdf` | 图片处理、PDF 解析 |
| 数据处理 | `csv`、`json-tools` | 结构化数据操作 |

查看所有可用 Skills：

```bash
openclaw skills list
```

### 1.2 安装社区 Skills

除内置技能外，OpenClaw 社区还提供了大量扩展 Skills。安装一个社区 Skill 只需：

```bash
# 从官方注册表安装
openclaw skills install <skill-name>

# 例：安装飞书集成 Skill
openclaw skills install feishu

# 安装后查看已安装列表
openclaw skills list --installed
```

目前热门的社区 Skills 包括：
- **feishu / dingtalk**：飞书、钉钉消息收发与 Webhook 集成
- **notion**：Notion 数据库读写与页面管理
- **browser**：无头浏览器操作，可填写表单、截图、爬取动态页面
- **docker**：容器管理操作
- **database**：SQLite、PostgreSQL、MySQL 查询

### 1.3 自定义 Skill：15 分钟上手

Skills 的核心是一个 `SKILL.md` 文件，格式非常简洁：

```markdown
---
name: my-custom-skill
description: "我的自定义技能，用于处理某类特定任务"
metadata:
  openclaw:
    emoji: "🔧"
    requires:
      bins: ["python3"]    # 依赖的系统命令
    install:
      - label: "说明文字"
---

# My Custom Skill

这里写给 Agent 看的操作指南，用自然语言描述：
- 当用户需要做什么时，应该怎么操作
- 有哪些注意事项
- 推荐的工作流程
```

把这个文件放入 `~/.openclaw/skills/my-custom-skill/SKILL.md`，重启 Gateway，Agent 就能自动识别并在合适的任务中调用这个 Skill。

## 二、多模型配置策略：聪明地花钱

不同任务对模型能力的要求差异巨大：一个简单的文件重命名任务，用 Claude Sonnet 和用 DeepSeek 差别不大；但一个需要深度推理的代码审查任务，模型质量的差距会非常明显。

OpenClaw 支持按任务类型配置不同的模型，做到既省钱又不牺牲关键任务的质量。

### 2.1 配置主模型与备用模型

```json
{
  "model": {
    "primary": {
      "provider": "anthropic",
      "apiKey": "sk-ant-xxxxxxx",
      "model": "claude-sonnet-4-5"
    },
    "fallback": {
      "provider": "openai",
      "apiKey": "sk-xxxxxxx",
      "baseURL": "https://api.deepseek.com/v1",
      "model": "deepseek-chat"
    }
  }
}
```

当主模型调用失败（如 API 限流、服务异常）时，OpenClaw 会自动切换到备用模型，保证 Agent 的可用性。

### 2.2 按任务类型路由模型

OpenClaw 3.x 引入了模型路由机制，可以根据任务复杂度自动选择模型：

```json
{
  "model": {
    "routing": {
      "simple": {
        "provider": "openai",
        "baseURL": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
      },
      "complex": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5"
      }
    },
    "complexityThreshold": 0.7
  }
}
```

`complexityThreshold` 控制切换阈值，数值越低越容易升级到强模型。你可以根据自己的使用场景和预算做调整。

### 2.3 结合本地模型降低成本

对于高频但低复杂度的任务（如文件整理、信息格式化、日常问答），可以优先路由到本地 Ollama 模型：

```json
{
  "model": {
    "routing": {
      "simple": {
        "provider": "ollama",
        "baseURL": "http://localhost:11434",
        "model": "qwen2.5:7b"
      },
      "complex": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5"
      }
    }
  }
}
```

这种"本地 + 云端混合"的方式，在实践中可以将 Token 费用降低 60%~80%，同时对复杂任务保持高质量输出。

## 三、上下文管理：避免长任务中的"失忆"

当任务复杂、步骤多时，对话历史会不断积累，导致上下文超出模型的 Token 限制。OpenClaw 内置了一套上下文管理机制，理解它有助于设计更稳健的任务流程。

### 3.1 软裁剪与硬清除

OpenClaw 采用两阶段的上下文保护策略：

- **软裁剪**（Soft Trim）：当上下文达到模型限制的 30% 时，保留最近的对话内容，压缩较早的历史记录
- **硬清除**（Hard Reset）：当上下文达到 50% 时，自动生成历史摘要，用摘要替换原始对话，释放空间

这意味着 OpenClaw 在执行非常长的任务时，会自动"总结压缩"历史，避免失控。但这也意味着：**对于超长任务，早期的细节可能会被压缩丢失**。

### 3.2 实践建议

对于需要保留全程细节的重要任务，建议在指令中明确要求：

```
【重要】请在每完成一个子任务时，输出一段结构化的进度摘要，包含：已完成步骤、当前状态、下一步计划。
```

这样即使上下文被压缩，关键的结构化信息也会保留在摘要中，不会丢失。

### 3.3 利用长期记忆

OpenClaw 有一套基于向量检索的长期记忆系统，会自动保存重要对话和用户偏好。你也可以主动让 Agent 记住某些信息：

```
记住：我习惯把所有下载文件整理到 ~/Documents/Downloads-Sorted/ 目录，并按年月分类。
```

下次执行相关任务时，Agent 会自动从记忆中检索这条偏好，无需重复说明。

## 四、安全加固：让你的"小龙虾"不失控

由于 OpenClaw 拥有操作文件、执行命令等高权限能力，安全配置是必须认真对待的环节。工信部在 2026 年初曾专门发布安全提示，建议用户在部署时注意以下几点。

### 4.1 严格的访问控制白名单

这是最基础也最重要的一步。在配置文件中，**一定要填写 `allowFrom` 白名单**，只允许你自己的账号下达指令：

```json
{
  "channels": {
    "telegram": {
      "botToken": "你的Token",
      "allowFrom": ["你的数字用户ID"]
    },
    "whatsapp": {
      "allowFrom": ["+861xxxxxxxxxx"]
    }
  }
}
```

没有白名单的情况下，任何知道你 Bot 地址的人都可以向你的 Agent 下达指令——这是极其危险的。

### 4.2 关闭不必要的公网暴露

默认情况下，OpenClaw Gateway 只监听本地回环地址（`127.0.0.1`），这是安全的。**不要**为了远程访问方便，将其直接绑定到 `0.0.0.0` 或公网 IP。

正确的远程访问方式是使用 Tailscale（一种零配置 VPN）建立私有网络通道：

```bash
# 安装 Tailscale 后，通过私有 IP 访问
openclaw gateway --port 18789 --host 100.x.x.x  # Tailscale 分配的私有 IP
```

这样就算你在外面，也可以安全地访问家里的 OpenClaw，而不需要暴露公网端口。

### 4.3 限制 Agent 的文件系统访问范围

可以在配置中设置工作目录白名单，防止 Agent 误操作不该碰的系统文件：

```json
{
  "agent": {
    "workspace": "~/OpenClaw-Workspace",
    "allowedPaths": [
      "~/Documents",
      "~/Desktop",
      "~/OpenClaw-Workspace"
    ],
    "blockedPaths": [
      "~/.ssh",
      "~/.aws",
      "/etc",
      "/System"
    ]
  }
}
```

`blockedPaths` 中的目录，Agent 无法读写，有效防止敏感凭证泄露。

### 4.4 启用 ACP 身份溯源（3.8 新特性）

OpenClaw 3.8 新增了 ACP（Agent Communication Protocol）身份溯源机制，对于多 Agent 协作场景尤为重要。它为每个 Agent 会话附加可验证的签名来源和 Trace ID，形成完整的可审计链路：

```json
{
  "acp": {
    "provenance": "meta+receipt"
  }
}
```

- `off`：默认关闭，不影响现有行为
- `meta`：每个会话携带来源元数据和 Trace ID
- `meta+receipt`：额外注入可见回执，在对话中形成完整审计记录

如果你的 OpenClaw 被用于自动化工作流或多 Agent 场景，建议至少开启 `meta` 级别。

### 4.5 定期检查日志与异常行为

OpenClaw 会记录所有工具调用日志，存放在 `~/.openclaw/logs/` 目录。养成定期检查的习惯：

```bash
# 查看最近的操作日志
tail -f ~/.openclaw/logs/gateway.log

# 查看工具调用记录
cat ~/.openclaw/logs/tool-calls.jsonl | tail -50
```

如果发现异常的命令执行记录（特别是你不记得下达过的指令），立即排查来源。

## 五、实用配置模板汇总

把上述所有最佳实践整合在一起，这是一份适合大多数用户的"安全生产级"配置模板：

```json
{
  "model": {
    "routing": {
      "simple": {
        "provider": "openai",
        "apiKey": "你的DeepSeek Key",
        "baseURL": "https://api.deepseek.com/v1",
        "model": "deepseek-chat"
      },
      "complex": {
        "provider": "anthropic",
        "apiKey": "你的Claude Key",
        "model": "claude-sonnet-4-5"
      }
    },
    "complexityThreshold": 0.6
  },
  "channels": {
    "telegram": {
      "botToken": "你的Bot Token",
      "allowFrom": ["你的Telegram用户ID"]
    }
  },
  "agent": {
    "workspace": "~/OpenClaw-Workspace",
    "allowedPaths": ["~/Documents", "~/Desktop", "~/OpenClaw-Workspace"],
    "blockedPaths": ["~/.ssh", "~/.aws", "~/.config"]
  },
  "acp": {
    "provenance": "meta"
  },
  "gateway": {
    "host": "127.0.0.1",
    "port": 18789
  }
}
```

## 六、小结：成为合格的"虾农"

用好 OpenClaw，本质上是在学习如何与一个有能力但需要引导的"数字员工"协作。几条实践经验：

**指令要具体**：越清晰的指令，执行效果越好。"整理文件"不如"把桌面所有 .xlsx 文件按创建日期归档到 ~/Documents/Reports/YYYY-MM/ 目录"。

**复杂任务分步下达**：与其一次发送一个超长任务描述，不如拆成几个清晰的子步骤，逐一确认执行结果后再进行下一步。

**善用 Skills**：在执行特定类型任务前，先告诉 Agent 激活对应的 Skill，可以显著提升成功率。

**定期维护记忆**：Agent 的长期记忆会随时间积累。定期整理，删除过时的偏好记录，保持记忆库的准确性。

---

到这里，OpenClaw 系列三篇文章就全部完成了。从理解 Agent 的本质，到完成第一次部署，再到进阶的扩展配置和安全实践，希望这个系列能帮你真正把这只"小龙虾"驯服，为你所用。

欢迎在评论区分享你的使用心得——你都让 OpenClaw 帮你自动化了哪些任务？

> 本文是「OpenClaw 系列」的第三篇：
> - 第一篇：[OpenClaw 系列（一）——它不只是聊天，它是你的数字员工](/posts/openclaw-intro)
> - 第二篇：[OpenClaw 系列（二）——手把手部署，从零到第一个自动化任务](/posts/openclaw-deploy)
