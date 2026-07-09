---
title: Hello LLM：第一次调用模型 API
published: 2026-07-09
description: 从 API Key、请求结构、流式输出、错误处理到成本估算，完成第一个可复用的大模型调用样板。
tags: [LLM, API, 实战案例, Hello LLM]
category: 实战案例
lang: zh_CN
draft: false
---

# Hello LLM：第一次调用模型 API

这是本站第一个实战案例。目标很简单：完成一次可复用的大模型 API 调用，并把后续 RAG、Agent、模型切换都会用到的工程习惯先建立起来。

这一节不追求复杂框架，只关注最小闭环：

1. 安全保存 API Key。
2. 组织一次模型请求。
3. 拿到模型回答。
4. 支持流式输出。
5. 处理常见错误。
6. 估算一次调用成本。

## 你会得到什么

完成后，你应该能得到一个最小项目骨架：

```text
hello-llm/
├─ .env
├─ .gitignore
├─ package.json
└─ src/
   ├─ basic.ts
   ├─ stream.ts
   └─ errors.ts
```

这个骨架后面可以继续扩展成 Prompt 工具、RAG 问答或 Agent 工作流。

## 第一步：准备 API Key

API Key 不要写进代码。推荐放到 `.env`：

```text
MODEL_API_KEY=你的模型平台密钥
MODEL_BASE_URL=https://api.example.com/v1
MODEL_NAME=your-model-name
```

`.gitignore` 至少要包含：

```text
.env
node_modules
dist
```

如果你还不熟悉 API Key，可以先看学习地图里的《API Key 与调用额度》。

## 第二步：理解一次请求

一次模型 API 调用通常包含四类信息：

| 字段 | 作用 |
|---|---|
| model | 使用哪个模型 |
| messages/input | 给模型的任务和上下文 |
| temperature | 控制输出随机性 |
| max_tokens | 控制输出长度 |

最小消息结构可以这样理解：

```json
[
  {
    "role": "system",
    "content": "你是一个简洁、准确的 AI 应用开发助手。"
  },
  {
    "role": "user",
    "content": "请用三句话解释什么是 RAG。"
  }
]
```

`system` 负责长期规则，`user` 负责当前任务。后续做多轮对话时，还会把历史 `assistant` 回答放回上下文。

## 第三步：最小调用脚本

下面是一个平台无关的 TypeScript 写法。你可以把 `MODEL_BASE_URL` 换成具体模型服务的 OpenAI-compatible 地址。

```ts
const apiKey = process.env.MODEL_API_KEY;
const baseUrl = process.env.MODEL_BASE_URL;
const model = process.env.MODEL_NAME;

if (!apiKey || !baseUrl || !model) {
  throw new Error("缺少 MODEL_API_KEY、MODEL_BASE_URL 或 MODEL_NAME");
}

const response = await fetch(`${baseUrl}/chat/completions`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  },
  body: JSON.stringify({
    model,
    messages: [
      {
        role: "system",
        content: "你是一个简洁、准确的 AI 应用开发助手。",
      },
      {
        role: "user",
        content: "请用三句话解释什么是 RAG。",
      },
    ],
    temperature: 0.3,
  }),
});

if (!response.ok) {
  const errorText = await response.text();
  throw new Error(`模型调用失败：${response.status} ${errorText}`);
}

const data = await response.json();
console.log(data.choices?.[0]?.message?.content);
```

这段代码有三个重点：

- 密钥从环境变量读取。
- 请求失败时读取错误信息。
- 输出只取模型回答正文。

## 第四步：加入流式输出

流式输出能让用户更快看到结果，适合聊天、长文生成和代码生成场景。

```ts
const response = await fetch(`${baseUrl}/chat/completions`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  },
  body: JSON.stringify({
    model,
    stream: true,
    messages: [
      { role: "system", content: "你是一个简洁、准确的 AI 应用开发助手。" },
      { role: "user", content: "请解释一次模型 API 调用的完整流程。" },
    ],
  }),
});

if (!response.body) {
  throw new Error("当前环境不支持流式响应");
}

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  process.stdout.write(chunk);
}
```

真实项目中，流式响应通常还需要解析 Server-Sent Events。初学阶段先理解“边接收边显示”的模式即可。

## 第五步：处理常见错误

大模型 API 常见错误可以先分成五类：

| 错误 | 可能原因 | 处理方式 |
|---|---|---|
| 401 | API Key 错误或过期 | 检查环境变量，重新生成 Key |
| 429 | 触发限流或额度不足 | 降低频率，增加重试和排队 |
| 400 | 请求格式错误 | 检查 model、messages、参数类型 |
| 413 | 输入过长 | 压缩上下文，减少历史或资料 |
| 500/503 | 服务端异常 | 稍后重试，切换备用模型 |

推荐封装一个统一错误处理函数：

```ts
function explainApiError(status: number) {
  if (status === 401) return "API Key 无效或权限不足";
  if (status === 429) return "触发限流或额度不足";
  if (status === 400) return "请求参数格式错误";
  if (status === 413) return "输入内容过长";
  if (status >= 500) return "模型服务暂时异常";
  return "未知错误";
}
```

## 第六步：估算成本

一次调用成本可以粗略拆成：

```text
总成本 = 输入 token * 输入单价 + 输出 token * 输出单价
```

开发阶段至少要记录：

- 用户输入长度。
- 系统提示词长度。
- 检索资料长度。
- 模型输出长度。
- 本次使用的模型。

如果模型服务返回 usage 字段，可以直接记录：

```json
{
  "usage": {
    "prompt_tokens": 320,
    "completion_tokens": 180,
    "total_tokens": 500
  }
}
```

你也可以把典型 Prompt 放到本站实验室的 Token 成本估算器里，先估算大致成本。

## 最小可复用封装

最后，把模型调用封装成函数：

```ts
type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export async function callModel(messages: ChatMessage[]) {
  const response = await fetch(`${process.env.MODEL_BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.MODEL_API_KEY}`,
    },
    body: JSON.stringify({
      model: process.env.MODEL_NAME,
      messages,
      temperature: 0.3,
    }),
  });

  if (!response.ok) {
    throw new Error(explainApiError(response.status));
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content ?? "";
}
```

后续所有案例都可以基于这个函数扩展：

- Prompt 工具箱：替换 messages。
- RAG 知识库：把检索片段加入 messages。
- Agent 工作流：把工具结果加入 messages。
- 模型对比：切换 `MODEL_NAME`。

## 检查清单

完成这个案例后，确认你已经做到：

- API Key 没有写进代码。
- `.env` 不会被提交。
- 最小请求能正常返回回答。
- 错误状态能被解释。
- 知道输入和输出都会产生 token 成本。
- 模型调用逻辑已经封装成函数。

## 下一步

下一篇实战建议做“Prompt 工具箱”：把不同任务的 Prompt 模板管理起来，并和实验室里的 Prompt 对比器形成联动。

