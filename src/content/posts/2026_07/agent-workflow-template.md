---
title: Agent 工程样板：任务规划、工具调用与失败恢复
published: 2026-07-09
description: 通过一个可复用 Agent 样板，理解任务规划、工具定义、执行循环、观察日志和失败恢复。
tags: [Agent, 工具调用, Workflow, 实战案例]
category: 实战案例
lang: zh_CN
draft: false
---

# Agent 工程样板：任务规划、工具调用与失败恢复

Agent 不是“让模型自由发挥”。一个可靠 Agent 应该有明确目标、可控工具、执行边界、日志和失败恢复机制。

这个案例会搭一个最小 Agent 工作流。

## Agent 最小结构

```text
用户目标
  ↓
任务规划
  ↓
选择工具
  ↓
执行工具
  ↓
观察结果
  ↓
反思修正
  ↓
最终输出
```

## 项目结构

```text
agent-template/
├─ src/
│  ├─ agent.ts
│  ├─ tools.ts
│  ├─ planner.ts
│  └─ logger.ts
└─ .env
```

## 第一步：定义目标

Agent 的输入不要太泛。好的目标应该包含任务、限制和成功标准。

```text
目标：整理三篇关于 RAG 的笔记，输出一份学习计划。
限制：只使用本地文档，不要联网搜索。
成功标准：输出 5 个学习主题、每个主题的摘要和下一步练习。
```

## 第二步：定义工具

工具要小而清楚。

```ts
type Tool = {
  name: string;
  description: string;
  run: (input: Record<string, unknown>) => Promise<string>;
};

const tools: Tool[] = [
  {
    name: "search_notes",
    description: "在本地笔记中搜索相关内容。",
    run: async ({ query }) => {
      return `和 ${query} 相关的笔记片段...`;
    },
  },
];
```

工具描述要说明：

- 能做什么。
- 什么时候用。
- 输入参数是什么。
- 不应该用来做什么。

## 第三步：让模型先规划

不要让 Agent 一上来就执行。先让它输出计划：

```text
你是一个谨慎的任务规划 Agent。

目标：
{goal}

可用工具：
{tools}

请先输出：
1. 任务拆解
2. 每一步需要的工具
3. 成功标准
4. 可能失败点
5. 是否需要用户确认
```

## 第四步：执行循环

最小循环可以这样写：

```ts
for (let step = 0; step < maxSteps; step++) {
  const action = await chooseNextAction(goal, history, tools);

  if (action.type === "final") {
    return action.answer;
  }

  const tool = tools.find((item) => item.name === action.toolName);
  if (!tool) throw new Error(`未知工具：${action.toolName}`);

  const observation = await tool.run(action.input);
  history.push({ action, observation });
}

throw new Error("超过最大执行步数");
```

## 第五步：记录执行日志

Agent 必须可观察。每一步都记录：

- 当前目标。
- 模型选择的工具。
- 工具输入。
- 工具输出。
- 模型下一步判断。

日志不是装饰，它是排查 Agent 失控的核心工具。

## 第六步：失败恢复

常见失败包括：

| 失败 | 处理方式 |
|---|---|
| 重复调用同一工具 | 检测重复 action，要求换策略 |
| 工具返回空 | 让模型缩小或改写查询 |
| 工具报错 | 返回错误信息，让模型选择备用方案 |
| 步数过多 | 设置 maxSteps，要求汇总已有结果 |
| 权限过大 | 工具权限最小化，只开放必要路径 |

## 最小验收标准

这个 Agent 样板完成后，应该做到：

- 能接收一个目标。
- 能先规划再执行。
- 至少有一个可调用工具。
- 每一步都有日志。
- 有最大步数限制。
- 工具失败时能给出解释。

## 下一步

把这个样板接入真实工具：本地文件搜索、网页搜索、数据库查询或代码执行。每加一个工具，都要重新评估权限边界。

