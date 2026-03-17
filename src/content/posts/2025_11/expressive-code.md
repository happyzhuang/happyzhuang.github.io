---
title: 大模型系列——代码高亮与样式指南
published: 2025-11-05
description: 全面展示Markdown代码块的高级样式和功能，让技术文档更易读、更专业，打造优雅的代码展示体验。
tags: [大模型, Markdown, 代码高亮, 博客写作]
category: 写作指南
lang: zh_CN
draft: false
image: https://images.unsplash.com/photo-1555066931-4365d14bab8c?w=800&h=400&fit=crop
series: 大模型写作指南
---

# 大模型系列——代码高亮与样式指南

在技术博客中，清晰的代码展示是提升阅读体验的关键因素之一。一个优秀的代码块不仅要语法高亮准确，还应该具备良好的可读性和视觉美感。本文将全面展示如何使用 [Expressive Code](https://expressive-code.com/) 来美化代码块，让你的技术文档更加专业和易读。

> 💡 **提示**: 优秀的代码展示能让读者更快理解技术概念，提升博客的专业度和可读性。

## 🎨 语法高亮

### 基础语法高亮

最基本的代码高亮功能，支持多种编程语言：

```js
console.log('这是一个语法高亮的代码示例')
```

```python
def greet(name):
    return f"Hello, {name}!"
```

### ANSI 终端色彩渲染

支持渲染带有 ANSI 转义序列的终端输出，这对于展示命令行工具的输出特别有用：

```ansi
ANSI 颜色演示：
- 常规: [31m红色[0m [32m绿色[0m [33m黄色[0m [34m蓝色[0m [35m洋红[0m [36m青色[0m
- 加粗: [1;31m红色[0m [1;32m绿色[0m [1;33m黄色[0m [1;34m蓝色[0m [1;35m洋红[0m [1;36m青色[0m
- 变暗: [2;31m红色[0m [2;32m绿色[0m [2;33m黄色[0m [2;34m蓝色[0m [2;35m洋红[0m [2;36m青色[0m

256 色彩范围（160-177色）：
[38;5;160m160 [38;5;161m161 [38;5;162m162 [38;5;163m163 [38;5;164m164 [38;5;165m165[0m
[38;5;166m166 [38;5;167m167 [38;5;168m168 [38;5;169m169 [38;5;170m170 [38;5;171m171[0m

完整 RGB 色彩：
[38;2;34;139;34m森林绿 - RGB(34, 139, 34)[0m

文本格式： [1m加粗[0m [2m变暗[0m [3m斜体[0m [4m下划线[0m
```

## 📦 代码编辑器框架

### 标题栏显示

通过在代码块中添加文件名或标题，可以让读者更容易理解代码的上下文：

```js title="main.js"
console.log('这个代码块有标题栏显示文件名')
```

### 文件注释标题

另一种方式是在代码第一行添加文件路径注释：

```html
<!-- src/components/Header.vue -->
<div class="header">
  <h1>我的网站标题</h1>
</div>
```

### 终端窗口模拟

模拟真实的终端窗口外观：

```bash
echo "这是一个没有标题的终端窗口"
```

```powershell title="PowerShell 示例"
Write-Output "这是一个有标题的 PowerShell 终端"
```

### 框架类型自定义

可以强制指定框架类型，例如将 PowerShell 显示为代码编辑器：

```ps frame="code" title="PowerShell 配置文件.ps1"
# 这会被显示为代码编辑器而不是终端窗口
function Watch-Tail { Get-Content -Tail 20 -Wait $args }
New-Alias tail Watch-Tail
```

### 无框架模式

当不需要任何框架装饰时：

```sh frame="none"
echo "这个代码块没有框架装饰"
```

## 🔍 行标记与文本标记

### 行号标记

高亮特定行号或行范围：

```js {1, 4, 7-8}
// 第1行 - 被行号标记高亮
// 第2行
// 第3行
// 第4行 - 被行号标记高亮
// 第5行
// 第6行
// 第7行 - 被范围 "7-8" 标记
// 第8行 - 被范围 "7-8" 标记
```

### 标记类型选择

支持不同的标记类型：默认标记、插入、删除：

```js title="line-markers.js" del={2} ins={3-4} {6}
function demo() {
  console.log('这一行被标记为删除')
  // 这两行被标记为插入
  console.log('这是第二行插入的代码')

  return '这一行使用默认标记类型'
}
```

### 标记标签添加

为标记添加说明性标签：

```jsx {"1": 这是关键参数:5} del={"2": 需要删除:7-8} ins={"3": 新增代码:10-12}
// labeled-line-markers.jsx
<button
  role="button"
  {...props}
  value={value}
  className={buttonClassName}
  disabled={disabled}
  active={active}
>
  {children &&
    !active &&
    (typeof children === 'string' ? <span>{children}</span> : children)}
</button>
```

### 长标签独立行

当标签较长时，可以放在独立行上：

```jsx {"1. 在这里提供 value 属性:":5-6} del={"2. 删除 disabled 和 active 状态:":8-10} ins={"3. 在按钮内渲染 children:":12-15}
// labeled-line-markers.jsx
<button
  role="button"
  {...props}

  value={value}
  className={buttonClassName}

  disabled={disabled}
  active={active}
>

  {children &&
    !active &&
    (typeof children === 'string' ? <span>{children}</span> : children)}
</button>
```

### Diff 风格语法

支持类似 Git diff 的语法：

```diff
+这一行会被标记为插入
-这一行会被标记为删除
这是常规行
```

### 真实 Diff 文件

显示完整的 Git diff 输出：

```diff
--- a/README.md
+++ b/README.md
@@ -1,3 +1,4 @@
+这是实际的 diff 文件
-所有内容将保持不变
空白也不会被删除
```

### Diff 与语法高亮结合

在 diff 格式中保持语法高亮：

```diff lang="js"
  function 这是JavaScript代码() {
    // 整个代码块都会被高亮为 JavaScript，
    // 同时我们还能添加 diff 标记！
-   console.log('要删除的旧代码')
+   console.log('全新的闪亮代码！')
  }
```

### 行内文本标记

高亮特定文本内容：

```js "demo"
function demo() {
  // 匹配并标记所有包含 "demo" 的文本
  return 'Multiple matches of the given text are supported';
}
```

### 正则表达式匹配

使用正则表达式进行更灵活的文本匹配：

```ts /ye[sp]/
console.log('单词 yes 和 yep 都会被标记')
```

### 转义特殊字符

正确处理正则表达式中的特殊字符：

```sh /\/ho.*\//
echo "Test" > /home/test.txt
```

### 行内标记类型

为不同的文本指定不同的标记类型：

```js "return true;" ins="插入的文本" del="删除的文本"
function demo() {
  console.log('这些是插入和删除标记类型')
  // return 语句使用默认标记类型
  return true;
}
```

## 📏 自动换行控制

### 启用换行

```js wrap
// 启用自动换行的示例
function getLongString() {
  return '这是一个很长的字符串，如果不启用自动换行，在可用空间不足的情况下，除非容器极宽，否则可能无法正常显示'
}
```

### 禁用换行

```js wrap=false
// 禁用自动换行的示例
function getLongString() {
  return '这是一个很长的字符串，如果不启用自动换行，在可用空间不足的情况下，除非容器极宽，否则可能无法正常显示'
}
```

### 保持缩进

```js wrap preserveIndent
// 保持缩进的示例（默认启用）
function getLongString() {
  return '这是一个很长的字符串，如果不启用自动换行，在可用空间不足的情况下，除非容器极宽，否则可能无法正常显示'
}
```

### 不保留缩进

```js wrap preserveIndent=false
// 不保留缩进的示例
function getLongString() {
  return '这是一个很长的字符串，如果不启用自动换行，在可用空间不足的情况下，除非容器极宽，否则可能无法正常显示'
}
```

## 📁 可折叠代码段

### 指定折叠范围

通过指定行号范围来创建可折叠的代码段，这对于隐藏样板代码很有用：

```js collapse={1-5, 12-14, 21-24}
// 所有这些样板设置代码都会被折叠
import { someBoilerplateEngine } from '@example/some-boilerplate'
import { evenMoreBoilerplate } from '@example/even-more-boilerplate'

const engine = someBoilerplateEngine(evenMoreBoilerplate())

// 这部分代码默认可见
engine.doSomething(1, 2, 3, calcFn)

function calcFn() {
  // 你可以有多个折叠区域
  const a = 1
  const b = 2
  const c = a + b

  // 这部分保持可见
  console.log(`计算结果: ${a} + ${b} = ${c}`)
  return c
}

// 代码块结束前的所有代码都会被再次折叠
engine.closeConnection()
engine.freeMemory()
engine.shutdown({ reason: '示例样板代码结束' })
```

## 🔢 行号显示

### 启用行号

```js showLineNumbers
// 这个代码块会显示行号
console.log('来自第2行的问候！')
console.log('我在第3行')
```

### 禁用行号

```js showLineNumbers=false
// 这个代码块禁用了行号
console.log('你好？')
console.log('抱歉，你知道我在第几行吗？')
```

### 自定义起始行号

```js showLineNumbers startLineNumber=5
console.log('来自第5行的问候！')
console.log('我在第6行')
```

## 🎯 实用技巧总结

### 博客写作最佳实践

1. **选择合适的框架类型**: 
   - 普通代码使用代码编辑器框架
   - 命令行示例使用终端框架
   - 需要简洁展示时使用无框架模式

2. **有效使用行标记**:
   - 重点行使用默认标记
   - 代码变更使用 diff 风格
   - 为关键操作添加标签说明

3. **控制换行行为**:
   - 默认启用换行确保移动端可读性
   - 特定情况下禁用换行以保持代码结构
   - 长代码考虑使用可折叠段

4. **文件名和注释**:
   - 始终为代码块添加文件名
   - 使用注释说明代码的上下文
   - 保持代码块的完整性

### 常见应用场景

| 场景 | 推荐配置 | 效果 |
|------|----------|------|
| 普通代码示例 | 默认配置 | 清晰的语法高亮 |
| 命令行教程 | 终端框架 | 模拟真实终端体验 |
| 代码对比 | diff 标记 | 突出显示变更 |
| 长代码片段 | 可折叠段 | 保持界面简洁 |
| 关键行强调 | 行标记 | 引导读者关注重点 |

## 🔧 高级技巧

### 组合多种特性

```python title="数据分析脚本.py" {3,7-9} wrap collapse={1-2}
import pandas as pd
import numpy as np

# 加载数据集（这部分代码被折叠）
df = pd.read_csv('data.csv')
```

### 正则表达式精确匹配

```ts /console\.(log|warn|error)/
console.log('这行会被标记')
console.warn('这行也会被标记')
console.error('这行同样会被标记')
console.info('但这行不会被标记')
```

通过合理使用这些代码展示技巧，你可以创建既美观又实用的技术文档，为读者提供更好的阅读体验。

---

> 📚 **延伸阅读**: 想了解更多 Expressive Code 的高级功能，请访问 [Expressive Code 官方文档](https://expressive-code.com/)。
