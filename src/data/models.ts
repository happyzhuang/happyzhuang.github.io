export type Scenario = "all" | "chat" | "coding" | "agent" | "document" | "rag" | "reasoning" | "search" | "image" | "video";
export type Model = {
  id: string; name: string; provider: string; mark: string; tone: string;
  input: number; output: number; cached: number;
  context: number; contextKind: "shared" | "input"; maxOutput: number;
  vision: boolean; tools: boolean; search: boolean; scenarios: Scenario[];
  summary: string; caveat: string; pricingNote: string;
  source: string; specs: string; tryUrl: string;
  longThreshold?: number; longInput?: number; longOutput?: number; longCached?: number;
  offPeak?: boolean;
};

export const checkedAt = "2026-09-07";
export const scenarios: { id: Scenario; label: string; en: string }[] = [
  { id: "all", label: "全部模型", en: "All models" },
  { id: "chat", label: "对话写作", en: "Chat" },
  { id: "coding", label: "代码开发", en: "Coding" },
  { id: "agent", label: "Agent", en: "Agent" },
  { id: "document", label: "文档理解", en: "Document" },
  { id: "rag", label: "RAG 问答", en: "RAG" },
  { id: "reasoning", label: "推理", en: "Reasoning" },
  { id: "search", label: "联网检索", en: "Search" },
  { id: "image", label: "图片生成", en: "Image" },
  { id: "video", label: "视频生成", en: "Video" },
];

export const models: Model[] = [
  {
    id: "gpt-6-astra", name: "GPT-6 Astra", provider: "OpenAI", mark: "O", tone: "green",
    input: 10, output: 50, cached: 1, context: 1050000, contextKind: "shared", maxOutput: 128000,
    longThreshold: 272000, longInput: 20, longOutput: 75, longCached: 2,
    vision: true, tools: true, search: true, scenarios: ["rag", "chat", "coding", "agent", "document", "reasoning", "search"],
    summary: "可用于复杂推理、代码与多步骤工具任务。",
    caveat: "单价较高；复杂任务应先用自己的样本比较质量收益。",
    pricingNote: "Standard；单次输入超过 272K 时整次请求使用长上下文价格。",
    source: "https://developers.openai.com/api/docs/pricing",
    specs: "https://developers.openai.com/api/docs/models/gpt-6-astra",
    tryUrl: "https://platform.openai.com/",
  },
  {
    id: "claude-sonnet-5", name: "Claude Sonnet 5", provider: "Anthropic", mark: "A", tone: "orange",
    input: 2, output: 10, cached: .2, context: 1000000, contextKind: "shared", maxOutput: 128000,
    vision: true, tools: true, search: false, scenarios: ["rag", "chat", "coding", "agent", "document", "reasoning"],
    summary: "兼顾文本、视觉输入与工具调用的通用候选。",
    caveat: "新分词器可能改变相同文本的 Token 数，需用真实 usage 校准。",
    pricingNote: "Claude API 标准价；缓存写入另计。",
    source: "https://platform.claude.com/docs/en/about-claude/pricing",
    specs: "https://platform.claude.com/docs/en/models/overview",
    tryUrl: "https://platform.claude.com/",
  },
  {
    id: "claude-haiku-4-5", name: "Claude Haiku 4.5", provider: "Anthropic", mark: "A", tone: "orange",
    input: 1, output: 5, cached: .1, context: 200000, contextKind: "shared", maxOutput: 64000,
    vision: true, tools: true, search: false, scenarios: ["rag", "chat", "coding", "agent", "document", "reasoning"],
    summary: "较低单价的 Claude 选项，可用于日常文本与工具任务。",
    caveat: "上下文为 200K；长文档可能需要分块。本站未测量响应速度。",
    pricingNote: "Claude API 标准价；缓存写入另计。",
    source: "https://platform.claude.com/docs/en/about-claude/pricing",
    specs: "https://platform.claude.com/docs/en/models/overview",
    tryUrl: "https://platform.claude.com/",
  },
  {
    id: "gemini-3-8-flash", name: "Gemini 3.8 Flash", provider: "Google", mark: "G", tone: "blue",
    input: .75, output: 3.75, cached: .075, context: 1048576, contextKind: "input", maxOutput: 65536,
    vision: true, tools: true, search: true, scenarios: ["rag", "chat", "coding", "agent", "document", "reasoning", "search"],
    summary: "支持 PDF、图片、音频和视频输入，以及联网检索工具。",
    caveat: "视频输入不等于视频生成；搜索和缓存存储费用需另计。",
    pricingNote: "Standard 付费层，当前优惠至 2026-12-31；2027 年起输入 $1.50、输出 $7.50。",
    source: "https://ai.google.dev/gemini-api/docs/pricing#gemini-3.8-flash",
    specs: "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",
    tryUrl: "https://aistudio.google.com/",
  },
  {
    id: "deepseek-v4-flash", name: "DeepSeek V4 Flash", provider: "DeepSeek", mark: "D", tone: "violet",
    input: .44, output: 1.32, cached: .014, context: 1000000, contextKind: "shared", maxOutput: 384000, offPeak: true,
    vision: false, tools: true, search: false, scenarios: ["rag", "chat", "coding", "agent", "reasoning"],
    summary: "支持思考模式与工具调用，可作为低成本文本候选。",
    caveat: "此条目不是 Vision 实验版；图片和扫描 PDF 需先转为文本。",
    pricingNote: "按高峰价展示；非高峰价格减半。周一至周五 UTC 01–04、06–10 为高峰。",
    source: "https://api-docs.deepseek.com/quick_start/pricing/",
    specs: "https://api-docs.deepseek.com/quick_start/pricing/",
    tryUrl: "https://platform.deepseek.com/",
  },
  {
    id: "deepseek-v4-pro", name: "DeepSeek V4 Pro", provider: "DeepSeek", mark: "D", tone: "violet",
    input: 1.32, output: 3.96, cached: .044, context: 1000000, contextKind: "shared", maxOutput: 384000, offPeak: true,
    vision: false, tools: true, search: false, scenarios: ["rag", "chat", "coding", "agent", "reasoning"],
    summary: "V4 系列的 Pro 选项，提供思考模式、JSON 输出与工具调用。",
    caveat: "Pro 名称不代表在你的任务上必然更好，建议与 Flash 做对照测试。",
    pricingNote: "按高峰价展示；非高峰价格减半。周一至周五 UTC 01–04、06–10 为高峰。",
    source: "https://api-docs.deepseek.com/quick_start/pricing/",
    specs: "https://api-docs.deepseek.com/quick_start/pricing/",
    tryUrl: "https://platform.deepseek.com/",
  },
];

export const modelPath = (id: string) => `/models/${id}/`;
export const money = (n: number) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 4 }).format(n);
export const tokens = (n: number) => n >= 1000000 ? `${+(n / 1000000).toFixed(3)}M` : `${+(n / 1000).toFixed(1)}K`;
export const comparePath = (ids: string[]) => `/compare/?models=${ids.map(encodeURIComponent).join(",")}`;
