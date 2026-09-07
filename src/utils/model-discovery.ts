import type { Model, Scenario } from "../data/models";

export type Usage = { input: number; output: number; requests: number; cachedPercent: number; offPeak: boolean };
export const defaultUsage: Usage = { input: 2000, output: 500, requests: 10000, cachedPercent: 0, offPeak: false };
export function validUsage(u: Usage) {
  return [u.input, u.output, u.requests, u.cachedPercent].every(Number.isFinite) &&
    u.input >= 0 && u.output >= 0 && u.requests >= 0 && [u.input,u.output,u.requests].every(Number.isSafeInteger) &&
    u.cachedPercent >= 0 && u.cachedPercent <= 100;
}
export function estimate(model: Model, usage: Usage) {
  if (!validUsage(usage)) return null;
  const long = !!model.longThreshold && usage.input > model.longThreshold;
  const inputRate = long ? model.longInput! : model.input;
  const outputRate = long ? model.longOutput! : model.output;
  const cachedRate = long ? model.longCached! : model.cached;
  const multiplier = usage.offPeak && model.offPeak ? .5 : 1;
  const inputCost = usage.input * ((1 - usage.cachedPercent / 100) * inputRate + usage.cachedPercent / 100 * cachedRate) / 1000000 * usage.requests * multiplier;
  const outputCost = usage.output * outputRate / 1000000 * usage.requests * multiplier;
  const fits = usage.output <= model.maxOutput && (model.contextKind === "input" ? usage.input <= model.context : usage.input + usage.output <= model.context);
  return { inputCost, outputCost, total: inputCost + outputCost, long, fits };
}
export function identifyScenario(text: string): Scenario {
  if (/视频生成|生成视频|video generat/i.test(text)) return "video";
  if (/图片生成|生成图片|画图|绘图|文生图|image generat/i.test(text)) return "image";
  if (/rag|知识库|检索增强/i.test(text)) return "rag";
  if (/pdf|文档|财报|扫描|图片|截图/i.test(text)) return "document";
  if (/搜索|联网|检索最新|search/i.test(text)) return "search";
  if (/agent|智能体|工具调用|自动化/i.test(text)) return "agent";
  if (/代码|编程|程序|bug|code|coding/i.test(text)) return "coding";
  if (/推理|数学|reason/i.test(text)) return "reasoning";
  return "chat";
}
export function candidates(models: Model[], scenario: Scenario, usage: Usage, maxBudget: number) {
  if (!validUsage(usage) || !Number.isFinite(maxBudget) || maxBudget < 0) return [];
  return models.filter(m => scenario === "all" || m.scenarios.includes(scenario))
    .map(model => ({ model, cost: estimate(model, usage)! }))
    .filter(r => r.cost.fits && r.cost.total <= maxBudget)
    .sort((a,b) => a.cost.total - b.cost.total || a.model.name.localeCompare(b.model.name));
}
