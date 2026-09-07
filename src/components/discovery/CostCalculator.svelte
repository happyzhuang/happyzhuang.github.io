<script lang="ts">
 import { models, money, checkedAt, modelPath } from "../../data/models";
 import { defaultUsage, estimate, validUsage } from "../../utils/model-discovery";
 let input = defaultUsage.input;
 let output = defaultUsage.output;
 let requests = defaultUsage.requests;
 let cachedPercent = 0;
 let offPeak = false;
 $: usage = { input, output, requests, cachedPercent, offPeak };
 $: valid = validUsage(usage);
 $: results = valid ? models.map(model => ({model, cost:estimate(model,usage)!})).sort((a,b) => a.cost.total-b.cost.total) : [];
 $: best = results.find(r => r.cost.fits);
 function preset(i:number,o:number,r:number) { input=i;output=o;requests=r;cachedPercent=0;offPeak=false; }
</script>
<div class="calculator-layout"><section class="control-panel"><h2>你的调用规模</h2><div class="preset-buttons"><button on:click={() => preset(2000,500,10000)}>聊天应用</button><button on:click={() => preset(20000,2000,100)}>100 份文档</button><button on:click={() => preset(10000,2000,100000)}>高频 API</button></div><label>每次输入 Token<input type="number" min="0" step="100" bind:value={input}/></label><label>每次输出 Token（含推理）<input type="number" min="0" step="100" bind:value={output}/></label><label>每月请求数<input type="number" min="0" step="1" bind:value={requests}/></label><label>输入缓存命中比例 · {cachedPercent ?? 0}%<input type="range" min="0" max="100" step="5" bind:value={cachedPercent}/></label><label class="checkbox-label"><input type="checkbox" bind:checked={offPeak}/>DeepSeek 全部按非高峰计算</label><p class="data-note">预设仅用于演算，文档长度需自行估计。这里的 Token 已包含历史消息、检索文本和工具结果。</p></section>
<section class="cost-results"><div class="cost-summary"><span>符合长度限制的最低估算</span><strong>{valid && best ? money(best.cost.total) : "—"}<small> / 月</small></strong><p>{valid && best ? best.model.name : "请检查输入及模型长度限制"}</p></div>
{#if !valid}<p role="alert" class="form-error">Token 和请求数须为非负安全整数，缓存比例应在 0–100% 之间。</p>{:else}<div class="table-scroll"><table class="model-table"><caption class="sr-only">每月文本 Token 费用对比</caption><thead><tr><th>模型</th><th>输入费用</th><th>输出费用</th><th>月度总计</th></tr></thead><tbody>{#each results as r}<tr><td><a href={modelPath(r.model.id)}>{r.model.name}</a><small>{!r.cost.fits ? "超出单次长度限制" : r.cost.long ? "长上下文价格" : "标准 Token 计费"}</small></td><td>{money(r.cost.inputCost)}</td><td>{money(r.cost.outputCost)}</td><td><strong>{r.cost.fits ? money(r.cost.total) : "不适用"}</strong></td></tr>{/each}</tbody></table></div>{/if}
<div class="formula-note"><strong>计算方式</strong><p>月费用 = 请求数 ×（未缓存输入 × 输入单价 + 缓存输入 × 命中单价 + 输出 × 输出单价）÷ 1,000,000。</p><p>仅估算 Token 费用，不含首次缓存写入、缓存存储、搜索与其他工具、图片生成、税费或地区加价。不同模型分词器不同，相同文本未必使用相同 Token 数。</p></div><p class="data-note">价格核验：{checkedAt}。<a href="/benchmarks/">查看官方价格来源及有效期 →</a></p></section></div>
