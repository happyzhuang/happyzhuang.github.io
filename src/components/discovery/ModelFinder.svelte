<script lang="ts">
 import { onMount } from "svelte";
 import { models, scenarios, money, modelPath, comparePath, type Scenario } from "../../data/models";
 import { identifyScenario, candidates, validUsage } from "../../utils/model-discovery";
 let task = "分析 PDF 文档并提取关键信息";
 let scenario: Scenario = "document";
 let input = 20000;
 let output = 2000;
 let requests = 100;
 let budget = 100;
 let interpreted = false;
 function interpret() { scenario=identifyScenario(task); interpreted=true; }
 onMount(() => { const q = new URLSearchParams(location.search).get("q"); if(q) { task=q.slice(0,1000); interpret(); } });
 $: usage = { input, output, requests, cachedPercent:0, offPeak:false };
 $: valid = validUsage(usage) && Number.isFinite(budget) && budget >= 0;
 $: results = valid ? candidates(models,scenario,usage,budget) : [];
</script>
<div class="finder-layout"><section class="control-panel"><h2>你希望 AI 做什么？</h2><label>描述任务<textarea rows="3" bind:value={task} maxlength="1000" placeholder="例如：分析 100 份 PDF 财报" on:input={() => interpreted=false}></textarea></label><button class="platform-button" on:click={interpret}>识别任务场景 →</button>{#if interpreted}<p role="status" class="data-note">已按关键词识别为「{scenarios.find(s => s.id === scenario)?.label}」，你也可以手动调整。</p>{/if}<label>任务场景<select bind:value={scenario}>{#each scenarios.filter(s => s.id !== "all") as s}<option value={s.id}>{s.label}</option>{/each}</select></label><div class="two-inputs"><label>单次输入 Token<input type="number" min="0" step="100" bind:value={input}/></label><label>单次输出 Token<input type="number" min="0" step="100" bind:value={output}/></label></div><div class="two-inputs"><label>总请求数<input type="number" min="0" step="1" bind:value={requests}/></label><label>总预算（USD）<input type="number" min="0" step="1" bind:value={budget}/></label></div><p class="data-note">按公开能力、长度和预算筛选，候选按估算成本排序。任务文字仅在本机匹配关键词，没有发送给模型。文档长度需手动估计。</p></section>
<section class="finder-results"><div class="section-heading"><div><span class="eyebrow">YOUR SHORTLIST</span><h2>从这些候选开始测试</h2></div><span class="count-badge">{results.length} 个匹配</span></div>{#if !valid}<p role="alert" class="form-error">请输入非负数；Token 和请求数须为安全整数。</p>{:else if !results.length}<div class="discovery-empty"><h3>暂无满足条件的候选</h3><p>{scenario === "image" || scenario === "video" ? "目前收录文本输出模型。图片与视频生成需使用独立的模型目录和计价方式。" : "可以提高预算、减少单次输入，或调整任务场景。超出上下文或输出限制的模型已排除。"}</p><a href="/benchmarks/">查看其他评测来源 ↗</a></div>{:else}{#each results.slice(0,3) as r,i}<article class="finder-card"><div class="finder-card-top"><span class="provider-mark {r.model.tone}">{r.model.mark}</span><div><span class="eyebrow">候选 {i+1} · {r.model.provider}</span><h3><a href={modelPath(r.model.id)}>{r.model.name}</a></h3></div><strong>{money(r.cost.total)}<small>本批次 Token 估算</small></strong></div><p>{r.model.summary}</p><ul><li>符合「{scenarios.find(s => s.id === scenario)?.label}」场景标签。</li><li>单次长度在所列规格范围内，估算费用不超过 {money(budget)}。</li></ul><p class="candidate-caveat">测试前留意：{r.model.caveat}</p><div class="finder-card-actions"><a href={modelPath(r.model.id)}>查看完整资料 →</a><a href={r.model.tryUrl} target="_blank" rel="noreferrer">官方平台 ↗</a></div></article>{/each}{#if results.length >= 2}<a class="platform-button" href={comparePath(results.slice(0,3).map(r => r.model.id))}>并排对比这些候选 →</a>{/if}{/if}<p class="data-note">这是基于规则的候选清单，不是质量评测。联网检索、OCR、缓存写入、存储等额外费用未计入，最终效果需要用真实任务验证。</p></section></div>
