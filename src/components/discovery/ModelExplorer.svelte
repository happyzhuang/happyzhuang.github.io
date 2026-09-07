<script lang="ts">
  import { models, scenarios, money, tokens, checkedAt, modelPath, comparePath, type Scenario } from "../../data/models";
  export let initialScenario: Scenario = "all";
  export let compact = false;
  let scenario = initialScenario;
  let search = "";
  let sort = "cost";
  let selected: string[] = [];
  $: filtered = models.filter(m => (scenario === "all" || m.scenarios.includes(scenario)) && `${m.name} ${m.provider}`.toLowerCase().includes(search.trim().toLowerCase()));
  $: sorted = [...filtered].sort((a,b) => sort === "context" ? b.context-a.context : sort === "input" ? a.input-b.input : a.input+a.output-b.input-b.output);
  function toggle(id: string) { selected = selected.includes(id) ? selected.filter(x => x !== id) : selected.length < 3 ? [...selected,id] : selected; }
</script>
<section class="model-explorer">
  <div class="discovery-tabs" aria-label="按任务筛选">{#each scenarios as s}<button class:active={scenario === s.id} aria-pressed={scenario === s.id} on:click={() => scenario = s.id}>{s.label}</button>{/each}</div>
  <div class="explorer-toolbar"><label class="model-search"><span aria-hidden="true">⌕</span><input aria-label="搜索模型或厂商" placeholder="搜索模型或厂商…" bind:value={search}/></label><label class="sort-label">排序<select bind:value={sort}><option value="cost">参考成本从低到高</option><option value="input">输入单价从低到高</option><option value="context">上下文从大到小</option></select></label></div>
  <div class="table-scroll"><table class="model-table"><caption class="sr-only">模型规格与 API 价格。美元 / 百万 Token，参考成本为一百万输入加一百万输出。</caption><thead><tr><th scope="col">序号</th><th scope="col">模型</th><th scope="col">输入 / 输出<small>USD / 1M Tokens</small></th><th scope="col">上下文上限</th><th scope="col">已核验能力</th><th scope="col">对比</th></tr></thead><tbody>{#each sorted as model,i}<tr><td class="rank-number">{String(i+1).padStart(2,"0")}</td><td><a class="model-identity" href={modelPath(model.id)}><span class="provider-mark {model.tone}">{model.mark}</span><span><strong>{model.name}</strong><small>{model.provider}</small></span></a></td><td class="price-cell"><strong>{money(model.input)}</strong><span> / {money(model.output)}</span></td><td>{tokens(model.context)}<small>{model.contextKind === "input" ? "输入上限" : "输入与输出共享"}</small></td><td><div class="capability-chips"><span>工具调用</span>{#if model.vision}<span>视觉输入</span>{:else}<span>文本输入</span>{/if}</div></td><td><button class="compare-toggle" class:chosen={selected.includes(model.id)} disabled={!selected.includes(model.id) && selected.length >= 3} aria-label={`${selected.includes(model.id) ? "移除" : "加入对比"} ${model.name}`} aria-pressed={selected.includes(model.id)} on:click={() => toggle(model.id)}>{selected.includes(model.id) ? "✓ 已选" : "+ 对比"}</button></td></tr>{/each}</tbody></table></div>
  {#if !sorted.length}<div class="discovery-empty"><h3>当前目录没有匹配的模型</h3><p>{scenario === "image" || scenario === "video" ? "图片与视频生成采用不同计费单位，暂未纳入这份文本 API 目录。可先查看独立评测来源。" : "试试其他关键词，或切换到全部模型。"}</p><a href="/benchmarks/">查看评测来源 →</a><button on:click={() => { scenario = "all"; search = ""; }}>重置筛选</button></div>{/if}
  <div class="table-footnote"><span>{sorted.length} 个模型 · 核验于 {checkedAt}</span><a href="/benchmarks/">数据与排序口径 ↗</a></div>
  <p class="data-note">按公开规格筛选，序号随排序变化，不代表质量名次。价格采用 Standard / 未缓存输入，DeepSeek 为高峰价；完整条件见模型详情。</p>
  {#if selected.length}<div class="compare-dock" aria-live="polite"><div><strong>对比清单 · {selected.length} / 3</strong><span>{selected.map(id => models.find(m => m.id === id)?.name).join(" · ")}</span></div><button on:click={() => selected = []}>清空</button>{#if selected.length >= 2}<a class="platform-button" href={comparePath(selected)}>开始对比 →</a>{:else}<span class="data-note">再选择一个模型</span>{/if}</div>{/if}
  {#if compact}<a class="explorer-more" href="/models/">浏览模型资料库 →</a>{/if}
</section>
