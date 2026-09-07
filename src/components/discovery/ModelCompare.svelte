<script lang="ts">
 import { onMount } from "svelte";
 import { models, modelPath, money, tokens, comparePath, checkedAt } from "../../data/models";
 export let initialIds = ["gpt-6-astra","claude-sonnet-5","gemini-3-8-flash"];
 let ids = [...initialIds];
 let copied = "";
 $: chosen = ids.map(id => models.find(m => m.id === id)).filter(Boolean);
 onMount(() => { const supplied = new URLSearchParams(location.search).get("models"); if (supplied) { const valid = [...new Set(supplied.split(","))].filter(id => models.some(m => m.id === id)).slice(0,3); if (valid.length >= 2) ids = valid; } });
 function select(index:number,value:string) { ids = ids.map((id,i) => i === index ? value : id); copied=""; }
 async function share() { try { await navigator.clipboard.writeText(new URL(comparePath(ids),location.origin).href); copied="链接已复制"; } catch { copied="复制失败，可使用下方链接打开后复制地址"; } }
</script>
<div class="compare-controls"><p>选择 2–3 个具体模型版本</p><div class="compare-selects">{#each ids as id,i}<label>模型 {i+1}<select value={id} on:change={e => select(i,e.currentTarget.value)}>{#each models as m}<option value={m.id} disabled={ids.includes(m.id) && m.id !== id}>{m.name}</option>{/each}</select></label>{/each}</div><div class="compare-actions"><button on:click={() => ids = ids.length === 3 ? ids.slice(0,2) : [...ids,models.find(m => !ids.includes(m.id))!.id]}>{ids.length === 3 ? "移除第三个模型" : "+ 添加第三个模型"}</button><button on:click={share}>复制对比链接 ↗</button><a href={comparePath(ids)}>打开当前组合</a><span role="status">{copied}</span></div></div>
<div class="table-scroll comparison-scroll"><table class="model-table comparison-table"><caption class="sr-only">所选模型的价格、规格与能力对比</caption><thead><tr><th>对比维度</th>{#each chosen as m}{#if m}<th><span class="provider-mark {m.tone}">{m.mark}</span><a href={modelPath(m.id)}>{m.name}</a><small>{m.provider}</small></th>{/if}{/each}</tr></thead><tbody>
<tr><th>输入 / 百万 Token</th>{#each chosen as m}<td>{m && money(m.input)}</td>{/each}</tr>
<tr><th>输出 / 百万 Token</th>{#each chosen as m}<td>{m && money(m.output)}</td>{/each}</tr>
<tr><th>缓存命中 / 百万 Token</th>{#each chosen as m}<td>{m && money(m.cached)}</td>{/each}</tr>
<tr><th>上下文上限</th>{#each chosen as m}<td>{m && tokens(m.context)}<small>{m?.contextKind === "input" ? "输入上限" : "输入与输出共享"}</small></td>{/each}</tr>
<tr><th>最大输出</th>{#each chosen as m}<td>{m && tokens(m.maxOutput)}</td>{/each}</tr>
<tr><th>视觉输入</th>{#each chosen as m}<td>{m?.vision ? "支持" : "此版本未收录"}</td>{/each}</tr>
<tr><th>工具调用</th>{#each chosen as m}<td>{m?.tools ? "支持" : "未核验"}</td>{/each}</tr>
<tr><th>联网检索工具</th>{#each chosen as m}<td>{m?.search ? "文档已确认 · 费用另计" : "本次未核验"}</td>{/each}</tr>
<tr><th>速度 / 质量评测</th>{#each chosen as m}<td>本站尚未独立实测</td>{/each}</tr>
<tr><th>选型提示</th>{#each chosen as m}<td>{m?.summary}<small>{m?.caveat}</small></td>{/each}</tr>
<tr><th>价格条件</th>{#each chosen as m}<td>{m?.pricingNote}</td>{/each}</tr>
<tr><th>官方来源 · {checkedAt}</th>{#each chosen as m}<td><a href={m?.source} target="_blank" rel="noreferrer">价格 ↗</a> · <a href={m?.specs} target="_blank" rel="noreferrer">规格 ↗</a></td>{/each}</tr>
<tr><th>下一步</th>{#each chosen as m}<td><a class="platform-button secondary" href={m?.tryUrl} target="_blank" rel="noreferrer">前往官方平台 ↗</a></td>{/each}</tr>
</tbody></table></div><p class="data-note">价格为 USD / 1M Tokens。相同 Token 数的成本比较不代表相同任务的实际账单；<a href="/pricing/">用你的调用规模计算 →</a></p>
