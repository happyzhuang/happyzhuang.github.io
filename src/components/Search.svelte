<script lang="ts">
 import { onMount, tick } from "svelte";
 export let items: {title:string;description:string;href:string}[]=[];
 let open=false;
 let query="";
 let mounted=false;
 let input:HTMLInputElement;
 let root:HTMLDivElement;
 let results:typeof items=[];
 let loading=false;
 let sequence=0;
 let timer:ReturnType<typeof setTimeout>;
 async function toggle() { open=!open; if(open) { await tick();input?.focus(); } }
 async function search(value:string) {
  const key=value.trim().toLowerCase();
  const request=++sequence;
  if(!key) { results=[];loading=false;return; }
  loading=true;
  let found=items.filter(i=>`${i.title} ${i.description}`.toLowerCase().includes(key)).slice(0,8);
  if(import.meta.env.PROD && window.pagefind) {
   try { const response=await window.pagefind.search(key);const data=await Promise.all(response.results.slice(0,8).map(r=>r.data()));if(data.length) found=data.map(r=>({title:r.meta.title,description:r.excerpt.replace(/<[^>]+>/g,""),href:r.url})); } catch { /* Keep the local title/description index available. */ }
  }
  if(request===sequence) { results=found;loading=false; }
 }
 onMount(()=>{
  mounted=true;
  const outside=(e:PointerEvent)=>{if(e.target instanceof Node && !root?.contains(e.target))open=false;};
  const ready=()=>search(query);
  document.addEventListener("pointerdown",outside);
  document.addEventListener("pagefindready",ready);
  return ()=>{clearTimeout(timer);sequence++;document.removeEventListener("pointerdown",outside);document.removeEventListener("pagefindready",ready);};
 });
 $: if(mounted) { clearTimeout(timer); const value=query; timer=setTimeout(()=>search(value),150); }
</script>
<svelte:window on:keydown={e=>{if(e.key==="Escape" && open){open=false;root?.querySelector("button")?.focus();}}}/>
<div bind:this={root} class="global-search">
 <button on:click={toggle} aria-label="搜索模型与教程" aria-expanded={open} aria-controls="search-panel" class="global-search-trigger"><span aria-hidden="true">⌕</span><span class="search-button-label">搜索模型与教程</span></button>
 {#if open}<div id="search-panel" class="global-search-panel"><label for="global-search-query">搜索模型与教程</label><input id="global-search-query" bind:this={input} bind:value={query} type="search" placeholder="模型名称、RAG、API…" autocomplete="off"/><div role="status" class="global-search-status">{loading ? "搜索中…" : query.trim() ? results.length ? `找到 ${results.length} 条结果` : "没有找到结果，试试其他关键词。" : "输入关键词，查找模型资料与教程。"}</div>{#each results as item}<a href={item.href} on:click={()=>open=false}><strong>{item.title}</strong><span>{item.description}</span></a>{/each}<button class="global-search-close" on:click={()=>open=false}>关闭 · Esc</button></div>{/if}
</div>
