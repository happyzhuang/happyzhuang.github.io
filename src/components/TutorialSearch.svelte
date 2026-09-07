<script lang="ts">
 export let items: {title:string; description:string; category:string; href:string}[] = [];
 let query = "";
 $: terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
 $: matches = terms.length ? items.filter(item => terms.every(term => `${item.title} ${item.description} ${item.category}`.toLocaleLowerCase().includes(term))) : [];
</script>
<section class="tutorial-search" aria-label="查找教程"><label for="tutorial-query">想学什么？</label><div class="tutorial-search-field"><span aria-hidden="true">⌕</span><input id="tutorial-query" type="search" bind:value={query} placeholder="搜索教程、概念或问题，如 RAG、API、工具调用" autocomplete="off" />{#if query}<button on:click={() => query = ""} aria-label="清空搜索">清空</button>{/if}</div>
{#if terms.length}<div class="tutorial-results"><p role="status">找到 {matches.length} 篇相关内容{matches.length > 8 ? "，展示前 8 篇" : ""}</p>{#each matches.slice(0,8) as item}<a href={item.href}><span>{item.category}</span><strong>{item.title}</strong><small>{item.description}</small></a>{/each}{#if !matches.length}<p>试试更短的关键词，例如「流式」「检索」或「Prompt」。</p>{/if}</div>{:else}<div class="search-topics"><span>试试搜索</span>{#each ["Prompt", "API", "RAG", "Agent", "部署"] as topic}<button on:click={() => query = topic}>{topic}</button>{/each}</div>{/if}</section>
