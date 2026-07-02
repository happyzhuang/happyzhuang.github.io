<script lang="ts">
	type RagStep = {
		key: string;
		name: string;
		description: string;
		outputTitle: string;
		output: string[];
	};

	let documentText =
		"LLM, Hello! 是一个面向中文开发者的大模型学习与实战入口。网站包含模型排名、学习地图、实验室、Prompt 配方和实战案例。实验室用于展示 Token 成本、Prompt 对比、RAG 流程和 Agent 工作流。";
	let question = "这个网站适合用来学习哪些大模型实践内容？";
	let activeStep = "chunk";

	const steps: RagStep[] = [
		{
			key: "chunk",
			name: "1. 文档切块",
			description: "把长文档拆成适合检索的小片段，避免一次塞入过多上下文。",
			outputTitle: "切块结果",
			output: [
				"片段 A：LLM, Hello! 是一个面向中文开发者的大模型学习与实战入口。",
				"片段 B：网站包含模型排名、学习地图、实验室、Prompt 配方和实战案例。",
				"片段 C：实验室用于展示 Token 成本、Prompt 对比、RAG 流程和 Agent 工作流。",
			],
		},
		{
			key: "embed",
			name: "2. 向量化",
			description: "把问题和文档片段转成向量，方便计算语义相似度。",
			outputTitle: "向量表示",
			output: [
				"问题向量：query → [0.18, 0.72, 0.41, ...]",
				"片段 A：chunk_a → [0.21, 0.65, 0.39, ...]",
				"片段 B：chunk_b → [0.44, 0.81, 0.52, ...]",
				"片段 C：chunk_c → [0.49, 0.86, 0.58, ...]",
			],
		},
		{
			key: "retrieve",
			name: "3. 检索召回",
			description: "按相似度找出最可能回答问题的片段。",
			outputTitle: "召回片段",
			output: [
				"Top 1：片段 C，相似度 0.91，包含 Token、Prompt、RAG、Agent。",
				"Top 2：片段 B，相似度 0.84，包含模型排名、学习地图、实战案例。",
				"Top 3：片段 A，相似度 0.67，说明网站定位。",
			],
		},
		{
			key: "rerank",
			name: "4. 重排过滤",
			description: "把召回结果按问题相关性重新排序，减少无关上下文干扰。",
			outputTitle: "重排后上下文",
			output: [
				"保留：片段 C，直接回答实践内容。",
				"保留：片段 B，补充网站栏目结构。",
				"降权：片段 A，只作为背景信息。",
			],
		},
		{
			key: "generate",
			name: "5. 生成回答",
			description: "把问题和检索到的上下文一起交给模型生成答案。",
			outputTitle: "模拟回答",
			output: [
				"这个网站适合学习模型选型、Prompt 编写、Token 成本估算、RAG 流程、Agent 工作流和实战项目落地。",
				"依据来自片段 B 和片段 C：它们分别提到模型排名、学习地图、实验室、Prompt 配方和实战案例。",
			],
		},
	];

	$: active = steps.find((step) => step.key === activeStep) ?? steps[0];
</script>

<div class="grid grid-cols-1 xl:grid-cols-[0.9fr_1.1fr] gap-4">
	<section class="rounded-2xl bg-[var(--btn-regular-bg)] p-4 md:p-5">
		<label for="rag-doc" class="block text-sm font-bold text-75 mb-2">示例文档</label>
		<textarea
			id="rag-doc"
			bind:value={documentText}
			rows="7"
			class="w-full resize-y rounded-2xl bg-[var(--card-bg)] text-90 px-4 py-3 leading-7 outline-none focus:ring-2 focus:ring-[var(--primary)]"
		></textarea>

		<label for="rag-question" class="block text-sm font-bold text-75 mb-2 mt-4">用户问题</label>
		<input
			id="rag-question"
			bind:value={question}
			class="w-full rounded-2xl bg-[var(--card-bg)] text-90 px-4 py-3 outline-none focus:ring-2 focus:ring-[var(--primary)]"
		/>
	</section>

	<section class="rounded-2xl bg-[var(--btn-regular-bg)] p-4 md:p-5">
		<div class="grid grid-cols-1 md:grid-cols-5 gap-2 mb-5">
			{#each steps as step}
				<button
					type="button"
					class="rounded-xl px-3 py-3 text-sm font-bold leading-5 transition text-left"
					class:bg-[var(--primary)]={activeStep === step.key}
					class:text-white={activeStep === step.key}
					class:bg-[var(--card-bg)]={activeStep !== step.key}
					class:text-75={activeStep !== step.key}
					on:click={() => (activeStep = step.key)}
				>
					{step.name}
				</button>
			{/each}
		</div>

		<div class="grid grid-cols-1 lg:grid-cols-[16rem_1fr] gap-4">
			<div class="rounded-2xl bg-[var(--card-bg)] p-4">
				<h3 class="font-bold text-90 mb-2">{active.name}</h3>
				<p class="text-sm text-75 leading-6">{active.description}</p>
			</div>

			<div class="rounded-2xl bg-[var(--card-bg)] p-4">
				<h3 class="font-bold text-90 mb-3">{active.outputTitle}</h3>
				<div class="space-y-2">
					{#each active.output as item}
						<div class="rounded-xl bg-[var(--btn-regular-bg)] px-3 py-2.5 text-sm text-75 leading-6">{item}</div>
					{/each}
				</div>
			</div>
		</div>
	</section>
</div>
