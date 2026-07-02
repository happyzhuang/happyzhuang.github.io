<script lang="ts">
	type PromptMode = {
		key: string;
		name: string;
		shortName: string;
		score: number;
		risk: string;
		parts: string[];
		buildPrompt: (task: string, context: string, outputFormat: string) => string;
		buildOutput: (task: string, outputFormat: string) => string[];
	};

	let task = "帮我解释 RAG 的基本原理";
	let context = "读者懂一点编程，但刚开始学习大模型应用开发。";
	let outputFormat = "用条目列出，最后给一个最小示例。";
	let activeMode = "structured";

	const modes: PromptMode[] = [
		{
			key: "plain",
			name: "普通提问",
			shortName: "普通",
			score: 56,
			risk: "容易得到泛泛解释，输出结构不稳定。",
			parts: ["任务"],
			buildPrompt: (task) => task,
			buildOutput: (task) => [
				`这是对“${task}”的直接回答。`,
				"模型会根据常识生成一段说明，但可能忽略读者背景、输出格式和边界条件。",
				"适合快速试探，不适合作为可复用模板。",
			],
		},
		{
			key: "structured",
			name: "结构化提问",
			shortName: "结构化",
			score: 84,
			risk: "结构清楚，但如果约束过多，可能牺牲自然度。",
			parts: ["任务", "背景", "输出格式", "约束"],
			buildPrompt: (task, context, outputFormat) =>
				[
					`任务：${task}`,
					`背景：${context}`,
					`输出格式：${outputFormat}`,
					"要求：先给一句话结论，再解释关键机制，避免堆砌术语。",
				].join("\n"),
			buildOutput: (task, outputFormat) => [
				`一句话结论：${task} 可以被拆成清晰的概念、流程和例子。`,
				"关键机制：先界定问题，再按步骤解释，并保留必要的上下文。",
				`输出形态：${outputFormat}`,
			],
		},
		{
			key: "role",
			name: "角色设定",
			shortName: "角色",
			score: 78,
			risk: "风格更稳定，但角色设定太强时可能产生不必要的口吻。",
			parts: ["角色", "任务", "受众", "风格"],
			buildPrompt: (task, context, outputFormat) =>
				[
					"你是一名擅长把复杂 AI 技术讲清楚的工程导师。",
					`请面向以下读者完成任务：${context}`,
					`任务：${task}`,
					`输出格式：${outputFormat}`,
					"风格：准确、克制、面向实践。",
				].join("\n"),
			buildOutput: (task) => [
				`我会把“${task}”拆成学习者容易理解的三层：概念、流程、实践入口。`,
				"先建立直觉，再补技术词，最后给一个可以继续动手的方向。",
				"适合教学、教程、技术博客等场景。",
			],
		},
		{
			key: "fewshot",
			name: "Few-shot 示例",
			shortName: "Few-shot",
			score: 91,
			risk: "效果最稳定，但模板更长，维护成本更高。",
			parts: ["任务", "示例", "迁移规则", "输出格式"],
			buildPrompt: (task, context, outputFormat) =>
				[
					`读者背景：${context}`,
					"示例：",
					"输入：解释 Embedding",
					"输出：一句话定义 / 为什么有用 / 最小例子 / 常见误区",
					"",
					`现在请按同样结构完成：${task}`,
					`输出格式：${outputFormat}`,
				].join("\n"),
			buildOutput: (task) => [
				`模型会把“${task}”迁移到示例结构中，输出更一致。`,
				"由于示例提供了目标形态，模型更容易复用结构和粒度。",
				"适合批量生成、内容规范化和团队协作模板。",
			],
		},
	];

	$: active = modes.find((mode) => mode.key === activeMode) ?? modes[1];
	$: prompt = active.buildPrompt(task.trim() || "请填写任务", context.trim(), outputFormat.trim());
	$: output = active.buildOutput(task.trim() || "当前任务", outputFormat.trim());
</script>

<div class="grid grid-cols-1 xl:grid-cols-[0.9fr_1.1fr] gap-4">
	<section class="rounded-2xl bg-[var(--btn-regular-bg)] p-4 md:p-5">
		<div class="grid grid-cols-1 gap-4">
			<div>
				<label for="prompt-task" class="block text-sm font-bold text-75 mb-2">任务</label>
				<textarea
					id="prompt-task"
					bind:value={task}
					rows="4"
					class="w-full resize-y rounded-2xl bg-[var(--card-bg)] text-90 px-4 py-3 leading-7 outline-none focus:ring-2 focus:ring-[var(--primary)]"
				></textarea>
			</div>

			<div>
				<label for="prompt-context" class="block text-sm font-bold text-75 mb-2">背景</label>
				<textarea
					id="prompt-context"
					bind:value={context}
					rows="3"
					class="w-full resize-y rounded-2xl bg-[var(--card-bg)] text-90 px-4 py-3 leading-7 outline-none focus:ring-2 focus:ring-[var(--primary)]"
				></textarea>
			</div>

			<div>
				<label for="prompt-output" class="block text-sm font-bold text-75 mb-2">输出要求</label>
				<input
					id="prompt-output"
					bind:value={outputFormat}
					class="w-full rounded-2xl bg-[var(--card-bg)] text-90 px-4 py-3 outline-none focus:ring-2 focus:ring-[var(--primary)]"
				/>
			</div>
		</div>
	</section>

	<section class="rounded-2xl bg-[var(--btn-regular-bg)] p-4 md:p-5">
		<div class="grid grid-cols-2 md:grid-cols-4 gap-2 mb-5">
			{#each modes as mode}
				<button
					type="button"
					class="h-10 rounded-xl text-sm font-bold transition"
					class:bg-[var(--primary)]={activeMode === mode.key}
					class:text-white={activeMode === mode.key}
					class:bg-[var(--card-bg)]={activeMode !== mode.key}
					class:text-75={activeMode !== mode.key}
					on:click={() => (activeMode = mode.key)}
				>
					{mode.shortName}
				</button>
			{/each}
		</div>

		<div class="grid grid-cols-1 lg:grid-cols-[1fr_15rem] gap-4">
			<div class="rounded-2xl bg-[var(--card-bg)] p-4">
				<div class="flex items-center justify-between gap-3 mb-3">
					<h3 class="font-bold text-90">{active.name}</h3>
					<span class="text-xs rounded-lg px-2.5 py-1 bg-[var(--btn-regular-bg)] text-50">模拟输出</span>
				</div>
				<pre class="whitespace-pre-wrap rounded-xl bg-[var(--codeblock-bg)] text-white/85 px-4 py-4 overflow-x-auto text-sm leading-6"><code>{prompt}</code></pre>
			</div>

			<div class="rounded-2xl bg-[var(--card-bg)] p-4">
				<div class="text-xs text-50 mb-1">清晰度评分</div>
				<div class="text-3xl font-bold text-90 mb-3">{active.score}</div>
				<div class="h-2 rounded-full bg-[var(--btn-regular-bg)] overflow-hidden mb-4">
					<div class="h-full bg-[var(--primary)] rounded-full" style={`width: ${active.score}%`}></div>
				</div>
				<div class="text-xs text-50 mb-2">包含要素</div>
				<div class="flex flex-wrap gap-2">
					{#each active.parts as part}
						<span class="rounded-lg bg-[var(--btn-regular-bg)] px-2.5 py-1 text-xs text-75">{part}</span>
					{/each}
				</div>
			</div>
		</div>

		<div class="grid grid-cols-1 lg:grid-cols-[1fr_15rem] gap-4 mt-4">
			<div class="rounded-2xl bg-[var(--card-bg)] p-4">
				<h3 class="font-bold text-90 mb-3">输出预览</h3>
				<div class="space-y-2">
					{#each output as item}
						<div class="rounded-xl bg-[var(--btn-regular-bg)] px-3 py-2.5 text-sm text-75 leading-6">{item}</div>
					{/each}
				</div>
			</div>

			<div class="rounded-2xl bg-[var(--card-bg)] p-4">
				<h3 class="font-bold text-90 mb-2">风险</h3>
				<p class="text-sm text-75 leading-6">{active.risk}</p>
			</div>
		</div>
	</section>
</div>
