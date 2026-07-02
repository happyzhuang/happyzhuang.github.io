<script lang="ts">
	type AgentStep = {
		key: string;
		name: string;
		status: string;
		description: string;
		trace: string[];
	};

	let goal = "帮我比较 DeepSeek 和 Qwen，给出适合个人开发者的模型选择建议。";
	let activeStep = "plan";

	const tools = [
		["模型资料库", "查询模型能力、上下文、价格和 API 特性。"],
		["成本估算器", "估算输入输出 token 成本。"],
		["网页搜索", "检查最新发布和官方文档。"],
		["表格生成器", "把结果整理成对比表。"],
	];

	const steps: AgentStep[] = [
		{
			key: "plan",
			name: "1. 任务规划",
			status: "分析中",
			description: "Agent 先理解目标，并拆成可执行步骤。",
			trace: [
				"识别目标：比较 DeepSeek 和 Qwen。",
				"拆解任务：能力、成本、API 易用性、适合场景。",
				"确定输出：给个人开发者的选择建议。",
			],
		},
		{
			key: "tool",
			name: "2. 选择工具",
			status: "选择中",
			description: "Agent 根据子任务选择需要调用的工具。",
			trace: [
				"需要模型能力信息 → 调用模型资料库。",
				"需要成本估算 → 调用成本估算器。",
				"需要最新信息 → 调用网页搜索。",
				"需要清晰交付 → 调用表格生成器。",
			],
		},
		{
			key: "execute",
			name: "3. 工具执行",
			status: "执行中",
			description: "Agent 调用工具并收集结构化结果。",
			trace: [
				"模型资料库返回：DeepSeek 偏性价比和代码，Qwen 模型谱系更完整。",
				"成本估算器返回：不同输出长度下的费用区间。",
				"网页搜索返回：官方 API 文档和模型更新入口。",
				"表格生成器生成：能力 / 成本 / 生态 / 建议。",
			],
		},
		{
			key: "reflect",
			name: "4. 反思修正",
			status: "校验中",
			description: "Agent 检查结果是否覆盖目标，并补充缺失信息。",
			trace: [
				"发现只比较了能力，缺少接入复杂度。",
				"补充 API 兼容性、SDK、国内访问稳定性。",
				"检查建议是否面向个人开发者，而不是企业采购。",
			],
		},
		{
			key: "answer",
			name: "5. 汇总输出",
			status: "完成",
			description: "Agent 把执行结果整理成最终回答。",
			trace: [
				"如果预算敏感、偏代码任务：优先试 DeepSeek。",
				"如果需要多模态、开源部署和模型谱系：优先试 Qwen。",
				"建议先用同一组 Prompt 做 A/B 测试，再决定默认模型。",
			],
		},
	];

	$: active = steps.find((step) => step.key === activeStep) ?? steps[0];
</script>

<div class="grid grid-cols-1 xl:grid-cols-[0.85fr_1.15fr] gap-4">
	<section class="rounded-2xl bg-[var(--btn-regular-bg)] p-4 md:p-5">
		<label for="agent-goal" class="block text-sm font-bold text-75 mb-2">任务目标</label>
		<textarea
			id="agent-goal"
			bind:value={goal}
			rows="5"
			class="w-full resize-y rounded-2xl bg-[var(--card-bg)] text-90 px-4 py-3 leading-7 outline-none focus:ring-2 focus:ring-[var(--primary)]"
		></textarea>

		<div class="mt-4">
			<div class="text-sm font-bold text-75 mb-2">可用工具</div>
			<div class="space-y-2">
				{#each tools as [name, description]}
					<div class="rounded-xl bg-[var(--card-bg)] px-3 py-2.5">
						<div class="font-bold text-sm text-90">{name}</div>
						<div class="text-xs text-50 mt-1">{description}</div>
					</div>
				{/each}
			</div>
		</div>
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

		<div class="rounded-2xl bg-[var(--card-bg)] p-4 mb-4">
			<div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-2">
				<h3 class="font-bold text-90">{active.name}</h3>
				<span class="self-start sm:self-auto rounded-lg bg-[var(--btn-regular-bg)] px-2.5 py-1 text-xs text-50">{active.status}</span>
			</div>
			<p class="text-sm text-75 leading-6">{active.description}</p>
		</div>

		<div class="rounded-2xl bg-[var(--card-bg)] p-4">
			<h3 class="font-bold text-90 mb-3">执行轨迹</h3>
			<div class="space-y-2">
				{#each active.trace as item, index}
					<div class="flex gap-3 rounded-xl bg-[var(--btn-regular-bg)] px-3 py-2.5">
						<div class="shrink-0 h-6 w-6 rounded-lg bg-[var(--card-bg)] text-xs text-[var(--primary)] font-bold flex items-center justify-center">{index + 1}</div>
						<div class="text-sm text-75 leading-6">{item}</div>
					</div>
				{/each}
			</div>
		</div>
	</section>
</div>
