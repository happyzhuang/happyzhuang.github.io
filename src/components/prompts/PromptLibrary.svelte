<script lang="ts">
	type PromptRecipe = {
		category: string;
		title: string;
		scene: string;
		when: string;
		template: string;
		variables: string[];
		tips: string[];
	};

	const categories = ["全部", "学习解释", "代码开发", "内容写作", "RAG", "Agent", "产品运营"];

	const recipes: PromptRecipe[] = [
		{
			category: "学习解释",
			title: "结构化技术解释",
			scene: "把一个技术概念讲给有基础但不熟悉该领域的读者。",
			when: "写教程、做学习笔记、给团队同步概念时使用。",
			template:
				"请用「一句话定义 / 核心机制 / 典型场景 / 常见误区 / 最小示例」解释：{topic}\n\n读者背景：{audience}\n要求：避免堆砌术语，每个术语第一次出现时都要解释。",
			variables: ["topic", "audience"],
			tips: ["先限定读者背景", "要求模型给最小示例", "让模型指出常见误区"],
		},
		{
			category: "学习解释",
			title: "费曼学习复盘",
			scene: "检查自己是否真正理解一个概念。",
			when: "学完一篇文章、一个视频或一个模型能力后使用。",
			template:
				"我刚学习了：{topic}\n\n请你像导师一样检查我的理解：\n1. 先让我用自己的话解释它。\n2. 找出解释中模糊、跳跃或错误的地方。\n3. 给我 3 个追问。\n4. 最后给一版更清晰的解释。",
			variables: ["topic"],
			tips: ["适合学习地图配套使用", "让模型先追问，不要直接给答案", "可反复迭代"],
		},
		{
			category: "代码开发",
			title: "代码审查助手",
			scene: "让模型优先找 bug、风险和遗漏测试，而不是泛泛总结。",
			when: "提交 PR、重构代码或接入第三方 API 前使用。",
			template:
				"请以资深代码审查者的视角检查以下变更。\n\n目标：{goal}\n变更内容：\n{diff}\n\n请按严重程度列出问题，格式为：\n- 严重级别\n- 文件位置\n- 问题原因\n- 可能影响\n- 修复建议\n\n只列真实风险，不要做泛泛总结。",
			variables: ["goal", "diff"],
			tips: ["提供目标比只贴 diff 更有效", "要求按严重程度排序", "明确不要泛泛总结"],
		},
		{
			category: "代码开发",
			title: "从报错到修复方案",
			scene: "把错误信息、上下文和已尝试方案整理成可诊断输入。",
			when: "调试构建失败、接口报错、依赖冲突时使用。",
			template:
				"我遇到了一个开发问题，请帮我定位原因并给出修复路径。\n\n环境：{environment}\n目标：{goal}\n错误信息：\n{error}\n\n相关代码或配置：\n{context}\n\n我已经尝试过：{attempts}\n\n请输出：最可能原因、验证步骤、最小修复方案、如果仍失败的下一步排查。",
			variables: ["environment", "goal", "error", "context", "attempts"],
			tips: ["不要只贴错误，补充环境", "写出已尝试方案", "要求最小修复"],
		},
		{
			category: "内容写作",
			title: "项目日志整理",
			scene: "把零散开发记录整理成一篇可读的工程复盘。",
			when: "写博客、周报、项目文档或作品集时使用。",
			template:
				"请把以下开发记录整理为项目日志。\n\n记录：\n{notes}\n\n请包含：背景、目标、关键决策、踩坑、解决方案、最终效果、下一步。\n风格：真实、克制、面向开发者，不要写成营销稿。",
			variables: ["notes"],
			tips: ["保留关键取舍", "写出失败路径", "不要让模型过度包装"],
		},
		{
			category: "内容写作",
			title: "长文大纲生成",
			scene: "把一个选题扩展成结构清晰的文章大纲。",
			when: "准备教程、观点文章、产品介绍或学习笔记时使用。",
			template:
				"请为主题「{topic}」生成一份长文大纲。\n\n目标读者：{audience}\n文章目标：{goal}\n限制：{constraints}\n\n请输出：标题建议、核心观点、章节结构、每节要点、需要补充的资料、容易写空的地方。",
			variables: ["topic", "audience", "goal", "constraints"],
			tips: ["先定读者和目标", "让模型指出薄弱章节", "大纲生成后再分段写"],
		},
		{
			category: "RAG",
			title: "RAG 答案生成约束",
			scene: "让模型基于检索片段回答，并标注引用依据。",
			when: "做知识库问答、企业文档助手、资料检索助手时使用。",
			template:
				"你是一个基于检索结果回答问题的助手。\n\n用户问题：{question}\n检索片段：\n{chunks}\n\n回答要求：\n1. 只使用检索片段中的信息回答。\n2. 如果片段不足以回答，请说明缺少什么信息。\n3. 每个关键结论后标注来源编号。\n4. 最后给出“可继续追问的问题”。",
			variables: ["question", "chunks"],
			tips: ["明确禁止编造", "要求标注来源编号", "让模型说出信息不足"],
		},
		{
			category: "RAG",
			title: "检索失败诊断",
			scene: "分析 RAG 没答准是切块、召回、重排还是生成的问题。",
			when: "知识库答案跑偏、引用错误或召回为空时使用。",
			template:
				"请诊断这次 RAG 问答失败的原因。\n\n用户问题：{question}\n期望答案：{expected}\n召回片段：\n{retrieved_chunks}\n模型回答：\n{answer}\n\n请按「问题改写 / 文档切块 / 向量召回 / 重排过滤 / 生成约束」五类分析，并给出优先修复建议。",
			variables: ["question", "expected", "retrieved_chunks", "answer"],
			tips: ["同时提供期望答案和召回片段", "按链路定位问题", "优先修复召回再修生成"],
		},
		{
			category: "Agent",
			title: "Agent 任务规划",
			scene: "让模型先拆解任务，再选择工具和执行顺序。",
			when: "做自动化工作流、资料整理、研究助理时使用。",
			template:
				"你是一个谨慎的任务规划 Agent。\n\n目标：{goal}\n可用工具：{tools}\n限制条件：{constraints}\n\n请先输出计划，不要直接执行：\n1. 任务拆解\n2. 每一步需要的工具\n3. 成功标准\n4. 可能失败点\n5. 需要用户确认的问题",
			variables: ["goal", "tools", "constraints"],
			tips: ["先规划再执行", "列出成功标准", "让模型主动暴露不确定性"],
		},
		{
			category: "Agent",
			title: "工具调用结果反思",
			scene: "让 Agent 在工具返回后检查结果是否足够可靠。",
			when: "做多步搜索、代码生成、数据整理流程时使用。",
			template:
				"请根据工具调用结果进行反思。\n\n原始目标：{goal}\n执行计划：{plan}\n工具结果：\n{tool_results}\n\n请判断：\n- 是否已经满足目标\n- 哪些结论有证据支持\n- 哪些地方仍不确定\n- 是否需要再次调用工具\n- 下一步最小行动是什么",
			variables: ["goal", "plan", "tool_results"],
			tips: ["避免一次工具结果就下结论", "区分证据和推测", "保持下一步最小化"],
		},
		{
			category: "产品运营",
			title: "用户需求澄清",
			scene: "把模糊想法拆成需求、约束和可执行版本。",
			when: "规划网站栏目、功能迭代、产品页面时使用。",
			template:
				"我有一个初步想法：{idea}\n\n请帮我澄清需求：\n1. 用户是谁\n2. 用户真正想完成什么\n3. 这个功能的最小可用版本\n4. 不应该做什么\n5. 衡量效果的指标\n6. 需要我补充的关键问题",
			variables: ["idea"],
			tips: ["适合功能规划早期", "要求给出不做什么", "先做最小可用版本"],
		},
		{
			category: "产品运营",
			title: "竞品页面拆解",
			scene: "分析一个页面的栏目、信息架构和可借鉴点。",
			when: "参考学习社区、榜单站、导航站、工具页时使用。",
			template:
				"请分析这个页面或产品：{product_or_url}\n\n关注点：{focus}\n\n请输出：核心用户、页面目标、信息架构、关键模块、交互方式、可借鉴点、不适合照搬的地方、适合我项目的改造建议。",
			variables: ["product_or_url", "focus"],
			tips: ["先声明关注点", "区分借鉴和照搬", "要求给出改造建议"],
		},
	];

	let activeCategory = "全部";
	let copiedTitle = "";

	$: filteredRecipes =
		activeCategory === "全部"
			? recipes
			: recipes.filter((recipe) => recipe.category === activeCategory);

	async function copyRecipe(recipe: PromptRecipe) {
		await navigator.clipboard.writeText(recipe.template);
		copiedTitle = recipe.title;
		setTimeout(() => {
			if (copiedTitle === recipe.title) copiedTitle = "";
		}, 1600);
	}
</script>

<section class="card-base px-5 py-5 md:px-7 md:py-7 mb-4 onload-animation">
	<div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
		<div>
			<h2 class="text-2xl font-bold text-90 mb-2">Prompt 配方库</h2>
			<p class="text-75 leading-7">
				按真实任务整理模板。选择场景后复制，再把大括号里的变量替换成你的具体内容。
			</p>
		</div>
		<div class="flex flex-wrap gap-2">
			{#each categories as category}
				<button
					type="button"
					class="h-10 rounded-xl px-3 text-sm font-bold transition"
					class:bg-[var(--primary)]={activeCategory === category}
					class:text-white={activeCategory === category}
					class:bg-[var(--btn-regular-bg)]={activeCategory !== category}
					class:text-75={activeCategory !== category}
					onclick={() => (activeCategory = category)}
				>
					{category}
				</button>
			{/each}
		</div>
	</div>
</section>

<section class="grid grid-cols-1 gap-4 mb-4">
	{#each filteredRecipes as recipe}
		<article class="card-base px-5 py-5 md:px-7 md:py-6 onload-animation">
			<div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
				<div class="min-w-0">
					<div class="mb-2 flex flex-wrap items-center gap-2">
						<span class="rounded-lg bg-[var(--btn-regular-bg)] px-2.5 py-1 text-xs font-bold text-[var(--primary)]">
							{recipe.category}
						</span>
						<span class="text-sm text-50">{recipe.when}</span>
					</div>
					<h3 class="text-2xl font-bold text-90 mb-2">{recipe.title}</h3>
					<p class="text-75 leading-7">{recipe.scene}</p>
				</div>
				<button
					type="button"
					class="h-10 shrink-0 rounded-xl bg-[var(--primary)] px-4 text-sm font-bold text-white transition hover:opacity-90"
					onclick={() => copyRecipe(recipe)}
					aria-label={`复制 ${recipe.title} 模板`}
				>
					{copiedTitle === recipe.title ? "已复制" : "复制模板"}
				</button>
			</div>

			<div class="mt-5 grid grid-cols-1 xl:grid-cols-[1fr_18rem] gap-4">
				<pre class="whitespace-pre-wrap rounded-2xl bg-[var(--codeblock-bg)] text-white/85 px-4 py-4 overflow-x-auto text-sm leading-6"><code>{recipe.template}</code></pre>
				<div class="grid grid-cols-1 gap-4">
					<div class="rounded-2xl bg-[var(--btn-regular-bg)] p-4">
						<div class="text-sm font-bold text-90 mb-3">需要替换的变量</div>
						<div class="flex flex-wrap gap-2">
							{#each recipe.variables as variable}
								<span class="rounded-lg bg-[var(--card-bg)] px-2.5 py-1 text-xs text-75">{variable}</span>
							{/each}
						</div>
					</div>
					<div class="rounded-2xl bg-[var(--btn-regular-bg)] p-4">
						<div class="text-sm font-bold text-90 mb-3">使用要点</div>
						<ul class="space-y-2">
							{#each recipe.tips as tip}
								<li class="text-sm leading-6 text-75">- {tip}</li>
							{/each}
						</ul>
					</div>
				</div>
			</div>
		</article>
	{/each}
</section>
