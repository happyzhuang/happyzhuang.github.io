<script lang="ts">
	type LeaderboardRow = {
		rank: number;
		model: string;
		score: string;
		pending?: boolean;
	};

	type Leaderboard = {
		key: string;
		label: string;
		metric: string;
		rows: LeaderboardRow[];
	};

	const fillTop20 = (rows: LeaderboardRow[]) => {
		const nextRows = [...rows];
		for (let rank = nextRows.length + 1; rank <= 20; rank += 1) {
			nextRows.push({
				rank,
				model: "待补齐官方数据",
				score: "—",
				pending: true,
			});
		}
		return nextRows;
	};

	const leaderboards: Leaderboard[] = [
		{
			key: "agent",
			label: "Agent",
			metric: "胜率",
			rows: fillTop20([
				{ rank: 1, model: "Anthropic Claude Fable 5 (High)", score: "13.34% ±1.55%" },
				{ rank: 2, model: "Anthropic Claude Opus 4.8 (Thinking)", score: "9.37% ±1.29%" },
				{ rank: 3, model: "GPT 5.5 (xHigh)", score: "8.21% ±1.02%" },
				{ rank: 4, model: "Anthropic Claude Opus 4.7", score: "8.16% ±1.28%" },
				{ rank: 5, model: "Anthropic Claude Opus 4.7 (Thinking)", score: "8.07% ±1.23%" },
			]),
		},
		{
			key: "text",
			label: "Text",
			metric: "Arena 分",
			rows: fillTop20([
				{ rank: 1, model: "Anthropic claude-fable-5", score: "1508 ±9" },
				{ rank: 2, model: "Anthropic claude-opus-4-6-thinking", score: "1503 ±4" },
				{ rank: 3, model: "Anthropic claude-opus-4-7-thinking", score: "1502 ±4" },
				{ rank: 4, model: "Anthropic claude-opus-4-6", score: "1499 ±4" },
				{ rank: 5, model: "Anthropic claude-opus-4-7", score: "1494 ±4" },
			]),
		},
		{
			key: "webdev",
			label: "WebDev",
			metric: "Arena 分",
			rows: fillTop20([
				{ rank: 1, model: "Anthropic claude-fable-5", score: "1654 +16/-16" },
				{ rank: 2, model: "glm-5.2 (max)", score: "1593 +15/-15" },
				{ rank: 3, model: "Anthropic claude-opus-4-8-thinking", score: "1565 +12/-12" },
				{ rank: 4, model: "Anthropic claude-opus-4-7-thinking", score: "1563 +8/-8" },
				{ rank: 5, model: "Anthropic claude-opus-4-7", score: "1557 +8/-8" },
			]),
		},
		{
			key: "vision",
			label: "Vision",
			metric: "Arena 分",
			rows: fillTop20([
				{ rank: 1, model: "Anthropic claude-fable-5", score: "1311 ±14" },
				{ rank: 2, model: "Anthropic claude-opus-4-7-thinking", score: "1308 ±7" },
				{ rank: 3, model: "Anthropic claude-opus-4-6-thinking", score: "1299 ±7" },
				{ rank: 4, model: "Anthropic claude-opus-4-7", score: "1298 ±7" },
				{ rank: 5, model: "Anthropic claude-opus-4-6", score: "1297 ±7" },
			]),
		},
		{
			key: "document",
			label: "Document",
			metric: "Arena 分",
			rows: fillTop20([]),
		},
		{
			key: "search",
			label: "Search",
			metric: "Arena 分",
			rows: fillTop20([]),
		},
	];

	let activeKey = leaderboards[0].key;

	$: activeBoard = leaderboards.find((board) => board.key === activeKey) ?? leaderboards[0];
	$: confirmedScores = activeBoard.rows
		.filter((row) => !row.pending)
		.map((row) => Number.parseFloat(row.score));
	$: maxScore = Math.max(...confirmedScores, 1);

	const barWidth = (row: LeaderboardRow) => {
		if (row.pending) return "0%";
		return `${Math.max(8, (Number.parseFloat(row.score) / maxScore) * 100)}%`;
	};
</script>

<div class="rounded-2xl bg-[var(--btn-regular-bg)] px-4 py-4 md:px-5 md:py-5">
	<div class="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4 mb-4">
		<div>
			<h3 class="text-xl font-bold text-90 mb-1">{activeBoard.label}</h3>
			<p class="text-sm text-50">{activeBoard.metric} · Top 20 · 未确认位次以占位行标记</p>
		</div>
		<div class="flex flex-wrap justify-start lg:justify-end gap-2">
			{#each leaderboards as board}
				<button
					type="button"
					class="h-9 rounded-xl px-3 text-sm font-bold transition"
					class:bg-[var(--primary)]={activeKey === board.key}
					class:text-white={activeKey === board.key}
					class:bg-[var(--card-bg)]={activeKey !== board.key}
					class:text-75={activeKey !== board.key}
					on:click={() => (activeKey = board.key)}
				>
					{board.label}
				</button>
			{/each}
		</div>
	</div>

	<div class="overflow-x-auto">
		<div class="min-w-[48rem]">
			<div class="grid grid-cols-[4rem_minmax(16rem,1fr)_8rem_18rem] gap-4 px-4 pb-2 text-xs font-bold text-50">
				<div>排名</div>
				<div>模型</div>
				<div class="text-right">分数</div>
				<div>相对表现</div>
			</div>
			<div class="space-y-2">
				{#each activeBoard.rows as row}
					<div class="grid grid-cols-[4rem_minmax(16rem,1fr)_8rem_18rem] gap-4 items-center rounded-xl bg-[var(--card-bg)] px-4 py-3">
						<div class="font-bold text-[var(--primary)]">#{row.rank}</div>
						<div class="font-medium leading-5" class:text-90={!row.pending} class:text-50={row.pending}>
							{row.model}
						</div>
						<div class="text-sm text-right whitespace-nowrap" class:text-75={!row.pending} class:text-50={row.pending}>
							{row.score}
						</div>
						<div class="h-3 rounded-full bg-[var(--btn-regular-bg)] overflow-hidden">
							<div
								class="h-full rounded-full bg-[var(--primary)] transition-[width] duration-300"
								style={`width: ${barWidth(row)}`}
							></div>
						</div>
					</div>
				{/each}
			</div>
		</div>
	</div>
</div>
