<script lang="ts">
	const sampleText = "LLM, Hello! 用一个小工具估算文本 token 和调用成本。";

	let text = sampleText;
	let inputPrice: number | undefined = 0.15;
	let outputPrice: number | undefined = 0.6;
	let expectedOutputTokens: number | undefined = 800;

	$: cjkCount = Array.from(text).filter((char) => /[\u3400-\u9fff]/u.test(char))
		.length;
	$: asciiCount = Math.max(text.length - cjkCount, 0);
	$: estimatedInputTokens = Math.max(
		1,
		Math.ceil(cjkCount * 1.15 + asciiCount / 4),
	);
	$: safeInputPrice = Number(inputPrice) || 0;
	$: safeOutputPrice = Number(outputPrice) || 0;
	$: safeOutputTokens = Number(expectedOutputTokens) || 0;
	$: totalTokens = estimatedInputTokens + safeOutputTokens;
	$: estimatedCost =
		(estimatedInputTokens / 1_000_000) * safeInputPrice +
		(safeOutputTokens / 1_000_000) * safeOutputPrice;
</script>

<div class="grid grid-cols-1 lg:grid-cols-[1fr_18rem] gap-4">
	<div>
		<label for="token-input" class="block text-sm font-bold text-75 mb-2">输入文本</label>
		<textarea
			id="token-input"
			bind:value={text}
			rows="9"
			class="w-full resize-y rounded-2xl bg-[var(--btn-regular-bg)] text-90 px-4 py-3 leading-7 outline-none focus:ring-2 focus:ring-[var(--primary)]"
		></textarea>
		<p class="text-xs text-50 mt-2">
			这是一个前端粗估工具：中文按略高于字符数估算，英文按约 4 字符 1 token 估算。实际结果以模型 tokenizer 为准。
		</p>
	</div>

	<div class="rounded-2xl bg-[var(--btn-regular-bg)] p-4">
		<div class="text-sm font-bold text-75 mb-3">价格参数</div>
		<label class="block text-xs text-50 mb-1" for="input-price">输入价格 / 百万 token</label>
		<input id="input-price" type="number" step="0.01" min="0" bind:value={inputPrice} class="w-full rounded-xl bg-[var(--card-bg)] text-90 px-3 py-2 outline-none focus:ring-2 focus:ring-[var(--primary)]" />

		<label class="block text-xs text-50 mb-1 mt-3" for="output-price">输出价格 / 百万 token</label>
		<input id="output-price" type="number" step="0.01" min="0" bind:value={outputPrice} class="w-full rounded-xl bg-[var(--card-bg)] text-90 px-3 py-2 outline-none focus:ring-2 focus:ring-[var(--primary)]" />

		<label class="block text-xs text-50 mb-1 mt-3" for="output-tokens">预计输出 token</label>
		<input id="output-tokens" type="number" step="100" min="0" bind:value={expectedOutputTokens} class="w-full rounded-xl bg-[var(--card-bg)] text-90 px-3 py-2 outline-none focus:ring-2 focus:ring-[var(--primary)]" />
	</div>
</div>

<div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
	<div class="rounded-2xl bg-[var(--btn-regular-bg)] p-4">
		<div class="text-xs text-50 mb-1">输入估算</div>
		<div class="text-2xl font-bold text-90">{estimatedInputTokens.toLocaleString()}</div>
	</div>
	<div class="rounded-2xl bg-[var(--btn-regular-bg)] p-4">
		<div class="text-xs text-50 mb-1">总 token</div>
		<div class="text-2xl font-bold text-90">{totalTokens.toLocaleString()}</div>
	</div>
	<div class="rounded-2xl bg-[var(--btn-regular-bg)] p-4">
		<div class="text-xs text-50 mb-1">成本估算</div>
		<div class="text-2xl font-bold text-90">${estimatedCost.toFixed(6)}</div>
	</div>
</div>
