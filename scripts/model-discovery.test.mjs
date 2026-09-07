import test from "node:test";
import assert from "node:assert/strict";
import { models } from "../src/data/models.ts";
import { estimate, candidates, validUsage, identifyScenario } from "../src/utils/model-discovery.ts";

const model = id => models.find(m => m.id === id);
const usage = {input:20000,output:2000,requests:100,cachedPercent:0,offPeak:false};
const near = (a,b) => assert.ok(Math.abs(a-b)<1e-9, `${a} != ${b}`);

test("document batch: counts tokens per request exactly once", () => {
  near(estimate(model("gemini-3-8-flash"),usage).total,2.25);
  near(estimate(model("claude-sonnet-5"),usage).total,6);
});
test("cache hit discount is applied only to input", () => {
  const r=estimate(model("claude-sonnet-5"),{...usage,cachedPercent:100});
  near(r.inputCost,.4); near(r.outputCost,2); near(r.total,2.4);
});
test("off-peak only discounts DeepSeek; zero requests costs zero", () => {
  near(estimate(model("deepseek-v4-flash"),{...usage,offPeak:true}).total, .572);
  near(estimate(model("claude-sonnet-5"),{...usage,offPeak:true}).total,6);
  near(estimate(model("gpt-6-astra"),{...usage,requests:0}).total,0);
});
test("long-context threshold reprices the entire GPT request", () => {
  assert.equal(estimate(model("gpt-6-astra"),{...usage,input:272000}).long,false);
  const r=estimate(model("gpt-6-astra"),{...usage,input:272001,output:1000,requests:1});
  assert.equal(r.long,true);near(r.total,5.51502);
});
test("shared context and separate input caps are respected", () => {
  assert.equal(estimate(model("claude-haiku-4-5"),{...usage,input:199000,output:2000}).fits,false);
  assert.equal(estimate(model("gemini-3-8-flash"),{...usage,input:1048576,output:2000}).fits,true);
  assert.equal(estimate(model("gemini-3-8-flash"),{...usage,output:65537}).fits,false);
});
test("reject blank, negative, nonfinite and fractional request counts", () => {
  for (const patch of [{input:undefined},{output:NaN},{requests:Infinity},{input:-1},{requests:.5},{cachedPercent:101}]) {
    assert.equal(validUsage({...usage,...patch}),false);
    assert.equal(estimate(models[0],{...usage,...patch}),null);
  }
});
test("finder enforces scene, length and budget, including no-match results", () => {
  const result=candidates(models,"document",usage,5);
  assert.deepEqual(result.map(r=>r.model.id),["gemini-3-8-flash","claude-haiku-4-5"]);
  assert.deepEqual(candidates(models,"document",usage,0),[]);
  assert.deepEqual(candidates(models,"image",usage,10000),[]);
  assert.deepEqual(candidates(models,"chat",{...usage,output:999999},10000),[]);
});
test("generated media is distinct from media understanding", () => {
  assert.equal(identifyScenario("生成视频"),"video");
  assert.equal(identifyScenario("生成图片"),"image");
  assert.equal(identifyScenario("分析 PDF 财报"),"document");
});
test("catalog IDs and source links are valid", () => {
  assert.equal(new Set(models.map(m=>m.id)).size,models.length);
  for(const m of models) {
    assert.ok(m.input>=m.cached && m.output>0 && m.context>0);
    assert.equal(new URL(m.source).protocol,"https:");
    assert.equal(new URL(m.specs).protocol,"https:");
  }
});
