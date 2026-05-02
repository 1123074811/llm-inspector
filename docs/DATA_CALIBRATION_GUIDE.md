# Data Calibration Guide

> **谁该读这份文档？** 想让 LLM Inspector 真正达到生产级精度的运维/研究人员。
> 代码完成度 ~95%，但**两类生产数据**需要你用真实 API 调用才能采集到位。本指南是动手手册。

---

## TL;DR：两步让数据落地

| 校准目标 | 影响哪一层 | 命令 | 预估开销 |
|---|---|---|---|
| **timing 参考分布** | L18 时序侧信道、L19 token 分布 | `python backend/scripts/sample_timing_references.py --all --samples 100` | ~600 次请求，6 家厂商 ×100，约 30 分钟 / $1–3 |
| **IRT 题库参数** | Theta 标准分（500/100）、CAT 自适应、verdict 区分度 | 见下方"IRT 校准三步走" | 100 模型 × 数十题，~$50–200 |

不跑也能用——但 L18/L19 会主动返回 `confidence: 0.0`（`reason: all_baselines_placeholder`），Theta 的统计语义是"近似"。

---

## Part 1：L18/L19 timing 参考分布采样

### 现状
- `backend/app/_data/timing_refs.json` 与 `token_dist_refs.json` 是 `v15.0-placeholder`
- 每个家族 `sampled: false`、`ttft_ms_mean` 是工程师拍脑袋的初值
- L18 (`layers_l18_l19.py`) 检测到 `sampled=False` → 直接返回 `confidence: 0.0`，**等于这两层目前不参与 verdict**

### 一键脚本（推荐）
```bash
# 准备环境变量（用你自己的真 key）
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export DEEPSEEK_API_KEY=...
export XAI_API_KEY=...
export MISTRAL_API_KEY=...

# 跑批量采样（默认每家 50 样本，建议 100）
python backend/scripts/sample_timing_references.py --all --samples 100
```

脚本会：
1. 对每个 family 发 100 次 chat completion 请求（PROBE_PROMPT 是固定的"Explain briefly what a large language model is"）
2. 测 TTFT（首 token 时延，毫秒）、TPS（tokens/sec）
3. 计算 mean / std / p10–p90 quantile
4. 测 4-gram 重复率 + 平均回答长度（给 L19 用）
5. SHA256 校验原始数据
6. 写回 `timing_refs.json` + 把 `_provenance.sampling_required` 翻成 `False`、`version` 翻成 `v17.0-self-measurement`

### 单家族模式（缺某家 key 时）
```bash
python backend/scripts/sample_timing_references.py \
  --base-url https://api.openai.com/v1 \
  --api-key  $OPENAI_API_KEY \
  --family   gpt \
  --model    gpt-4o-mini \
  --auth     bearer \
  --samples  100
```

`--auth` 取值：
- `bearer`：OpenAI 兼容（DeepSeek、Grok、Mistral、Together、Groq…）
- `x-api-key`：Anthropic Messages API（用 `/v1/messages` 端点）
- `google-key`：Gemini `generateContent`（key 在 URL 里）

> ⚠️ **2026-05-01 修复前的 bug**：`--all` 模式会把所有家族都按 Bearer 发 `/chat/completions`，所以 anthropic / google 100% 失败、留下空记录。现已按 `auth` 字段分发。如果你之前跑过老版本，请删掉 timing_refs.json 重跑或者只重跑这两家。

### 验证落地
```bash
grep '"sampled":' backend/app/_data/timing_refs.json | sort | uniq -c
```
应该看到 `"sampled": true,` 占多数（每家至少一行）。同时 `_provenance.sampling_required: false` 才算真生效。

### 刷新节奏
不同模型版本的 TTFT 差异很大（例如 GPT-4o vs GPT-4o-mini），建议：
- **每个 minor 版本变化重采**（如 `claude-3-5-haiku-20241022` → `claude-3-5-haiku-20251030`）
- **每 90 天刷新一次**（v16 EWMA reference 默认 `stale_after_days=90` / `discard=180`）
- 网络抖动期不采（China → 海外 TLS handshake p90 可能从 800ms 飙到 3000ms）

---

## Part 2：IRT 题库参数校准

### 现状
- `suite_v13.json` / `suite_v15.json` / `suite_v16_*.json` 每个 case 含 `irt_a` / `irt_b` / `irt_c`，但 `calibrated: false`
- 这些值是研究员凭经验填的初值，**不是从真实通过率拟合出来**的
- 影响：Theta 标准分能算出数值，但"500=平均、100=1σ"的统计语义不严格成立；CAT 自适应选题的信息量计算不准；verdict 的 `discrimination_index` 会偏

### IRT 校准三步走

**Step 1. 跑模型矩阵**
```bash
# 跑 50–100 个模型 × suite_v13 (或 v15/v16 复合套件)
python -m backend.scripts.mass_model_test --config backend/scripts/models_config.yaml
```
要求：
- ≥ 100 个不同模型（覆盖弱到强的能力范围；不仅是顶级模型）
- 每个 case 至少 20 个不同模型答过，否则 IRT 拟合不收敛

> 如果你只能拿到几十个模型，可以暂时用 `irt_data_collection.py` 的 `generate_mock_data()` 跑通流程，但生成的参数**仅限开发自测**，不要用于正式 verdict。

**Step 2. 收集响应矩阵**
```bash
python -m backend.scripts.irt_data_collection
```
输出 `(model × case)` 矩阵：每格 1=pass / 0=fail。

**Step 3. 拟合 IRT 2PL/3PL 参数 + 维度权重**
```bash
# 拟合 IRT (a, b, c) 写回 suite_*.json
python -m backend.scripts.fit_weights_v14

# 检查 R²
python -m backend.scripts.fit_weights_v14 --dry-run
```
拟合后：
- 每个 case 的 `irt_a` (区分度) / `irt_b` (难度) / `irt_c` (猜测下限) 被替换成实测值
- `calibrated: true`
- `_data/weights/scoring_weights_v14.yaml` 被更新，`_provenance.r2` 字段是拟合质量指标（>0.85 是合格、>0.95 算优秀）

### 校准前后对比验证
```bash
pytest backend/tests/test_v14_phase2.py  # 验证权重加载链路
pytest backend/tests/test_v15_phase7.py  # 验证 calibration_metrics（Brier/ECE/可靠性曲线）
```

### 维持节奏
- 每加 ≥ 20 个新 case 重拟一次
- 每加入新一代旗舰模型（如 GPT-5、Claude-4）重拟一次（顶部能力扩展会改 b 参数分布）
- 每 6 个月做一次完整重采（response style drift）

---

## Part 3：相关脚手架

### 模型清单配置
看 `backend/scripts/models_config.yaml`（如果不存在，参考 `mass_model_test.py` 里 `ModelRegistry` 类的字段定义）。每条记录：
```yaml
- model_id: gpt-4o-mini
  model_name: GPT-4o-mini
  provider: openai
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  family: gpt
```

### 检查校准状态
```bash
# 查看哪些 suite 已校准
python -c "
import json, pathlib
for p in pathlib.Path('backend/app/fixtures').glob('suite_*.json'):
    cases = json.loads(p.read_text())
    cal = sum(1 for c in cases if c.get('calibrated'))
    print(f'{p.name}: {cal}/{len(cases)} calibrated')
"

# 查看 timing_refs 状态
python -c "
import json
d = json.load(open('backend/app/_data/timing_refs.json'))
for f, v in d['families'].items():
    print(f'{f:10s} sampled={v[\"sampled\"]} n={v.get(\"sample_size\")}')
"
```

### 已知坑
1. **测 timing 时不要并发**：脚本是 `time.sleep(0.5)` 串行的，因为 TTFT 测量受 TCP 排队影响，并发会把方差放大 3–5x
2. **采样地点要稳定**：从中国到海外厂商 TLS handshake p90 可能在 600–1500ms 漂移；如果你做生产部署，建议 Inspector 与采样脚本跑在**同一可用区**
3. **prompt cache 会污染样本**：所有家族都启用了 prompt caching；脚本的 PROBE_PROMPT 是固定的，第二次起 TTFT 会异常低。已知问题，**100 样本时 cache hit 不会显著扭曲均值**，但若你只跑 5 样本就别用这数据
4. **finish_reason="length" 处理**：v16 已修复，截断响应不再被判题跳过；但 IRT 拟合时仍应丢弃 `finish_reason="length"` 的 case，否则会把"题太难"误读成"模型答错"

---

## 附：当前 placeholder 数据的代码引用

这些位置在数据采到之前会触发安全短路，**不会**让系统给出错误的 verdict：

- [layers_l18_l19.py:189-219](../llm-inspector/backend/app/predetect/layers_l18_l19.py) 的 `_load_timing_refs()` + `sampled` 检查
- [layers_l18_l19.py:347-369](../llm-inspector/backend/app/predetect/layers_l18_l19.py) 的 token 分布同样检查
- [timing_refs.json:1-9](../llm-inspector/backend/app/_data/timing_refs.json) 的 `_provenance.sampling_required: true`
- [token_dist_refs.json:1-8](../llm-inspector/backend/app/_data/token_dist_refs.json) 同上
- IRT cold-start 先验表：`backend/app/_data/probes/`（v16 Phase 5）下的 6 类 × 4 难度先验，作为 `calibrated=false` 时的 fallback

只要 `_provenance.sampling_required` 还是 `true`，就当 L18/L19 没生效。
