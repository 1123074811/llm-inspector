# LLM Inspector v2.0 — LLM 套壳检测与能力评估工具

## Quick Start
- `cd llm-inspector/backend && python -m app.main` — 启动服务（默认 :8000）
- `pytest backend/tests/test_all.py` — 运行测试

## Tech Stack
- **后端**: Python stdlib（http.server + urllib + sqlite3 + dataclasses）
- **外部依赖**: `cryptography`, `numpy`, `scikit-learn`, `pytest`
- **前端**: 多文件 SPA（index.html + styles.css + app.js），纯 HTML/CSS/JS，无构建步骤
- **数据库**: SQLite WAL 模式，11 张表，线程局部连接
- **任务**: ThreadPoolExecutor(4 workers)，支持 Celery 分布式扩展

## Architecture
```
HTTP Handler (main.py) → Repository (repo.py) → Worker (worker.py)
  → Orchestrator: PreDetect(7层) → CaseExecutor → Judge(23种) → Analysis Pipeline
```
- API 路由: 正则表达式路由表，`/api/v1/` 前缀
- 预检测: 从 0 token 递增至 ~500 token，置信度 >=0.85 提前停止
- 分析: 特征提取(36维) → 评分(v2三维) → 余弦相似度+bootstrap CI → 风险评估 → 判定
- 测试套件: suite_v3 共 50 个用例，含 14 个分类维度

## Test Suite (suite_v3)
- **50 个测试用例**，覆盖 14 个分类：
  - protocol(2), instruction(5), system(2), param(2), reasoning(12), coding(5)
  - refusal(2), style(2), consistency(3), antispoof(3), extraction(6)
  - fingerprint(2), tool_use(3), performance(1)
- **梯度难度**: difficulty 0.3→0.95，含链式推理(3/5/8步)、数学竞赛、LRU Cache 实现
- **模型指纹探测**: 问候风格、拒绝风格指纹
- **Tool Use 测试**: 工具调用、工具选择、无需工具场景

## Scoring System (v2)
```
TotalScore = 0.50×Capability + 0.30×Authenticity + 0.20×Performance
  Capability  = 0.25×reasoning + 0.15×adversarial + 0.20×instruction
                + 0.20×coding + 0.10×safety + 0.10×protocol
  Authenticity = 0.35×similarity + 0.20×behavioral_invariant + 0.15×consistency
                 + 0.10×temperature + 0.20×usage_fingerprint
  Performance  = 0.40×speed + 0.30×stability + 0.30×cost_efficiency
```

## Benchmark System
- **基准模型 = 真实数据**: 用户将已完成的检测标记为基准，以 model_name 为唯一索引
- **相似度比对只使用基准模型** (golden_baselines 表)，不使用任何估算/假数据
- `benchmark_profiles` 表已废弃，不再参与任何逻辑
- 标记基准时用户可自定义模型名，同名基准自动覆盖

## VerdictEngine 硬规则
- 声称顶级模型但 difficulty_ceiling < 0.4 → 强制降到 50 分
- behavioral_invariant_score < 40 → 强制降到 55 分
- 声称顶级模型但 coding_score < 10 → 强制降到 45 分
- adversarial_spoof_signal_rate > 0.5 → 强制降到 45 分

## Code Conventions
- `from __future__ import annotations` 在文件顶部
- 标准库 import 在前，项目 import 用绝对路径 `from app.core.xxx`
- 命名: 文件 `snake_case`，类 `PascalCase`，私有 `_prefix`，常量 `UPPER_SNAKE`
- Handler 函数: `handle_` 前缀
- 日志: `logger = get_logger(__name__)` + structlog 风格关键字参数
- 代码分区: `# -- Section Name` 注释
- 类型注解: Python 3.10+ 风格（`str | None`）

## Testing
- pytest 测试框架，41+ 测试用例
- 测试用独立数据库 `test_inspector.db`，测试后自动清理
- 覆盖 config/security/db/seeder/judge/analysis/repo/predetect/executor/http

## Task Queue Architecture
- 本地模式: ThreadPoolExecutor（默认，零配置）
- 分布式模式: Celery + Redis（设置 CELERY_BROKER_URL 环境变量）
- 抽象接口: `app.tasks.queue.TaskQueue`
- 切换方式: `from app.tasks.worker import init_distributed_queue; init_distributed_queue()`

## Gotchas
- dev 模式下 ENCRYPTION_KEY 自动生成确定性密钥，生产环境必须手动设置
- 前端文件已拆分为 index.html + styles.css + app.js，修改时注意保持引用关系
- 数据库 schema 提供迁移函数 `migrate_json_columns_to_columns()`
- `test_runs` 表的 `evaluation_mode`, `calibration_case_id`, `scoring_profile_version`, `calibration_tag` 已迁移为独立列
- 相似度比对只使用 golden_baselines 中用户标记的基准模型，没有基准时相似度列表为空
