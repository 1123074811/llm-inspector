# LLM Inspector v1.0 — LLM 套壳检测工具

## Quick Start
- `cd llm-inspector/backend && python -m app.main` — 启动服务（默认 :8000）
- `pytest backend/tests/test_all.py` — 运行测试

## Tech Stack
- **后端**: Python stdlib（http.server + urllib + sqlite3 + dataclasses）
- **外部依赖**: `cryptography`, `numpy`, `scikit-learn`, `pytest`
- **前端**: 多文件 SPA（index.html + styles.css + app.js），纯 HTML/CSS/JS，无构建步骤
- **数据库**: SQLite WAL 模式，12 张表，线程局部连接
- **任务**: ThreadPoolExecutor(4 workers)，支持 Celery 分布式扩展

## Architecture
```
HTTP Handler (main.py) → Repository (repo.py) → Worker (worker.py)
  → Orchestrator: PreDetect(7层) → CaseExecutor → Judge(21种) → Analysis Pipeline
```
- API 路由: 正则表达式路由表，`/api/v1/` 前缀，共 16 个端点
- 预检测: 从 0 token 递增至 ~500 token，置信度 >=0.85 提前停止
- 分析: 特征提取 → 评分(v2三维) → 余弦相似度+bootstrap CI → 风险评估 → 判定

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
