# LLM Inspector v1.0 — LLM 套壳检测工具

## Quick Start
- `cd llm-inspector/backend && python -m app.main` — 启动服务（默认 :8000）
- `PYTHONPATH=llm-inspector/backend python llm-inspector/backend/tests/test_all.py` — 运行测试（自制框架，非 pytest）

## Tech Stack
- **零框架后端**: Python stdlib（http.server + urllib + sqlite3 + dataclasses），唯一外部依赖 `cryptography`
- **前端**: 单文件 SPA（frontend/index.html），纯 HTML/CSS/JS，无构建步骤
- **数据库**: SQLite WAL 模式，11 张表，线程局部连接
- **任务**: ThreadPoolExecutor(4 workers)，预留 Celery 支持

## Architecture
```
HTTP Handler (main.py) → Repository (repo.py) → Worker (worker.py)
  → Orchestrator: PreDetect(5层) → CaseExecutor → Judge(9种) → Analysis Pipeline
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
- 自制测试框架（非 pytest/unittest），全局 PASS/FAIL 计数器
- 测试用独立数据库 `test_inspector.db`，测试后自动清理
- 10 个 section，~60+ 用例，覆盖 config/security/db/seeder/judge/analysis/repo/predetect/executor/http

## Gotchas
- 无 requirements.txt/pyproject.toml，依赖仅在 README.md 中记录（cryptography, numpy, scikit-learn）
- dev 模式下 ENCRYPTION_KEY 自动生成确定性密钥，生产环境必须手动设置
- 前端是单文件 1080 行，修改时注意不要破坏内联 CSS/JS 结构
