# LLM Inspector v5.0 升级方案

> **目标**: 从 v4.0 升级到 v5.0，引入自适应学习机制、增强语义理解能力、建立数据驱动的评分校准体系，实现真正的模型能力评估与真伪辨别。
>
> **核心原则**: 
> - 从规则驱动 → 数据驱动
> - 从静态阈值 → 自适应校准  
> - 从关键词匹配 → 语义嵌入理解
> - 从单一信号 → 多模态融合

---

## 目录

1. [P0 — 判题系统智能化重构](#1-p0--判题系统智能化重构)
2. [P1 — 评分体系自适应校准](#2-p1--评分体系自适应校准)
3. [P2 — 测试用例IRT校准与动态生成](#3-p2--测试用例irt校准与动态生成)
4. [P3 — 相似度引擎深度学习增强](#4-p3--相似度引擎深度学习增强)
5. [P4 — 预检测跨层信号融合](#5-p4--预检测跨层信号融合)
6. [P5 — 前端报告专业化升级](#6-p5--前端报告专业化升级)
7. [P6 — 数据基础设施重建](#7-p6--数据基础设施重建)
8. [P7 — 可解释性与审计系统](#8-p7--可解释性与审计系统)
9. [P8 — 性能与架构优化](#9-p8--性能与架构优化)
10. [P9 — 新增测试维度](#10-p9--新增测试维度)

---

## 1. P0 — 判题系统智能化重构

### 1.1 语义判题引擎升级（LLM-as-Judge 2.0）

**现状问题**: `semantic_judge` 依赖关键词覆盖率（45%阈值），对中文支持弱，无法捕捉深层语义等价

**升级方案**:

```python
# backend/app/judge/semantic_v2.py

@dataclass
class SemanticJudgeV2:
    """
    新一代语义判题引擎，结合：
    1. 本地嵌入模型（BGE-large-zh-v1.5 / GTE-large-en-v1.5）
    2. LLM-as-Judge（外部API）
    3. 结构化评分标准（Rubric-based）
    """
    
    # 本地嵌入模型（离线运行，零API成本）
    _embedding_model: Optional[SentenceTransformer] = None
    _cache_dir: str = field(default="./models")
    
    # Rubric维度定义
    RUBRIC_DIMENSIONS = {
        "correctness": {
            "weight": 0.35,
            "criteria": ["答案事实准确", "推理过程正确", "结论合理"],
            "scoring": "0-10",
        },
        "completeness": {
            "weight": 0.25,
            "criteria": ["覆盖问题要点", "没有遗漏关键信息"],
            "scoring": "0-10",
        },
        "clarity": {
            "weight": 0.20,
            "criteria": ["表达清晰", "结构合理", "易于理解"],
            "scoring": "0-10",
        },
        "reasoning_quality": {
            "weight": 0.20,
            "criteria": ["逻辑严密", "论证充分", "没有跳跃"],
            "scoring": "0-10",
        },
    }
    
    def judge(
        self,
        prompt: str,
        response: str,
        reference: str | None = None,
        rubric: dict | None = None,
    ) -> SemanticJudgeResult:
        """
        三层级联评判：
        1. 快速过滤：嵌入相似度（本地，<10ms）
        2. 结构化评分：LLM按维度打分
        3. 一致性校验：多轮采样降低方差
        """
        
        # Layer 1: 本地嵌入过滤
        emb_score = self._embedding_similarity(prompt, response, reference)
        if emb_score < 0.3:  # 明显不相关，直接失败
            return SemanticJudgeResult(
                passed=False,
                score=emb_score * 100,
                confidence=0.8,
                method="embedding_filter",
                reasoning="与问题语义关联度过低",
            )
        
        # Layer 2: LLM结构化评分
        llm_scores = self._llm_rubric_scoring(prompt, response, reference)
        
        # Layer 3: 多轮一致性校验
        if llm_scores.confidence < 0.7:
            llm_scores = self._multi_round_consensus(
                prompt, response, reference, n_rounds=3
            )
        
        # 综合评分
        final_score = self._weighted_rubric_score(llm_scores)
        
        return SemanticJudgeResult(
            passed=final_score >= 60,  # 从45提升到60
            score=final_score,
            confidence=llm_scores.confidence,
            method="semantic_v2",
            dimensions=llm_scores.dimension_scores,
            reasoning=llm_scores.reasoning,
        )
    
    def _embedding_similarity(
        self, prompt: str, response: str, reference: str | None
    ) -> float:
        """基于本地嵌入模型的语义相似度计算"""
        if self._embedding_model is None:
            # 延迟加载模型
            self._embedding_model = SentenceTransformer(
                "BAAI/bge-large-zh-v1.5",
                cache_folder=self._cache_dir,
                device="cpu",  # 可根据环境改为cuda
            )
        
        # 编码
        query_emb = self._embedding_model.encode(prompt, normalize_embeddings=True)
        resp_emb = self._embedding_model.encode(response, normalize_embeddings=True)
        
        # 计算余弦相似度
        sim = float(np.dot(query_emb, resp_emb))
        
        # 如果有参考答案，计算与参考答案的相似度
        if reference:
            ref_emb = self._embedding_model.encode(reference, normalize_embeddings=True)
            ref_sim = float(np.dot(resp_emb, ref_emb))
            # 取两者中较高者（响应既应回答问题，也应接近参考答案）
            sim = max(sim, ref_sim * 0.8)  # 参考答案权重略降
        
        return sim
```

**技术细节**:
- **依赖**: `sentence-transformers>=2.3.0`, `torch>=2.0.0`
- **模型**: BGE-large-zh-v1.5 (中文) / GTE-large-en-v1.5 (英文)
- **资源**: 模型大小约1.3GB，内存占用约2GB
- **延迟**: 本地嵌入<50ms，LLM评分~500ms

---

### 1.2 幻觉检测增强（Uncertainty-Aware Hallucination Detection）

**现状问题**: `_hallucination_detect` 依赖规则和关键词，无法检测新型幻觉模式

**升级方案**:

```python
# backend/app/judge/methods.py — _hallucination_detect_v2

def _hallucination_detect_v2(text: str, params: dict) -> tuple[bool, dict]:
    """
    v5 幻觉检测：结合不确定性量化和事实核查
    """
    fake_entity = params.get("fake_entity", "")
    fake_entity_2 = params.get("fake_entity_2", "")
    expect_refusal = params.get("expect_refusal", True)
    
    # 1. 基于规则的信号检测（原有逻辑，优化精度）
    rule_signals = _detect_hallucination_signals(text, fake_entity, fake_entity_2)
    
    # 2. 不确定性量化（新增）
    uncertainty = _quantify_uncertainty(text)
    
    # 3. 知识图谱事实核查（新增，可选）
    if params.get("enable_fact_check") and settings.KNOWLEDGE_GRAPH_URL:
        fact_check_result = _check_against_knowledge_graph(text, fake_entity)
        rule_signals["fact_check"] = fact_check_result
    
    # 4. 置信度校准
    is_hallucinating = _calibrate_hallucination_verdict(
        rule_signals, uncertainty, expect_refusal
    )
    
    passed = not is_hallucinating if expect_refusal else not is_hallucinating
    
    return passed, {
        "fake_entity": fake_entity,
        "rule_signals": rule_signals,
        "uncertainty_score": uncertainty.score,
        "uncertainty_markers": uncertainty.markers,
        "confidence_calibration": uncertainty.calibration,
        "is_hallucinating": is_hallucinating,
        "passed": passed,
    }


def _quantify_uncertainty(text: str) -> UncertaintyMetrics:
    """量化文本中的不确定性表达"""
    
    # 明确的不确定性表达（正面信号）
    explicit_markers = [
        r"(?:我|笔者|模型).{0,3}(?:不确定|不清楚|不知道|不了解)",
        r"(?:无法|不能|难以).{0,5}(?:确认|证实|核实|查证)",
        r"(?:信息|知识).{0,5}(?:截至|截止|更新至)",
        r"(?:可能|也许|大概|或许|恐怕)",
        r"(?:据我所知|据我了解|我记得)",
        r"(?:没有|未找到).{0,5}(?:相关|确切).{0,3}(?:信息|资料|记录)",
        r"(?:I'm not sure|I don't know|I cannot confirm)",
        r"(?:as of my last update|to the best of my knowledge)",
        r"(?:possibly|perhaps|maybe|probably|likely)",
    ]
    
    # 虚假的确定性表达（负面信号 - 过度自信）
    false_confidence = [
        r"(?:毫无疑问|绝对|肯定|一定|必然|毫无疑问地)",
        r"(?:这是事实|这是真相|这是确定的)",
        r"(?:众所周知|大家公认|普遍接受)",  # 当描述虚假实体时
        r"(?:certainly|absolutely|definitely|without doubt)",
    ]
    
    # 计算不确定性分数
    explicit_count = sum(1 for p in explicit_markers if re.search(p, text, re.I))
    false_conf_count = sum(1 for p in false_confidence if re.search(p, text, re.I))
    
    # 校准：高不确定性表达 + 低虚假确定性 = 好的不确定性管理
    uncertainty_score = min(1.0, explicit_count / 3) - min(0.5, false_conf_count / 2)
    
    # 置信度校准：模型是否恰当地表达了不确定性
    calibration = "good" if uncertainty_score > 0.3 else "poor" if uncertainty_score < 0 else "neutral"
    
    return UncertaintyMetrics(
        score=max(0, uncertainty_score),
        markers={
            "explicit": explicit_count,
            "false_confidence": false_conf_count,
        },
        calibration=calibration,
    )
```

---

### 1.3 代码判题沙箱增强

**现状问题**: `_code_execution` 仅支持Python，无资源限制，测试用例覆盖不足

**升级方案**:

```python
# backend/app/judge/code_sandbox.py

@dataclass
class SandboxConfig:
    """代码执行沙箱配置"""
    language: str = "python"
    timeout_sec: float = 5.0
    max_memory_mb: int = 256
    max_output_chars: int = 10000
    allowed_imports: list[str] = field(default_factory=list)
    disallowed_imports: list[str] = field(default_factory=lambda: [
        "os", "sys", "subprocess", "socket", "requests", "urllib"
    ])


class SecureCodeSandbox:
    """
    安全代码执行环境，支持多语言
    """
    
    SUPPORTED_LANGUAGES = {
        "python": PythonExecutor(),
        "javascript": JavaScriptExecutor(),  # 使用deno
        "cpp": CppExecutor(),  # 使用g++ + seccomp
    }
    
    def execute(
        self,
        code: str,
        test_cases: list[dict],
        config: SandboxConfig | None = None,
    ) -> CodeExecutionResult:
        """
        在沙箱中执行代码并运行测试用例
        """
        config = config or SandboxConfig()
        
        executor = self.SUPPORTED_LANGUAGES.get(config.language)
        if not executor:
            return CodeExecutionResult(
                passed=False,
                error=f"Unsupported language: {config.language}",
            )
        
        results = []
        for tc in test_cases:
            result = executor.run(
                code=code,
                test_input=tc["input"],
                expected_output=tc["expected"],
                config=config,
            )
            results.append(result)
        
        # 计算部分通过分数
        pass_rate = sum(1 for r in results if r.passed) / len(results)
        
        # 额外评分维度
        efficiency_score = self._score_efficiency(results)
        robustness_score = self._score_robustness(results, test_cases)
        
        return CodeExecutionResult(
            passed=pass_rate >= 0.8,
            pass_rate=pass_rate,
            efficiency_score=efficiency_score,
            robustness_score=robustness_score,
            test_results=results,
            syntax_valid=self._check_syntax(code, config.language),
        )
```

---

## 2. P1 — 评分体系自适应校准

### 2.1 数据驱动的权重校准（Bayesian Weight Calibration）

**现状问题**: `ScoreCardCalculator` 的权重是硬编码的经验值，未随数据积累而优化

**升级方案**:

```python
# backend/app/analysis/adaptive_scoring.py

class AdaptiveScoreCalibrator:
    """
    基于历史检测数据的贝叶斯权重校准
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self._load_historical_data()
    
    def _load_historical_data(self):
        """加载历史检测数据用于校准"""
        query = """
        SELECT 
            claimed_model,
            final_verdict,
            feature_vector,
            dimension_scores,
            ground_truth_verified  -- 人工验证的真实标签
        FROM detection_runs
        WHERE created_at >= date('now', '-90 days')
        AND status = 'completed'
        """
        self.historical_data = self.db.execute(query).fetchall()
    
    def calibrate_capability_weights(
        self,
        claimed_model_family: str,
    ) -> dict[str, float]:
        """
        基于历史数据校准能力维度权重
        
        优化目标：最大化真实模型的区分度（AUC）
        """
        # 筛选同家族模型的历史数据
        family_data = [
            r for r in self.historical_data
            if r["claimed_model"].startswith(claimed_model_family)
        ]
        
        if len(family_data) < 50:  # 数据不足，使用默认权重
            return ScoreCardCalculator.DEFAULT_CAPABILITY_WEIGHTS
        
        # 使用梯度下降优化权重
        # 目标函数：最大化真实模型的AUC，最小化误判率
        
        from scipy.optimize import minimize
        
        def objective(weights):
            """负AUC作为损失函数"""
            weighted_scores = []
            true_labels = []
            
            for record in family_data:
                dim_scores = json.loads(record["dimension_scores"])
                # 计算加权总分
                weighted = sum(
                    weights.get(dim, 0) * score
                    for dim, score in dim_scores.items()
                )
                weighted_scores.append(weighted)
                # 真实标签：ground_truth_verified 或 verdict推断
                is_real = record["final_verdict"] == "trusted"
                true_labels.append(1 if is_real else 0)
            
            # 计算AUC
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(true_labels, weighted_scores)
            except ValueError:
                auc = 0.5  # 无法计算时返回随机水平
            
            return -auc  # 最小化负AUC = 最大化AUC
        
        # 约束：权重和为1，每个权重>=0.01
        constraints = [
            {"type": "eq", "fun": lambda w: sum(w.values()) - 1.0},
        ]
        bounds = [(0.01, 0.5) for _ in range(len(dimensions))]
        
        # 优化
        result = minimize(
            objective,
            x0=list(ScoreCardCalculator.DEFAULT_CAPABILITY_WEIGHTS.values()),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        
        # 返回优化后的权重
        optimized = dict(zip(dimensions, result.x))
        
        # 记录校准历史
        self._log_calibration(claimed_model_family, optimized, -result.fun)
        
        return optimized
    
    def detect_score_drift(
        self,
        current_scores: dict,
        baseline_scores: dict,
        threshold: float = 0.15,
    ) -> list[ScoreDriftAlert]:
        """检测分数相对于基线的漂移"""
        alerts = []
        
        for dimension, current in current_scores.items():
            baseline = baseline_scores.get(dimension, current)
            drift = abs(current - baseline) / max(baseline, 1)
            
            if drift > threshold:
                alerts.append(ScoreDriftAlert(
                    dimension=dimension,
                    baseline=baseline,
                    current=current,
                    drift_pct=drift * 100,
                    severity="high" if drift > 0.3 else "medium",
                ))
        
        return alerts
```

---

### 2.2 评分置信度量化（Score Confidence Intervals）

**现状问题**: 分数是点估计，无置信区间，用户无法判断分数可靠性

**升级方案**:

```python
# backend/app/analysis/confidence.py

@dataclass
class ScoreWithConfidence:
    """带置信区间的分数"""
    point_estimate: float  # 点估计值
    ci_95_low: float       # 95%置信区间下限
    ci_95_high: float      # 95%置信区间上限
    std_error: float       # 标准误
    sample_size: int       # 样本量
    reliability: str       # 可靠性等级：high/medium/low/insufficient
    
    def to_dict(self) -> dict:
        return {
            "score": round(self.point_estimate, 1),
            "ci_95": [round(self.ci_95_low, 1), round(self.ci_95_high, 1)],
            "std_error": round(self.std_error, 2),
            "n_samples": self.sample_size,
            "reliability": self.reliability,
        }


class ScoreConfidenceEstimator:
    """
    基于Bootstrap和IRT模型的评分置信度估计
    """
    
    def estimate_confidence(
        self,
        case_results: list[CaseResult],
        dimension: str,
        n_bootstrap: int = 1000,
    ) -> ScoreWithConfidence:
        """
        估计维度分数的置信区间
        """
        # 筛选相关用例
        relevant_cases = [
            r for r in case_results
            if (r.case.dimension or r.case.category) == dimension
        ]
        
        if not relevant_cases:
            return ScoreWithConfidence(
                point_estimate=50.0,
                ci_95_low=40.0,
                ci_95_high=60.0,
                std_error=5.0,
                sample_size=0,
                reliability="insufficient",
            )
        
        # Bootstrap重采样
        bootstrap_scores = []
        n_cases = len(relevant_cases)
        
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            # 有放回采样
            sampled_indices = rng.choice(n_cases, size=n_cases, replace=True)
            sampled_cases = [relevant_cases[i] for i in sampled_indices]
            
            # 计算该次采样的分数
            score = self._calculate_weighted_pass_rate(sampled_cases)
            bootstrap_scores.append(score)
        
        # 计算统计量
        point_estimate = np.mean(bootstrap_scores)
        std_error = np.std(bootstrap_scores, ddof=1)
        ci_95_low, ci_95_high = np.percentile(bootstrap_scores, [2.5, 97.5])
        
        # 可靠性等级
        reliability = self._assess_reliability(
            n_cases, std_error, bootstrap_scores
        )
        
        return ScoreWithConfidence(
            point_estimate=float(point_estimate),
            ci_95_low=float(ci_95_low),
            ci_95_high=float(ci_95_high),
            std_error=float(std_error),
            sample_size=n_cases,
            reliability=reliability,
        )
    
    def _assess_reliability(
        self,
        n_cases: int,
        std_error: float,
        bootstrap_scores: list[float],
    ) -> str:
        """评估分数可靠性等级"""
        
        # 检查Bootstrap分布的稳定性
        score_variance = np.var(bootstrap_scores)
        
        if n_cases < 5 or std_error > 15:
            return "insufficient"
        elif n_cases >= 10 and std_error < 5 and score_variance < 100:
            return "high"
        elif n_cases >= 8 and std_error < 10:
            return "medium"
        else:
            return "low"
```

---

## 3. P2 — 测试用例IRT校准与动态生成

### 3.1 IRT 2PL模型实现（区分度+难度）

**现状问题**: 测试用例的难度系数是人工设定，无区分度度量，无法自适应选题

**升级方案**:

```python
# backend/app/analysis/irt_engine.py

@dataclass
class IRTItemStats:
    """IRT 2PL模型参数"""
    item_id: str
    a: float  # 区分度 (discrimination)
    b: float  # 难度 (difficulty)
    c: float = 0.0  # 猜测参数（可选3PL）
    
    information: float = 0.0  # 项目信息函数值
    sample_size: int = 0
    last_calibrated: str = ""


class IRTEngine:
    """
    IRT (Item Response Theory) 引擎
    用于：
    1. 校准测试用例参数（a, b）
    2. 估计被测模型能力(theta)
    3. 自适应选题(Adaptive Testing)
    """
    
    def calibrate_items(
        self,
        historical_results: list[dict],
    ) -> dict[str, IRTItemStats]:
        """
        基于历史检测数据校准测试用例参数
        
        使用EM算法估计a(区分度)和b(难度)
        """
        # 构建响应矩阵
        # 行：模型实例，列：测试用例，值：通过(1)/失败(0)
        
        response_matrix = self._build_response_matrix(historical_results)
        
        # EM算法估计参数
        item_params = {}
        
        for item_id in response_matrix.columns:
            # 提取该题目的响应数据
            responses = response_matrix[item_id].values
            abilities = response_matrix["theta_estimate"].values
            
            # 最大似然估计 a 和 b
            def neg_log_likelihood(params):
                a, b = params
                predictions = 1 / (1 + np.exp(-a * (abilities - b)))
                # 添加正则化防止过拟合
                reg = 0.01 * (a ** 2 + b ** 2)
                return -np.sum(
                    responses * np.log(predictions + 1e-10) +
                    (1 - responses) * np.log(1 - predictions + 1e-10)
                ) + reg
            
            # 优化
            from scipy.optimize import minimize
            result = minimize(
                neg_log_likelihood,
                x0=[1.0, 0.0],  # 初始值：中等区分度，平均难度
                bounds=[(0.1, 3.0), (-3.0, 3.0)],
                method="L-BFGS-B",
            )
            
            a_opt, b_opt = result.x
            
            # 计算项目信息函数（在theta=0处的信息）
            p = 1 / (1 + np.exp(-a_opt * (0 - b_opt)))
            info = a_opt ** 2 * p * (1 - p)
            
            item_params[item_id] = IRTItemStats(
                item_id=item_id,
                a=round(a_opt, 3),
                b=round(b_opt, 3),
                information=round(info, 4),
                sample_size=int(response_matrix[item_id].notna().sum()),
                last_calibrated=datetime.now().isoformat(),
            )
        
        return item_params
    
    def select_next_item(
        self,
        current_theta: float,
        available_items: list[IRTItemStats],
        administered_items: list[str],
    ) -> IRTItemStats | None:
        """
        自适应选题：选择能提供最大信息量的未使用题目
        
        信息量函数：I(theta) = a^2 * P(theta) * (1 - P(theta))
        其中 P(theta) = 1 / (1 + exp(-a*(theta-b)))
        """
        # 过滤已使用题目
        available = [i for i in available_items if i.item_id not in administered_items]
        
        if not available:
            return None
        
        # 计算每个可用题目在当前theta处的信息量
        best_item = None
        max_info = -1
        
        for item in available:
            # 2PL模型的信息函数
            p = 1 / (1 + math.exp(-item.a * (current_theta - item.b)))
            info = item.a ** 2 * p * (1 - p)
            
            if info > max_info:
                max_info = info
                best_item = item
        
        return best_item
    
    def estimate_ability(
        self,
        responses: list[tuple[str, bool]],  # (item_id, passed)
        item_params: dict[str, IRTItemStats],
    ) -> float:
        """
        最大似然估计被测模型的能力参数theta
        """
        def neg_log_likelihood(theta):
            ll = 0
            for item_id, passed in responses:
                item = item_params.get(item_id)
                if not item:
                    continue
                # 2PL模型概率
                p = 1 / (1 + math.exp(-item.a * (theta - item.b)))
                # 对数似然
                if passed:
                    ll += math.log(p + 1e-10)
                else:
                    ll += math.log(1 - p + 1e-10)
            return -ll
        
        # 优化求解最大似然
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(neg_log_likelihood, bounds=(-4, 4), method="bounded")
        
        return round(result.x, 4)
```

---

### 3.2 同构变体自动生成（Isomorphic Variant Generator 2.0）

**现状问题**: 同构变体生成规则有限，无法覆盖复杂推理题的语义等价转换

**升级方案**:

```python
# backend/app/tools/variant_generator_v2.py

class IsomorphicVariantGenerator:
    """
    v5 同构变体生成器：支持语义级等价转换
    """
    
    TRANSFORMATION_RULES = {
        "numerical": {
            # 数值等价转换
            "scale": lambda x, factor: x * factor,
            "offset": lambda x, delta: x + delta,
            "unit_convert": lambda x, from_unit, to_unit: convert_units(x, from_unit, to_unit),
        },
        "structural": {
            # 结构等价转换
            "reorder": lambda problem: reorder_constraints(problem),
            "negation": lambda problem: negate_question(problem),
            "analogy": lambda problem: convert_to_analogy(problem),
        },
        "semantic": {
            # 语义等价转换（使用LLM）
            "rephrase": lambda text: llm_rephrase(text, style="formal"),
            "domain_shift": lambda problem, new_domain: shift_domain(problem, new_domain),
            "complexity_adjust": lambda problem, delta: adjust_complexity(problem, delta),
        },
    }
    
    def generate_variants(
        self,
        base_case: TestCase,
        n_variants: int = 3,
        verification_level: str = "strict",
    ) -> list[TestCase]:
        """
        生成经过验证的同构变体
        """
        variants = []
        used_transforms = set()
        
        for i in range(n_variants):
            # 选择变换策略
            transform = self._select_transformation(
                base_case, used_transforms
            )
            used_transforms.add(transform["name"])
            
            # 应用变换
            variant = self._apply_transformation(base_case, transform)
            
            # 验证等价性
            if verification_level == "strict":
                is_valid = self._verify_equivalence_strict(base_case, variant)
            elif verification_level == "semantic":
                is_valid = self._verify_equivalence_semantic(base_case, variant)
            else:
                is_valid = self._verify_equivalence_basic(base_case, variant)
            
            if is_valid:
                variant.id = f"{base_case.id}_iso_v{i+1}"
                variant.params["_meta"]["variant_of"] = base_case.id
                variant.params["_meta"]["transform"] = transform["name"]
                variants.append(variant)
        
        return variants
    
    def _verify_equivalence_semantic(
        self,
        original: TestCase,
        variant: TestCase,
    ) -> bool:
        """
        使用语义嵌入验证变体与原题等价
        """
        # 1. 嵌入相似度检查
        orig_emb = self._embed(original.user_prompt)
        var_emb = self._embed(variant.user_prompt)
        
        similarity = cosine_similarity(orig_emb, var_emb)
        if similarity < 0.85:  # 语义相似度阈值
            return False
        
        # 2. 关键约束保持检查
        orig_constraints = self._extract_constraints(original)
        var_constraints = self._extract_constraints(variant)
        
        if set(orig_constraints) != set(var_constraints):
            return False
        
        # 3. 答案等价性检查（使用ground truth solver）
        if original.params.get("has_deterministic_answer"):
            orig_answer = self._solve_deterministic(original)
            var_answer = self._solve_deterministic(variant)
            if orig_answer != var_answer:
                return False
        
        return True
```

---

## 4. P3 — 相似度引擎深度学习增强

### 4.1 对比学习特征提取器

**现状问题**: 相似度计算依赖手工特征，无法捕捉深层行为模式

**升级方案**:

```python
# backend/app/analysis/neural_similarity.py

class BehavioralEmbeddingExtractor:
    """
    基于对比学习的模型行为嵌入提取器
    """
    
    def __init__(self, model_path: str | None = None):
        """
        加载预训练的对比学习模型
        
        模型架构：
        - 输入：行为特征向量（归一化后的FEATURE_ORDER）
        - 编码器：3层MLP with residual connections
        - 输出：128维行为嵌入（L2归一化）
        """
        self.encoder = self._load_encoder(model_path)
    
    def extract_embedding(
        self,
        features: dict[str, float],
        case_results: list[CaseResult],
    ) -> np.ndarray:
        """
        提取模型行为嵌入
        """
        # 1. 构建特征向量
        feature_vec = self._build_feature_vector(features)
        
        # 2. 添加时序模式特征
        temporal_features = self._extract_temporal_patterns(case_results)
        
        # 3. 添加风格指纹
        style_fingerprint = self._extract_style_fingerprint(case_results)
        
        # 4. 合并并编码
        combined = np.concatenate([
            feature_vec,
            temporal_features,
            style_fingerprint,
        ])
        
        # 5. 神经网络编码
        with torch.no_grad():
            embedding = self.encoder(
                torch.FloatTensor(combined).unsqueeze(0)
            ).squeeze().numpy()
        
        # L2归一化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def compute_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        """计算余弦相似度"""
        return float(np.dot(embedding_a, embedding_b))
    
    def _extract_temporal_patterns(
        self,
        case_results: list[CaseResult],
    ) -> np.ndarray:
        """
        提取时序行为模式特征
        
        包括：
        - TTFT分布的统计特征
        - 延迟-长度关系曲线
        - token生成速率模式
        """
        patterns = []
        
        # TTFT分布特征
        ttfts = [
            s.response.first_token_ms
            for r in case_results
            for s in r.samples
            if s.response.first_token_ms
        ]
        if ttfts:
            patterns.extend([
                np.mean(ttfts) / 1000,  # 秒为单位
                np.std(ttfts) / 1000,
                np.percentile(ttfts, 25) / 1000,
                np.percentile(ttfts, 75) / 1000,
            ])
        else:
            patterns.extend([0, 0, 0, 0])
        
        # 延迟-长度关系
        lat_len_pairs = [
            (s.response.latency_ms, len(s.response.content or ""))
            for r in case_results
            for s in r.samples
            if s.response.latency_ms and s.response.content
        ]
        if len(lat_len_pairs) >= 5:
            lats, lens = zip(*lat_len_pairs)
            correlation = np.corrcoef(lats, lens)[0, 1]
            patterns.extend([correlation, np.mean(lats) / np.mean(lens)])
        else:
            patterns.extend([0, 0])
        
        return np.array(patterns, dtype=np.float32)
```

---

### 4.2 多模态相似度融合

```python
# backend/app/analysis/multi_modal_similarity.py

class MultiModalSimilarityFusion:
    """
    融合多维度相似度信号
    """
    
    def compute_fused_similarity(
        self,
        target_run: DetectionRun,
        benchmark_run: DetectionRun,
    ) -> FusedSimilarityResult:
        """
        计算融合相似度
        """
        # 1. 行为向量相似度（传统）
        behavioral_sim = self._behavioral_cosine_similarity(
            target_run.features, benchmark_run.features
        )
        
        # 2. 神经网络嵌入相似度（新增）
        neural_sim = self._neural_embedding_similarity(
            target_run.behavioral_embedding,
            benchmark_run.behavioral_embedding,
        )
        
        # 3. 响应风格相似度（编辑距离-based）
        style_sim = self._response_style_similarity(
            target_run.case_results,
            benchmark_run.case_results,
        )
        
        # 4. 时间指纹相似度（延迟模式）
        temporal_sim = self._temporal_pattern_similarity(
            target_run.case_results,
            benchmark_run.case_results,
        )
        
        # 5. 融合（使用 learned weights 或 attention）
        similarities = {
            "behavioral": behavioral_sim,
            "neural": neural_sim,
            "style": style_sim,
            "temporal": temporal_sim,
        }
        
        # 自适应权重（基于各信号的可靠性）
        weights = self._adaptive_fusion_weights(similarities)
        
        fused = sum(
            weights.get(modality, 0.25) * score
            for modality, score in similarities.items()
        )
        
        return FusedSimilarityResult(
            fused_score=fused,
            component_scores=similarities,
            weights=weights,
            confidence=self._compute_fusion_confidence(similarities, weights),
        )
```

---

## 5. P4 — 预检测跨层信号融合

### 5.1 贝叶斯网络信号融合

**现状问题**: 预检测各层独立运行，信号间矛盾未调和，置信度计算简单

**升级方案**:

```python
# backend/app/predetect/bayesian_fusion.py

class BayesianSignalFusion:
    """
    使用贝叶斯网络融合多层预检测信号
    """
    
    # 定义信号间的依赖关系
    NETWORK_STRUCTURE = {
        "root": ["platform_type"],  # 隐藏变量：真实平台类型
        "evidence_nodes": [
            "http_headers",
            "error_shape",
            "model_field",
            "identity_probe",
            "tokenizer_fingerprint",
            "routing_detected",
        ],
        "dependencies": {
            "http_headers": ["platform_type"],
            "error_shape": ["platform_type"],
            "model_field": ["platform_type", "routing_detected"],
            "identity_probe": ["platform_type", "routing_detected"],
            "tokenizer_fingerprint": ["platform_type"],
            "routing_detected": ["platform_type"],
        },
    }
    
    # 条件概率表（基于历史数据学习）
    CPT_PRIOR = {
        "openai_direct": 0.25,
        "azure_openai": 0.15,
        "anthropic_direct": 0.15,
        "proxy_wrapper": 0.30,
        "unknown": 0.15,
    }
    
    def fuse_signals(
        self,
        layer_results: list[LayerResult],
    ) -> FusionResult:
        """
        贝叶斯融合所有层信号
        """
        # 构建证据
        evidence = self._extract_evidence(layer_results)
        
        # 计算后验概率 P(Platform | Evidence)
        posteriors = {}
        for platform in self.CPT_PRIOR:
            # P(Platform | E) ∝ P(E | Platform) * P(Platform)
            likelihood = self._compute_likelihood(evidence, platform)
            prior = self.CPT_PRIOR[platform]
            posteriors[platform] = likelihood * prior
        
        # 归一化
        total = sum(posteriors.values())
        posteriors = {k: v / total for k, v in posteriors.items()}
        
        # 最可能的平台
        best_platform = max(posteriors, key=posteriors.get)
        confidence = posteriors[best_platform]
        
        # 信号一致性分析
        consistency = self._analyze_signal_consistency(evidence, posteriors)
        
        return FusionResult(
            identified_platform=best_platform,
            confidence=confidence,
            posterior_distribution=posteriors,
            evidence_weights=self._compute_evidence_weights(evidence, posteriors),
            inconsistencies=consistency.inconsistencies if not consistency.is_consistent else [],
        )
    
    def _compute_likelihood(
        self,
        evidence: dict,
        platform: str,
    ) -> float:
        """
        计算似然 P(Evidence | Platform)
        
        使用朴素贝叶斯假设（条件独立简化）
        """
        likelihood = 1.0
        
        for evidence_type, value in evidence.items():
            # 查询条件概率 P(Evidence=value | Platform)
            prob = self._query_cpt(evidence_type, value, platform)
            likelihood *= prob
        
        return likelihood
    
    def _analyze_signal_consistency(
        self,
        evidence: dict,
        posteriors: dict[str, float],
    ) -> ConsistencyAnalysis:
        """
        分析各层信号的一致性
        
        检测矛盾信号，识别可能的模型混合/路由
        """
        inconsistencies = []
        
        # 检查是否有信号与最大后验矛盾
        best_platform = max(posteriors, key=posteriors.get)
        
        for evidence_type, value in evidence.items():
            # 该证据对各个平台的支持度
            support = {
                platform: self._query_cpt(evidence_type, value, platform)
                for platform in posteriors
            }
            
            best_supported = max(support, key=support.get)
            if best_supported != best_platform and support[best_supported] > 0.7:
                inconsistencies.append({
                    "evidence": evidence_type,
                    "value": value,
                    "suggests": best_supported,
                    "confidence": support[best_supported],
                    "contradicts": best_platform,
                })
        
        # 如果有多个强矛盾信号，可能存在模型混合
        is_mixed = len([i for i in inconsistencies if i["confidence"] > 0.8]) >= 2
        
        return ConsistencyAnalysis(
            is_consistent=len(inconsistencies) == 0,
            inconsistencies=inconsistencies,
            possible_mixture=is_mixed,
        )
```

---

## 6. P5 — 前端报告专业化升级

### 6.1 交互式可视化报告

```javascript
// frontend/components/InteractiveReport.jsx

function InteractiveReport({ report }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedDimension, setSelectedDimension] = useState(null);
  
  return (
    <div className="report-container">
      {/* 顶部信任评级卡片 */}
      <TrustVerdictCard 
        verdict={report.verdict}
        confidence={report.confidence_intervals}
      />
      
      {/* 导航标签 */}
      <TabBar active={activeTab} onChange={setActiveTab}>
        <Tab id="overview">总览</Tab>
        <Tab id="dimensions">维度详情</Tab>
        <Tab id="similarity">相似度分析</Tab>
        <Tab id="evidence">证据链</Tab>
        <Tab id="raw">原始数据</Tab>
      </TabBar>
      
      {/* 内容区域 */}
      {activeTab === 'overview' && (
        <OverviewTab report={report} />
      )}
      
      {activeTab === 'dimensions' && (
        <DimensionsTab 
          dimensions={report.dimensions}
          onSelect={setSelectedDimension}
          selected={selectedDimension}
        />
      )}
      
      {activeTab === 'similarity' && (
        <SimilarityTab 
          similarities={report.similarity}
          radarData={report.radar_chart}
        />
      )}
      
      {/* ... */}
    </div>
  );
}

// 置信区间可视化组件
function ConfidenceIntervalViz({ score, ciLow, ciHigh, reliability }) {
  const reliabilityColors = {
    high: '#22c55e',
    medium: '#f59e0b',
    low: '#ef4444',
    insufficient: '#6b7280',
  };
  
  return (
    <div className="ci-visualization">
      <div className="score-point" 
           style={{ left: `${score}%`, 
                   background: reliabilityColors[reliability] }}>
        {score.toFixed(1)}
      </div>
      <div className="ci-bar">
        <div className="ci-range" 
             style={{ 
               left: `${ciLow}%`, 
               width: `${ciHigh - ciLow}%`,
               background: reliabilityColors[reliability] + '40',  // 40 = 25% opacity
             }}>
        </div>
      </div>
      <div className="ci-labels">
        <span>95% CI: [{ciLow.toFixed(1)}, {ciHigh.toFixed(1)}]</span>
        <span className={`reliability-${reliability}`}>
          可靠性: {reliability}
        </span>
      </div>
    </div>
  );
}
```

---

### 6.2 证据链可视化

```javascript
// 证据链图形化展示
function EvidenceChainViz({ evidenceChain }) {
  return (
    <div className="evidence-chain">
      {evidenceChain.map((evidence, idx) => (
        <EvidenceNode 
          key={idx}
          phase={evidence.phase}
          signal={evidence.signal}
          severity={evidence.severity}
          confidence={evidence.confidence}
          details={evidence.details}
        />
      ))}
    </div>
  );
}

function EvidenceNode({ phase, signal, severity, confidence, details }) {
  const severityIcons = {
    critical: '🔴',
    warn: '🟡',
    info: '🔵',
  };
  
  return (
    <div className={`evidence-node severity-${severity}`}>
      <div className="node-header">
        <span className="phase-badge">{phase}</span>
        <span className="severity-icon">{severityIcons[severity]}</span>
      </div>
      <div className="node-body">
        <div className="signal-text">{signal}</div>
        {confidence && (
          <div className="confidence-bar">
            <div 
              className="confidence-fill"
              style={{ width: `${confidence}%` }}
            />
            <span>{confidence.toFixed(0)}%</span>
          </div>
        )}
      </div>
      {details && (
        <div className="node-details">
          <pre>{JSON.stringify(details, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
```

---

## 7. P6 — 数据基础设施重建

### 7.1 特征值数据库（从硬编码到数据驱动）

**现状问题**: `GLOBAL_FEATURE_MEANS` 和 `KNOWN_TTFT_BASELINES` 仍是硬编码经验值

**升级方案**:

```python
# backend/app/repository/feature_stats.py

class FeatureStatisticsRepository:
    """
    特征统计值的数据库存储与自动更新
    替代原有的硬编码 GLOBAL_FEATURE_MEANS
    """
    
    TABLE_SCHEMA = """
    CREATE TABLE feature_statistics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT NOT NULL,
        model_family TEXT,  -- NULL表示全局统计
        model_name TEXT,    -- 具体模型，如gpt-4o-2024-08-06
        
        -- 统计值
        mean REAL,
        median REAL,
        std_dev REAL,
        p5 REAL,   -- 5th percentile
        p95 REAL,  -- 95th percentile
        
        -- 样本信息
        sample_count INTEGER,
        sample_sources TEXT,  -- JSON: ["run_id1", "run_id2", ...]
        
        -- 时间戳
        calculated_at TEXT,
        expires_at TEXT,  -- 统计值过期时间，需要重新计算
        
        -- 元数据
        data_quality_tag TEXT,  -- "golden", "verified", "estimated"
        notes TEXT,
        
        UNIQUE(feature_name, model_family, model_name)
    );
    
    CREATE INDEX idx_feature_name ON feature_statistics(feature_name);
    CREATE INDEX idx_model_family ON feature_statistics(model_family);
    CREATE INDEX idx_expires ON feature_statistics(expires_at);
    """
    
    def get_feature_mean(
        self,
        feature_name: str,
        model_family: str | None = None,
        fallback_to_global: bool = True,
    ) -> float | None:
        """
        获取特征均值（替代原有的字典查找）
        """
        query = """
        SELECT mean, median, std_dev, sample_count, data_quality_tag
        FROM feature_statistics
        WHERE feature_name = ?
        AND (model_family = ? OR (? IS NULL AND model_family IS NULL))
        AND (expires_at IS NULL OR expires_at > datetime('now'))
        ORDER BY 
            CASE data_quality_tag
                WHEN 'golden' THEN 1
                WHEN 'verified' THEN 2
                WHEN 'estimated' THEN 3
                ELSE 4
            END,
            calculated_at DESC
        LIMIT 1
        """
        
        result = self.db.execute(query, (feature_name, model_family, model_family)).fetchone()
        
        if result:
            return {
                "mean": result["mean"],
                "median": result["median"],
                "std_dev": result["std_dev"],
                "sample_count": result["sample_count"],
                "quality": result["data_quality_tag"],
            }
        
        if fallback_to_global and model_family is not None:
            return self.get_feature_mean(feature_name, None, False)
        
        return None
    
    def update_statistics(
        self,
        feature_name: str,
        new_values: list[float],
        model_family: str | None = None,
        model_name: str | None = None,
        source_run_ids: list[str] | None = None,
    ):
        """
        基于新检测数据更新统计值
        """
        if len(new_values) < 3:
            return  # 样本不足，不更新
        
        arr = np.array(new_values)
        
        stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std_dev": float(np.std(arr, ddof=1)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "sample_count": len(new_values),
            "calculated_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
        }
        
        # 确定数据质量标签
        if len(new_values) >= 100 and source_run_ids:
            quality = "golden"
        elif len(new_values) >= 20:
            quality = "verified"
        else:
            quality = "estimated"
        
        # 插入或更新
        upsert_sql = """
        INSERT INTO feature_statistics (
            feature_name, model_family, model_name,
            mean, median, std_dev, p5, p95,
            sample_count, sample_sources, calculated_at, expires_at,
            data_quality_tag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(feature_name, model_family, model_name) DO UPDATE SET
            mean = excluded.mean,
            median = excluded.median,
            std_dev = excluded.std_dev,
            p5 = excluded.p5,
            p95 = excluded.p95,
            sample_count = excluded.sample_count,
            sample_sources = excluded.sample_sources,
            calculated_at = excluded.calculated_at,
            expires_at = excluded.expires_at,
            data_quality_tag = excluded.data_quality_tag
        """
        
        self.db.execute(upsert_sql, (
            feature_name, model_family, model_name,
            stats["mean"], stats["median"], stats["std_dev"],
            stats["p5"], stats["p95"],
            stats["sample_count"],
            json.dumps(source_run_ids or []),
            stats["calculated_at"], stats["expires_at"],
            quality,
        ))
```

---

## 8. P7 — 可解释性与审计系统

### 8.1 自动归因分析（Auto-attribution）

```python
# backend/app/analysis/attribution.py

class ScoreAttributionAnalyzer:
    """
    自动分析分数构成，归因到具体用例
    """
    
    def attribute_score(
        self,
        scorecard: ScoreCard,
        case_results: list[CaseResult],
    ) -> AttributionReport:
        """
        分析总分是如何由各个用例贡献的
        """
        attributions = []
        
        for dimension in ["reasoning", "coding", "instruction", "safety", "knowledge"]:
            dim_score = getattr(scorecard, f"{dimension}_score", 0)
            dim_cases = [
                r for r in case_results
                if (r.case.dimension or r.case.category) == dimension
            ]
            
            # 计算每个用例的贡献
            case_contributions = []
            for case_result in dim_cases:
                contribution = self._calculate_case_contribution(
                    case_result, dim_score, len(dim_cases)
                )
                case_contributions.append({
                    "case_id": case_result.case.id,
                    "case_name": case_result.case.name,
                    "pass_rate": case_result.pass_rate,
                    "contribution": contribution,
                    "impact": "high" if abs(contribution) > 5 else "medium" if abs(contribution) > 2 else "low",
                })
            
            # 按贡献排序
            case_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            
            attributions.append({
                "dimension": dimension,
                "dimension_score": dim_score,
                "top_positive_contributors": [c for c in case_contributions if c["contribution"] > 0][:3],
                "top_negative_contributors": [c for c in case_contributions if c["contribution"] < 0][:3],
            })
        
        return AttributionReport(
            total_score=scorecard.total_score,
            dimension_attributions=attributions,
            summary=self._generate_attribution_summary(attributions),
        )
    
    def _calculate_case_contribution(
        self,
        case_result: CaseResult,
        dimension_score: float,
        n_cases: int,
    ) -> float:
        """
        计算单个用例对维度分数的贡献
        
        简化模型：假设维度分数是各用例通过率的加权平均
        """
        # 基础贡献
        base_contribution = (case_result.pass_rate * 100 - 50) * (case_result.case.weight / sum(r.case.weight for r in [case_result]))
        
        # 考虑用例难度（困难用例的贡献权重更高）
        difficulty_multiplier = 1 + (case_result.case.difficulty or 0.5) * 0.5
        
        return base_contribution * difficulty_multiplier
```

---

## 9. P8 — 性能与架构优化

### 9.1 异步流水线重构

```python
# backend/app/runner/async_pipeline.py

class AsyncDetectionPipeline:
    """
    完全异步的检测流水线，支持背压和优先级
    """
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.task_queue = asyncio.PriorityQueue()
        self.results_buffer = asyncio.Queue()
        self.metrics = PipelineMetrics()
    
    async def run_case_with_backpressure(
        self,
        case: TestCase,
        adapter: LLMAdapter,
        priority: int = 5,  # 1=最高，10=最低
    ) -> CaseResult:
        """
        带背压控制的用例执行
        """
        # 等待信号量（背压控制）
        async with self.semaphore:
            start_time = time.monotonic()
            
            try:
                # 执行用例
                result = await self._execute_case_async(case, adapter)
                
                # 记录指标
                latency = time.monotonic() - start_time
                self.metrics.record(case.category, latency, result.pass_rate >= 0.5)
                
                return result
                
            except asyncio.TimeoutError:
                self.metrics.record_timeout(case.category)
                return CaseResult(
                    case=case,
                    samples=[SampleResult(
                        sample_index=0,
                        response=LLMResponse(
                            error_type="timeout",
                            error_message=f"Case {case.id} timed out",
                        ),
                        judge_passed=False,
                    )],
                )
    
    async def _execute_case_async(
        self,
        case: TestCase,
        adapter: LLMAdapter,
    ) -> CaseResult:
        """异步执行单个用例的所有采样"""
        samples = []
        
        # 并发执行所有采样（受semaphore控制）
        sample_tasks = [
            self._execute_sample_async(case, adapter, i)
            for i in range(case.n_samples)
        ]
        
        sample_results = await asyncio.gather(*sample_tasks, return_exceptions=True)
        
        for i, result in enumerate(sample_results):
            if isinstance(result, Exception):
                samples.append(SampleResult(
                    sample_index=i,
                    response=LLMResponse(
                        error_type="execution_error",
                        error_message=str(result),
                    ),
                    judge_passed=False,
                ))
            else:
                samples.append(result)
        
        return CaseResult(case=case, samples=samples)
```

---

## 10. P9 — 新增测试维度

### 10.1 多模态能力测试（预留架构）

```python
# backend/app/core/schemas.py — 扩展

@dataclass
class MultimodalTestCase:
    """多模态测试用例（v5预留）"""
    id: str
    category: str = "multimodal"
    
    # 输入可以是文本、图片URL、音频URL等
    inputs: list[ModalityInput]
    
    # 期望输出类型
    expected_output_type: str  # "text", "image", "audio", "mixed"
    
    # 多模态特定的评判方法
    judge_method: str  # "vision_qa", "image_generation", "audio_transcription", etc.
    
    # 参考资源（用于评判）
    reference_resources: list[ReferenceResource]


@dataclass
class ModalityInput:
    modality: str  # "text", "image", "audio", "video"
    content: str | bytes  # 文本内容或base64编码的二进制
    mime_type: str | None = None
    

@dataclass
class ReferenceResource:
    """用于多模态判题的参考资源"""
    resource_type: str  # "image", "text", "embedding"
    content: str | bytes
    similarity_threshold: float = 0.8  # 对于图像相似度评判
```

---

### 10.2 长上下文能力测试

```python
# backend/app/fixtures/long_context_cases.py

LONG_CONTEXT_TEST_CASES = [
    {
        "id": "long_ctx_needle_haystack",
        "category": "long_context",
        "dimension": "long_context",
        "name": "needle_in_haystack",
        "description": "在长篇文档中定位特定信息（大海捞针测试）",
        
        # 动态生成长文档，在特定位置插入"针"
        "document_generator": {
            "type": "synthetic",
            "base_content": "random_wikipedia_paragraphs",  # 从维基百科获取的真实段落
            "needle": "The secret code is 42-PLATYPUS-777.",
            "needle_positions": ["beginning", "middle", "end", "distributed"],
            "lengths": [4_000, 8_000, 16_000, 32_000, 64_000, 128_000],  # tokens
        },
        
        "user_prompt": "What is the secret code mentioned in the document? Reply with just the code.",
        "judge_method": "exact_match",
        "params": {
            "target": "42-PLATYPUS-777",
        },
        "weight": 3.0,
    },
    {
        "id": "long_ctx_summarization",
        "category": "long_context",
        "dimension": "long_context",
        "name": "long_document_summarization",
        "description": "长文档摘要能力测试",
        
        "document_generator": {
            "type": "synthetic",
            "base_content": "technical_paper_sections",
            "n_sections": 20,
            "length_per_section": 1000,  # tokens
        },
        
        "user_prompt": "Summarize the main contributions and methodology of the paper in 3-5 sentences.",
        "judge_method": "semantic_judge",
        "params": {
            "required_key_points": [
                "methodology",
                "contributions", 
                "experimental_results",
            ],
        },
        "weight": 2.5,
    },
]
```

---

## 实施路线图

### Phase 1 (v5.0-alpha, 4周)
- P1.1 语义判题引擎 v2（引入本地嵌入模型）
- P1.2 幻觉检测增强
- P2.1 IRT引擎基础实现
- P6.1 特征值数据库（替代硬编码）

### Phase 2 (v5.0-beta, 4周)
- P2.2 自适应选题实现
- P3.1 神经网络相似度（基础版）
- P4.1 贝叶斯信号融合
- P5.1 交互式报告前端

### Phase 3 (v5.0-rc, 4周)
- P7.1 自动归因分析
- P8.1 异步流水线优化
- P9.1 多模态预留架构
- 全链路集成测试

### Phase 4 (v5.0-stable, 2周)
- 性能调优
- 文档完善
- 生产部署

---

## 附录：关键技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 本地嵌入模型 | BAAI/bge-large-zh-v1.5 + GTE-large-en | 中文SOTA，MIT协议 |
| 对比学习框架 | PyTorch + SimCLR | 成熟生态 |
| IRT计算 | scipy.optimize + 自研EM | 轻量，无需R/pyinfra |
| 贝叶斯网络 | pgmpy 或自研简化版 | 考虑依赖复杂度 |
| 前端可视化 | D3.js + ECharts | 成熟，文档完善 |
| 异步框架 | asyncio + aiohttp | Python原生，无需额外依赖 |

---

## 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 嵌入模型资源占用高 | 部署成本增加 | 提供云端fallback，本地可选安装 |
| IRT数据不足 | 校准不准确 | 前3个月使用混合权重（经验+数据） |
| 用户对新UI不适应 | 用户流失 | 保留v4界面切换选项 |
| 计算量增加 | 检测时间延长 | 智能采样，难例多采，易例少采 |

---

*文档版本: v5.0-draft*
*最后更新: 2026-04-10*
*作者: AI Architect*
