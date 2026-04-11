"""
v8.0 Phase 4 Architecture Optimization Demo

展示新功能：
1. 插件化判题架构
2. 结构化日志系统
3. 透明判题过程
4. 阈值来源追踪

运行: python examples/v8_phase4_demo.py
"""
import sys
sys.path.insert(0, 'llm-inspector/backend')

from app.judge import (
    PluginManager, get_plugin_manager, register_builtin_plugins,
    ExactMatchPlugin, RegexMatchPlugin, ConstraintReasoningPlugin,
    create_transparent_judge, judge_with_transparency,
    JudgeTier
)
from app.core import get_structured_logger, LogEventType


def demo_plugin_system():
    """演示插件系统。"""
    print("=" * 60)
    print("演示 1: 插件化判题架构")
    print("=" * 60)
    
    manager = get_plugin_manager()
    register_builtin_plugins(manager)
    
    # 列出所有可用插件
    print("\n可用判题插件:")
    for name in manager.list_plugins():
        meta = manager.get_metadata(name)
        tier_emoji = {"local": "⚡", "embedding": "🔍", "llm": "🤖"}
        tier_icon = tier_emoji.get(meta.tier.value, "❓")
        print(f"  {tier_icon} {name:20} v{meta.version:6} [{meta.tier.value}]")
        print(f"     └─ {meta.description}")
    
    # 执行判题
    print("\n执行判题测试:")
    
    # 1. 精确匹配
    result = manager.judge("exact_match", "hello", {"target": "hello"})
    print(f"  exact_match('hello', 'hello') => {result.passed} ✓")
    
    # 2. 正则匹配
    result = manager.judge("regex_match", "test123", {"pattern": r"\d+"})
    print(f"  regex_match('test123', r'\\d+') => {result.passed} ✓")
    
    # 3. 约束推理
    result = manager.judge("constraint_reasoning",
        "The answer is 42, using the boundary method",
        {
            "target_pattern": r"42",
            "key_constraints": ["boundary", "answer"],
            "coverage_threshold": 0.5,
            "threshold_source": "irt_calibration_v2026q1"
        }
    )
    print(f"  constraint_reasoning() => passed={result.passed}, grade={result.detail.get('quality_grade')} ✓")
    
    print(f"\n✅ 插件系统演示完成")
    print()


def demo_structured_logging():
    """演示结构化日志系统。"""
    print("=" * 60)
    print("演示 2: 结构化日志系统")
    print("=" * 60)
    
    logger = get_structured_logger()
    logger.clear()
    
    # 模拟判题过程并记录日志
    print("\n模拟判题并记录日志...")
    
    logger.log_judge_start("case_001", "constraint_reasoning", trace_id="demo_trace_1")
    
    logger.log_judge_step(
        case_id="case_001",
        step="keyword_coverage",
        input_data={"constraints": ["boundary", "answer", "calculated"]},
        output_data={"coverage": 0.67, "matched": ["boundary", "answer"]},
        threshold=0.50,
        threshold_source="irt_calibration_v2026q1"
    )
    
    logger.log_threshold_apply(
        component="judge",
        threshold_name="coverage_threshold",
        value=0.50,
        source="irt_calibration_v2026q1",
        context={"case_id": "case_001", "mode": "standard"}
    )
    
    logger.log_judge_complete(
        case_id="case_001",
        method="constraint_reasoning",
        passed=True,
        confidence=0.85,
        detail={"quality_grade": "A", "keyword_coverage": 0.67},
        tokens_used=0,
        latency_ms=45
    )
    
    # 查询日志
    print("\n最近判题日志:")
    recent = logger.get_recent(10)
    for entry in recent:
        emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(entry.level, "•")
        print(f"  {emoji} [{entry.timestamp[:19]}] {entry.event_type}")
        if entry.data.get("threshold_source"):
            print(f"     └─ 阈值来源: {entry.data['threshold_source']}")
    
    # 统计
    print(f"\n日志统计: {len(logger.get_recent(100))} 条记录")
    
    print(f"\n✅ 日志系统演示完成")
    print()


def demo_transparent_judging():
    """演示透明判题。"""
    print("=" * 60)
    print("演示 3: 透明判题过程")
    print("=" * 60)
    
    # 清空之前的日志
    get_structured_logger().clear()
    
    manager = get_plugin_manager()
    register_builtin_plugins(manager)
    
    print("\n使用透明包装器执行判题...")
    
    # 使用透明判题
    result, transparency = judge_with_transparency(
        "constraint_reasoning",
        "The answer is 100, calculated using the optimization boundary method",
        {
            "target_pattern": r"100",
            "key_constraints": ["optimization", "boundary", "calculated"],
            "boundary_signals": ["boundary"],
            "coverage_threshold": 0.6,
            "threshold_source": "irt_calibration_v2026q1"
        },
        case_id="demo_transparent_001"
    )
    
    print(f"\n判题结果:")
    print(f"  通过: {result.passed}")
    print(f"  置信度: {result.confidence:.2%}")
    print(f"  质量等级: {result.detail.get('quality_grade', 'N/A')}")
    print(f"  关键词覆盖: {result.detail.get('keyword_coverage', 'N/A')}")
    
    print(f"\n阈值追踪:")
    print(f"  应用阈值: {result.threshold_value}")
    print(f"  阈值来源: {result.threshold_source}")
    
    print(f"\n✅ 透明判题演示完成")
    print()


def demo_plugin_statistics():
    """演示插件统计功能。"""
    print("=" * 60)
    print("演示 4: 插件运行时统计")
    print("=" * 60)
    
    manager = get_plugin_manager()
    
    print("\n执行多次判题以累积统计...")
    
    # 执行多次判题
    for i in range(5):
        manager.judge("exact_match", f"test{i}", {"target": "test0", "case_id": f"stat_case_{i}"})
    
    # 获取统计
    print("\n插件运行统计:")
    all_stats = manager.get_all_stats()
    for method, stats in all_stats.items():
        if stats.call_count > 0:
            print(f"  {method:20} 调用:{stats.call_count:3}次  平均延迟:{stats.avg_latency_ms:.2f}ms")
    
    print(f"\n✅ 统计功能演示完成")
    print()


def main():
    """主入口。"""
    print("\n" + "=" * 60)
    print("  LLM Inspector v8.0 Phase 4 架构优化演示")
    print("  插件化架构 + 结构化日志 + 透明判题")
    print("=" * 60)
    print()
    
    try:
        demo_plugin_system()
        demo_structured_logging()
        demo_transparent_judging()
        demo_plugin_statistics()
        
        print("=" * 60)
        print("  ✅ 所有演示完成!")
        print("=" * 60)
        print()
        print("Phase 4 实现内容:")
        print("  • 插件接口 (JudgePlugin)")
        print("  • 插件管理器 (PluginManager)")
        print("  • 6个内置插件适配器")
        print("  • 结构化日志系统 (StructuredLogger)")
        print("  • 透明判题包装器 (TransparentJudgeWrapper)")
        print("  • 阈值来源追踪")
        print("  • 完整的测试套件 (25个测试)")
        print()
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
