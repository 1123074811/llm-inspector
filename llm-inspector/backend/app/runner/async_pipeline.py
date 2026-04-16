"""
Async Detection Pipeline — 完全异步的检测流水线

支持背压和优先级

v5.0 升级组件 - P8 性能与架构优化
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from app.core.logging import get_logger
from app.core.schemas import CaseResult, SampleResult, LLMResponse

logger = get_logger(__name__)


@dataclass
class PipelineMetrics:
    """流水线指标"""
    total_cases: int = 0
    completed_cases: int = 0
    failed_cases: int = 0
    timeout_cases: int = 0
    
    category_stats: dict[str, dict] = field(default_factory=dict)
    
    def record(self, category: str, latency: float, success: bool):
        """记录用例执行指标"""
        if category not in self.category_stats:
            self.category_stats[category] = {
                "count": 0,
                "total_latency": 0.0,
                "success_count": 0,
            }
        
        stats = self.category_stats[category]
        stats["count"] += 1
        stats["total_latency"] += latency
        if success:
            stats["success_count"] += 1
    
    def record_timeout(self, category: str):
        """记录超时"""
        self.timeout_cases += 1
        self.record(category, 30.0, False)  # 假设30秒超时


@dataclass
class PipelineTask:
    """流水线任务"""
    priority: int
    case: object  # TestCase
    adapter: object  # LLMAdapter
    run_id: str
    
    def __lt__(self, other):
        # 优先级数字越小越优先
        return self.priority < other.priority


class AsyncDetectionPipeline:
    """
    完全异步的检测流水线，支持背压和优先级
    """
    
    def __init__(self, max_concurrent: int = 10, default_timeout: float = 30.0):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.results_buffer: asyncio.Queue = asyncio.Queue()
        self.metrics = PipelineMetrics()
        self.default_timeout = default_timeout
        
        self._running = False
        self._worker_tasks: list[asyncio.Task] = []
    
    async def start(self, num_workers: int = 3):
        """启动流水线"""
        self._running = True
        
        # 启动工作线程
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        logger.info(f"Async pipeline started with {num_workers} workers")
    
    async def stop(self):
        """停止流水线"""
        self._running = False
        
        # 取消所有工作线程
        for task in self._worker_tasks:
            task.cancel()
        
        # 等待取消完成
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        
        logger.info("Async pipeline stopped")
    
    async def submit_case(
        self,
        case: object,
        adapter: object,
        run_id: str,
        priority: int = 5,
    ) -> asyncio.Future:
        """
        提交用例到队列
        
        Args:
            case: 测试用例
            adapter: LLM适配器
            run_id: 运行ID
            priority: 优先级（1=最高，10=最低）
        
        Returns:
            Future: 用例结果Future
        """
        future = asyncio.get_event_loop().create_future()
        task = PipelineTask(priority=priority, case=case, adapter=adapter, run_id=run_id)
        
        # 将future和task一起放入队列
        await self.task_queue.put((task, future))
        self.metrics.total_cases += 1
        
        return future
    
    async def _worker_loop(self, worker_id: str):
        """工作线程主循环"""
        logger.info(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # 从队列获取任务（带超时）
                task_item = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0,
                )
                task, future = task_item
                
                # 执行用例
                try:
                    result = await self._execute_case_with_backpressure(
                        task.case, task.adapter
                    )
                    
                    # 设置结果
                    if not future.done():
                        future.set_result(result)
                    
                    self.metrics.completed_cases += 1
                    
                except Exception as e:
                    logger.error(f"Case execution failed: {e}")
                    if not future.done():
                        future.set_exception(e)
                    self.metrics.failed_cases += 1
                
                finally:
                    self.task_queue.task_done()
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _execute_case_with_backpressure(
        self,
        case: object,
        adapter: object,
    ) -> CaseResult:
        """
        带背压控制的用例执行
        
        Args:
            case: 测试用例
            adapter: LLM适配器
        
        Returns:
            CaseResult: 用例结果
        """
        # 等待信号量（背压控制）
        async with self.semaphore:
            start_time = time.monotonic()
            category = getattr(case, "category", "unknown")
            
            try:
                # 执行用例
                result = await self._execute_case_async(case, adapter)
                
                # 记录指标
                latency = time.monotonic() - start_time
                success = result.samples and any(
                    getattr(s, "judge_passed", False) for s in result.samples
                )
                self.metrics.record(category, latency, success)
                
                return result
                
            except asyncio.TimeoutError:
                self.metrics.record_timeout(category)
                return CaseResult(
                    case=case,
                    samples=[SampleResult(
                        sample_index=0,
                        response=LLMResponse(
                            error_type="timeout",
                            error_message=f"Case {getattr(case, 'id', 'unknown')} timed out",
                        ),
                        judge_passed=False,
                    )],
                )
            except Exception as e:
                logger.error(f"Case execution error: {e}")
                self.metrics.failed_cases += 1
                return CaseResult(
                    case=case,
                    samples=[SampleResult(
                        sample_index=0,
                        response=LLMResponse(
                            error_type="execution_error",
                            error_message=str(e),
                        ),
                        judge_passed=False,
                    )],
                )
    
    async def _execute_case_async(
        self,
        case: object,
        adapter: object,
    ) -> CaseResult:
        """
        异步执行单个用例的所有采样
        
        Args:
            case: 测试用例
            adapter: LLM适配器
        
        Returns:
            CaseResult: 用例结果
        """
        n_samples = getattr(case, "n_samples", 1)
        
        # 并发执行所有采样
        sample_tasks = [
            self._execute_sample_async(case, adapter, i)
            for i in range(n_samples)
        ]
        
        sample_results = await asyncio.gather(*sample_tasks, return_exceptions=True)
        
        samples = []
        for i, result in enumerate(sample_results):
            if isinstance(result, Exception):
                logger.warning(f"Sample {i} failed: {result}")
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
    
    async def _execute_sample_async(
        self,
        case: object,
        adapter: object,
        sample_index: int,
    ) -> SampleResult:
        """
        异步执行单个采样
        
        Args:
            case: 测试用例
            adapter: LLM适配器
            sample_index: 采样索引
        
        Returns:
            SampleResult: 采样结果
        """
        try:
            # 构建请求
            user_prompt = getattr(case, "user_prompt", "")
            system_prompt = getattr(case, "system_prompt", None)
            temperature = getattr(case, "temperature", 0.0)
            max_tokens = getattr(case, "max_tokens", None)
            
            # 调用适配器
            from app.core.schemas import Message, LLMRequest
            
            _PLAIN_TEXT_INSTRUCTION = (
                "请用纯文本回复，不要使用 Markdown 格式（不要使用 **加粗**、*斜体*、# 标题、"
                "- 列表符号、> 引用块等格式符号），直接给出答案内容。"
            )
            _SKIP_PLAIN_TEXT_CATEGORIES = frozenset({"coding", "tool_use"})
            category = getattr(case, "category", "")

            messages = [Message(role="user", content=user_prompt)]
            if system_prompt:
                sys_content = system_prompt
                if category not in _SKIP_PLAIN_TEXT_CATEGORIES:
                    sys_content = sys_content + "\n" + _PLAIN_TEXT_INSTRUCTION
                messages.insert(0, Message(role="system", content=sys_content))
            elif category not in _SKIP_PLAIN_TEXT_CATEGORIES:
                messages.insert(0, Message(role="system", content=_PLAIN_TEXT_INSTRUCTION))
            
            request = LLMRequest(
                model=getattr(adapter, "model", "unknown"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # 执行请求（带超时）
            response = await asyncio.wait_for(
                self._call_adapter(adapter, request),
                timeout=self.default_timeout,
            )
            
            # 评判结果
            judge_method = getattr(case, "judge_method", "exact_match")
            judge_params = getattr(case, "params", {}) or {}
            
            # 导入评判函数
            from app.judge.methods import judge
            
            passed, detail = judge(
                judge_method,
                response.content,
                {**judge_params, "_original_prompt": user_prompt},
            )
            
            return SampleResult(
                sample_index=sample_index,
                response=response,
                judge_passed=passed,
                judge_detail=detail,
            )
            
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error(f"Sample {sample_index} execution error: {e}")
            return SampleResult(
                sample_index=sample_index,
                response=LLMResponse(
                    error_type="execution_error",
                    error_message=str(e),
                ),
                judge_passed=False,
            )
    
    async def _call_adapter(self, adapter: object, request: object) -> LLMResponse:
        """调用适配器（转换为异步）"""
        # 使用线程池执行同步适配器调用
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # 使用默认executor
            self._sync_call_adapter,
            adapter,
            request,
        )
    
    def _sync_call_adapter(self, adapter: object, request: object) -> LLMResponse:
        """同步调用适配器"""
        # 检查适配器类型
        if hasattr(adapter, "complete"):
            return adapter.complete(request)
        elif hasattr(adapter, "chat"):
            return adapter.chat(request)
        else:
            raise ValueError(f"Adapter {adapter} has no callable method")
    
    def get_metrics(self) -> dict:
        """获取流水线指标"""
        return {
            "total_cases": self.metrics.total_cases,
            "completed_cases": self.metrics.completed_cases,
            "failed_cases": self.metrics.failed_cases,
            "timeout_cases": self.metrics.timeout_cases,
            "success_rate": (
                self.metrics.completed_cases / max(self.metrics.total_cases, 1)
            ),
            "category_stats": self.metrics.category_stats,
        }


# 全局实例
_pipeline: Optional[AsyncDetectionPipeline] = None


def get_async_pipeline(
    max_concurrent: int = 10,
    default_timeout: float = 30.0,
) -> AsyncDetectionPipeline:
    """获取全局异步流水线实例"""
    global _pipeline
    if _pipeline is None:
        _pipeline = AsyncDetectionPipeline(max_concurrent, default_timeout)
    return _pipeline


async def run_detection_async(
    cases: list,
    adapter: object,
    run_id: str,
    max_concurrent: int = 10,
) -> list[CaseResult]:
    """
    便捷函数：异步运行检测
    
    Args:
        cases: 测试用例列表
        adapter: LLM适配器
        run_id: 运行ID
        max_concurrent: 最大并发数
    
    Returns:
        list[CaseResult]: 用例结果列表
    """
    pipeline = get_async_pipeline(max_concurrent)
    await pipeline.start(num_workers=min(3, max_concurrent // 3 + 1))
    
    try:
        # 提交所有用例
        futures = []
        for case in cases:
            # 根据用例类别设置优先级
            priority = _get_case_priority(case)
            future = await pipeline.submit_case(case, adapter, run_id, priority)
            futures.append(future)
        
        # 等待所有用例完成
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # 处理异常结果
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Case {i} failed: {result}")
                final_results.append(CaseResult(
                    case=cases[i],
                    samples=[SampleResult(
                        sample_index=0,
                        response=LLMResponse(
                            error_type="pipeline_error",
                            error_message=str(result),
                        ),
                        judge_passed=False,
                    )],
                ))
            else:
                final_results.append(result)
        
        return final_results
        
    finally:
        await pipeline.stop()


def _get_case_priority(case: object) -> int:
    """根据用例类别获取优先级"""
    category = getattr(case, "category", "unknown")
    
    # 优先级映射（数字越小优先级越高）
    priority_map = {
        "protocol": 1,  # 协议检测最高优先级
        "safety": 2,    # 安全检测高优先级
        "identity": 2,
        "reasoning": 3,
        "coding": 3,
        "instruction": 4,
        "knowledge": 5,
        "style": 6,     # 风格检测低优先级
    }
    
    return priority_map.get(category, 5)
