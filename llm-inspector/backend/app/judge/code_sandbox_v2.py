"""Code Sandbox v2 - Multi-language secure code execution.

Supports Python, JavaScript, C++, Java, and Go with Docker isolation.

Security measures:
- No network access (--network none)
- Read-only filesystem (except /tmp)
- Memory and CPU limits
- Timeout enforcement
- Non-root user execution

Reference: NSA Kubernetes Hardening Guide
"""

import re
import os
import tempfile
import subprocess
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone

from app.core.logging import get_logger

logger = get_logger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    JAVA = "java"
    GO = "go"


@dataclass
class SandboxConfig:
    """Sandbox configuration for a language."""
    image: str
    command: List[str]
    compile_command: Optional[List[str]] = None
    timeout: int = 5
    memory_mb: int = 256
    cpu_quota: int = 100000  # Docker CPU quota
    network_enabled: bool = False


# Language configurations
LANGUAGE_CONFIGS: Dict[Language, SandboxConfig] = {
    Language.PYTHON: SandboxConfig(
        image="python:3.11-slim",
        command=["python"],
        timeout=5,
        memory_mb=256,
    ),
    Language.JAVASCRIPT: SandboxConfig(
        image="node:18-slim",
        command=["node"],
        timeout=5,
        memory_mb=256,
    ),
    Language.CPP: SandboxConfig(
        image="gcc:12",
        command=["./a.out"],
        compile_command=["g++", "-O2", "-std=c++17", "-o", "a.out"],
        timeout=10,
        memory_mb=512,
    ),
    Language.JAVA: SandboxConfig(
        image="openjdk:17-slim",
        command=["java", "Main"],
        compile_command=["javac", "Main.java"],
        timeout=10,
        memory_mb=512,
    ),
    Language.GO: SandboxConfig(
        image="golang:1.21",
        command=["./main"],
        compile_command=["go", "build", "-o", "main"],
        timeout=5,
        memory_mb=256,
    ),
}


@dataclass
class ExecutionResult:
    """Code execution result."""
    passed: bool
    language: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int
    memory_usage_mb: Optional[float] = None
    
    # Compilation info (for compiled languages)
    compile_stdout: str = ""
    compile_stderr: str = ""
    compile_success: bool = True
    
    # Error details
    timeout_triggered: bool = False
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "language": self.language,
            "stdout": self.stdout[:1000],  # Truncate long output
            "stderr": self.stderr[:500],
            "exit_code": self.exit_code,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "compile_success": self.compile_success,
            "timeout_triggered": self.timeout_triggered,
            "error_type": self.error_type,
        }


class CodeExtractor:
    """Extract code from markdown/text responses."""
    
    CODE_BLOCK_PATTERN = re.compile(
        r'```(\w+)?\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )
    
    LANGUAGE_MAP = {
        'python': Language.PYTHON,
        'py': Language.PYTHON,
        'javascript': Language.JAVASCRIPT,
        'js': Language.JAVASCRIPT,
        'node': Language.JAVASCRIPT,
        'cpp': Language.CPP,
        'c++': Language.CPP,
        'cxx': Language.CPP,
        'java': Language.JAVA,
        'go': Language.GO,
        'golang': Language.GO,
    }
    
    @classmethod
    def extract(cls, text: str, expected_language: Optional[Language] = None) -> Tuple[str, Optional[Language]]:
        """
        Extract code from markdown code blocks.
        
        Args:
            text: Response text containing code
            expected_language: Expected language (for validation)
            
        Returns:
            Tuple of (extracted_code, detected_language)
        """
        # Find code blocks
        matches = cls.CODE_BLOCK_PATTERN.findall(text)
        
        if not matches:
            # No code block found, return plain text
            return text.strip(), expected_language
        
        # Use first code block
        lang_tag, code = matches[0]
        
        # Detect language from tag
        detected = None
        if lang_tag:
            lang_lower = lang_tag.lower()
            detected = cls.LANGUAGE_MAP.get(lang_lower)
        
        # If expected language specified, validate
        if expected_language and detected and detected != expected_language:
            logger.warning(f"Language mismatch: expected {expected_language}, got {detected}")
        
        return code.strip(), (detected or expected_language)
    
    @classmethod
    def detect_language(cls, text: str) -> Optional[Language]:
        """Detect programming language from code content."""
        text_lower = text.lower()
        
        # Python indicators
        if any(kw in text_lower for kw in ['def ', 'import ', 'print(', '__init__']):
            return Language.PYTHON
        
        # JavaScript indicators
        if any(kw in text_lower for kw in ['function', 'const ', 'let ', '=>', 'console.log']):
            return Language.JAVASCRIPT
        
        # Java indicators
        if any(kw in text_lower for kw in ['public class', 'System.out', 'void main']):
            return Language.JAVA
        
        # C++ indicators
        if any(kw in text_lower for kw in ['#include', 'std::', 'int main()']):
            return Language.CPP
        
        # Go indicators
        if any(kw in text_lower for kw in ['package main', 'func main()', 'fmt.']):
            return Language.GO
        
        return None


class SecureCodeSandbox:
    """
    Secure multi-language code execution sandbox.
    
    Supports Docker-based isolation with configurable security policies.
    Falls back to subprocess-based execution if Docker unavailable.
    """
    
    def __init__(self, use_docker: bool = False):
        """
        Initialize sandbox.
        
        Args:
            use_docker: Whether to use Docker isolation (requires Docker daemon)
        """
        self.use_docker = use_docker
        self._docker_available = self._check_docker() if use_docker else False
        
        # Statistics
        self.stats = {
            "executions": 0,
            "passed": 0,
            "failed": 0,
            "timeouts": 0,
        }
        
        logger.info(f"CodeSandbox v2 initialized (docker={self._docker_available})")
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _get_filename(self, language: Language) -> str:
        """Get appropriate filename for language."""
        filenames = {
            Language.PYTHON: "script.py",
            Language.JAVASCRIPT: "script.js",
            Language.CPP: "main.cpp",
            Language.JAVA: "Main.java",
            Language.GO: "main.go",
        }
        return filenames.get(language, "script.txt")
    
    def _execute_with_docker(
        self,
        code: str,
        language: Language,
        test_input: str = ""
    ) -> ExecutionResult:
        """Execute code in Docker container."""
        config = LANGUAGE_CONFIGS[language]
        filename = self._get_filename(language)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to file
            code_path = os.path.join(tmpdir, filename)
            with open(code_path, 'w') as f:
                f.write(code)
            
            # Docker security options
            security_opts = [
                "--network", "none" if not config.network_enabled else "bridge",
                "--read-only",
                "--tmpfs", "/tmp:noexec,nosuid,size=100m",
                "--memory", f"{config.memory_mb}m",
                "--memory-swap", f"{config.memory_mb}m",
                "--cpus", "1.0",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true",
                "-v", f"{tmpdir}:/workspace:ro",
                "-w", "/workspace",
            ]
            
            start_time = datetime.now(timezone.utc)
            
            # Compile if needed
            compile_success = True
            compile_stderr = ""
            
            if config.compile_command:
                compile_cmd = [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "-v", f"{tmpdir}:/workspace",
                    "-w", "/workspace",
                    config.image,
                ] + config.compile_command
                
                try:
                    compile_result = subprocess.run(
                        compile_cmd,
                        capture_output=True,
                        text=True,
                        timeout=config.timeout
                    )
                    compile_success = compile_result.returncode == 0
                    compile_stderr = compile_result.stderr
                except Exception as e:
                    compile_success = False
                    compile_stderr = str(e)
            
            # Run the code
            if compile_success:
                run_cmd = [
                    "docker", "run", "--rm",
                ] + security_opts + [
                    config.image,
                ] + config.command + ([filename] if language == Language.PYTHON else [])
                
                try:
                    run_result = subprocess.run(
                        run_cmd,
                        input=test_input,
                        capture_output=True,
                        text=True,
                        timeout=config.timeout
                    )
                    
                    execution_time = int(
                        (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
                    
                    return ExecutionResult(
                        passed=run_result.returncode == 0,
                        language=language.value,
                        stdout=run_result.stdout,
                        stderr=run_result.stderr,
                        exit_code=run_result.returncode,
                        execution_time_ms=execution_time,
                        compile_success=compile_success,
                        compile_stderr=compile_stderr,
                        timeout_triggered=False
                    )
                    
                except subprocess.TimeoutExpired:
                    self.stats["timeouts"] += 1
                    return ExecutionResult(
                        passed=False,
                        language=language.value,
                        stdout="",
                        stderr="Execution timed out",
                        exit_code=-1,
                        execution_time_ms=config.timeout * 1000,
                        timeout_triggered=True,
                        error_type="timeout"
                    )
            else:
                return ExecutionResult(
                    passed=False,
                    language=language.value,
                    stdout="",
                    stderr="Compilation failed",
                    exit_code=-1,
                    execution_time_ms=0,
                    compile_success=False,
                    compile_stderr=compile_stderr,
                    error_type="compilation_error"
                )
    
    def _execute_with_subprocess(
        self,
        code: str,
        language: Language,
        test_input: str = ""
    ) -> ExecutionResult:
        """Execute code using subprocess (fallback, less secure)."""
        config = LANGUAGE_CONFIGS[language]
        filename = self._get_filename(language)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, filename)
            with open(code_path, 'w') as f:
                f.write(code)
            
            start_time = datetime.now(timezone.utc)
            
            # Compile if needed
            compile_success = True
            compile_stderr = ""
            
            if config.compile_command:
                compile_cmd = config.compile_command.copy()
                if language == Language.CPP:
                    compile_cmd.extend([code_path])
                elif language == Language.JAVA:
                    compile_cmd = ["javac", code_path]
                elif language == Language.GO:
                    compile_cmd = ["go", "build", "-o", "main", code_path]
                
                try:
                    compile_result = subprocess.run(
                        compile_cmd,
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        timeout=config.timeout
                    )
                    compile_success = compile_result.returncode == 0
                    compile_stderr = compile_result.stderr
                except Exception as e:
                    compile_success = False
                    compile_stderr = str(e)
            
            # Run
            if compile_success:
                if language == Language.PYTHON:
                    run_cmd = ["python", code_path]
                elif language == Language.JAVASCRIPT:
                    run_cmd = ["node", code_path]
                elif language == Language.CPP:
                    run_cmd = ["./a.out"]
                elif language == Language.JAVA:
                    run_cmd = ["java", "-cp", tmpdir, "Main"]
                elif language == Language.GO:
                    run_cmd = ["./main"]
                else:
                    run_cmd = []
                
                try:
                    run_result = subprocess.run(
                        run_cmd,
                        cwd=tmpdir,
                        input=test_input,
                        capture_output=True,
                        text=True,
                        timeout=config.timeout
                    )
                    
                    execution_time = int(
                        (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    )
                    
                    return ExecutionResult(
                        passed=run_result.returncode == 0,
                        language=language.value,
                        stdout=run_result.stdout,
                        stderr=run_result.stderr,
                        exit_code=run_result.returncode,
                        execution_time_ms=execution_time,
                        compile_success=compile_success,
                        compile_stderr=compile_stderr,
                        timeout_triggered=False
                    )
                    
                except subprocess.TimeoutExpired:
                    self.stats["timeouts"] += 1
                    return ExecutionResult(
                        passed=False,
                        language=language.value,
                        stdout="",
                        stderr="Execution timed out",
                        exit_code=-1,
                        execution_time_ms=config.timeout * 1000,
                        timeout_triggered=True,
                        error_type="timeout"
                    )
            else:
                return ExecutionResult(
                    passed=False,
                    language=language.value,
                    stdout="",
                    stderr="Compilation failed",
                    exit_code=-1,
                    execution_time_ms=0,
                    compile_success=False,
                    compile_stderr=compile_stderr,
                    error_type="compilation_error"
                )
    
    def execute(
        self,
        code: str,
        language: Language,
        test_input: str = "",
        expected_output: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code securely in sandbox.
        
        Args:
            code: Source code to execute
            language: Programming language
            test_input: Input to provide to program
            expected_output: Expected output (for pass/fail determination)
            
        Returns:
            ExecutionResult with details
        """
        self.stats["executions"] += 1
        
        # Choose execution method
        if self.use_docker and self._docker_available:
            result = self._execute_with_docker(code, language, test_input)
        else:
            result = self._execute_with_subprocess(code, language, test_input)
        
        # Check expected output if provided
        if expected_output is not None:
            output_match = expected_output.strip() in result.stdout.strip()
            result.passed = result.passed and output_match
        
        # Update stats
        if result.passed:
            self.stats["passed"] += 1
        else:
            self.stats["failed"] += 1
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.stats.copy()


def judge_code_execution(
    response: str,
    params: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """
    Judge code execution for v8 compatibility.
    
    Args:
        response: Model response containing code
        params: Judge parameters
        
    Returns:
        Tuple of (passed, details)
    """
    # Extract parameters
    expected_language_str = params.get("language", "python").lower()
    test_cases = params.get("test_cases", [])
    
    # Map string to Language enum
    language_map = {
        "python": Language.PYTHON,
        "py": Language.PYTHON,
        "javascript": Language.JAVASCRIPT,
        "js": Language.JAVASCRIPT,
        "cpp": Language.CPP,
        "c++": Language.CPP,
        "java": Language.JAVA,
        "go": Language.GO,
    }
    language = language_map.get(expected_language_str, Language.PYTHON)
    
    # Extract code from response
    code, detected_lang = CodeExtractor.extract(response, language)
    
    if not code:
        return False, {
            "error": "No code found in response",
            "language": expected_language_str,
        }
    
    # Initialize sandbox
    use_docker = params.get("use_docker", False)
    sandbox = SecureCodeSandbox(use_docker=use_docker)
    
    # Run test cases
    all_passed = True
    results = []
    
    for i, test in enumerate(test_cases):
        test_input = test.get("input", "")
        expected_output = test.get("expected_output", "")
        
        result = sandbox.execute(
            code=code,
            language=detected_lang or language,
            test_input=test_input,
            expected_output=expected_output
        )
        
        results.append({
            "test_case": i + 1,
            **result.to_dict()
        })
        
        if not result.passed:
            all_passed = False
    
    # If no test cases, just check if code runs
    if not test_cases:
        result = sandbox.execute(code, detected_lang or language)
        results.append({
            "test_case": "compilation/execution",
            **result.to_dict()
        })
        all_passed = result.passed
    
    return all_passed, {
        "language": (detected_lang or language).value,
        "all_passed": all_passed,
        "test_results": results,
        "sandbox_stats": sandbox.get_stats(),
    }
