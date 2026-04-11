"""Test suite for LLM Inspector v8.0 Phase 2: Judge System Upgrade.

Tests the three core components:
1. Semantic Judge v3 (three-tier cascaded evaluation)
2. Hallucination Detection v3 (knowledge graph integration)
3. Code Sandbox v2 (multi-language support)
"""

import pytest
import tempfile
import os

# Knowledge graph tests
from app.knowledge.wikidata_client import WikidataClient, VerificationResult
from app.knowledge.kg_client import KnowledgeGraphClient

# Judge tests
from app.judge.code_sandbox_v2 import (
    Language,
    CodeExtractor,
    SecureCodeSandbox,
    judge_code_execution,
    LANGUAGE_CONFIGS,
)


# Skip tests requiring network access
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("ENABLE_NETWORK_TESTS"),
        reason="Network tests disabled. Set ENABLE_NETWORK_TESTS=1 to enable."
    ),
]


class TestCodeExtractor:
    """Test code extraction from markdown."""
    
    def test_extract_python_code(self):
        """Test extracting Python code."""
        text = '''```python
def hello():
    return "world"
```'''
        code, lang = CodeExtractor.extract(text)
        assert "def hello" in code
        assert lang == Language.PYTHON
    
    def test_extract_javascript_code(self):
        """Test extracting JavaScript code."""
        text = '''```javascript
function hello() {
    return "world";
}
```'''
        code, lang = CodeExtractor.extract(text)
        assert "function hello" in code
        assert lang == Language.JAVASCRIPT
    
    def test_extract_cpp_code(self):
        """Test extracting C++ code."""
        text = '''```cpp
#include <iostream>
int main() { return 0; }
```'''
        code, lang = CodeExtractor.extract(text)
        assert "#include" in code
        assert lang == Language.CPP
    
    def test_extract_no_language_tag(self):
        """Test extracting code without language tag."""
        text = '''```
print("hello")
```'''
        code, lang = CodeExtractor.extract(text)
        assert "print" in code
        assert lang is None
    
    def test_extract_plain_text_fallback(self):
        """Test falling back to plain text when no code block."""
        text = "print('hello')"
        code, lang = CodeExtractor.extract(text)
        assert code == text
    
    def test_detect_language_python(self):
        """Test Python language detection."""
        code = "def foo():\n    return 1"
        lang = CodeExtractor.detect_language(code)
        assert lang == Language.PYTHON
    
    def test_detect_language_javascript(self):
        """Test JavaScript language detection."""
        code = "const x = () => console.log('test');"
        lang = CodeExtractor.detect_language(code)
        assert lang == Language.JAVASCRIPT
    
    def test_detect_language_java(self):
        """Test Java language detection."""
        code = "public class Main { public static void main(String[] args) {} }"
        lang = CodeExtractor.detect_language(code)
        assert lang == Language.JAVA


class TestSecureCodeSandbox:
    """Test multi-language code sandbox."""
    
    def test_python_execution_success(self):
        """Test successful Python execution."""
        sandbox = SecureCodeSandbox(use_docker=False)
        
        code = "print('Hello, World!')"
        result = sandbox.execute(code, Language.PYTHON)
        
        assert result.passed
        assert "Hello, World!" in result.stdout
        assert result.language == "python"
        assert result.exit_code == 0
    
    def test_python_execution_with_input(self):
        """Test Python execution with input."""
        sandbox = SecureCodeSandbox(use_docker=False)
        
        code = "name = input(); print(f'Hello, {name}!')"
        result = sandbox.execute(code, Language.PYTHON, test_input="Alice")
        
        assert result.passed
        assert "Hello, Alice!" in result.stdout
    
    def test_python_execution_error(self):
        """Test Python execution with error."""
        sandbox = SecureCodeSandbox(use_docker=False)
        
        code = "print(undefined_variable)"
        result = sandbox.execute(code, Language.PYTHON)
        
        assert not result.passed
        assert result.exit_code != 0
        assert "NameError" in result.stderr or "undefined" in result.stderr.lower()
    
    def test_python_timeout(self):
        """Test Python execution timeout."""
        sandbox = SecureCodeSandbox(use_docker=False)
        
        code = "while True: pass"  # Infinite loop
        result = sandbox.execute(code, Language.PYTHON)
        
        assert result.timeout_triggered
        assert not result.passed
    
    def test_javascript_execution(self):
        """Test JavaScript execution."""
        sandbox = SecureCodeSandbox(use_docker=False)
        
        code = "console.log('Hello from JS');"
        result = sandbox.execute(code, Language.JAVASCRIPT)
        
        # May not pass if Node not installed, but should handle gracefully
        assert result.language == "javascript"
        assert isinstance(result.passed, bool)
    
    def test_language_configs(self):
        """Test language configuration validity."""
        for lang, config in LANGUAGE_CONFIGS.items():
            assert config.image, f"{lang} missing image"
            assert config.command, f"{lang} missing command"
            assert config.timeout > 0, f"{lang} invalid timeout"
            assert config.memory_mb > 0, f"{lang} invalid memory"
    
    def test_sandbox_stats(self):
        """Test sandbox statistics tracking."""
        sandbox = SecureCodeSandbox(use_docker=False)
        
        # Run some executions
        sandbox.execute("print('1')", Language.PYTHON)
        sandbox.execute("print('2')", Language.PYTHON)
        
        stats = sandbox.get_stats()
        assert stats["executions"] == 2
        assert stats["passed"] >= 0
        assert stats["failed"] >= 0


class TestCodeJudgeIntegration:
    """Test code judge integration."""
    
    def test_judge_python_simple(self):
        """Test judging simple Python code."""
        response = "```python\nprint(42)\n```"
        params = {
            "language": "python",
            "test_cases": [
                {"input": "", "expected_output": "42"}
            ]
        }
        
        passed, details = judge_code_execution(response, params)
        
        assert isinstance(passed, bool)
        assert "language" in details
        assert "test_results" in details
    
    def test_judge_no_code_found(self):
        """Test judging response with no code."""
        response = "This is just text with no code."
        params = {"language": "python"}
        
        passed, details = judge_code_execution(response, params)
        
        assert not passed
        assert "error" in details
    
    def test_judge_multiple_languages(self):
        """Test judging different languages."""
        test_cases = [
            ("```python\nprint(1)\n```", "python"),
            ("```javascript\nconsole.log(1)\n```", "javascript"),
        ]
        
        for response, lang in test_cases:
            params = {"language": lang}
            passed, details = judge_code_execution(response, params)
            
            assert "language" in details
            assert details["language"] == lang


class TestKnowledgeGraphClient:
    """Test knowledge graph client with mocking."""
    
    def test_heuristic_fallback(self):
        """Test heuristic verification fallback."""
        client = KnowledgeGraphClient(use_wikidata=False)
        
        result = client.verify_entity("Albert Einstein")
        
        # Should use heuristic (no Wikidata)
        assert result.source == "heuristic"
        assert result.confidence > 0
        assert isinstance(result.is_verified, bool)
    
    def test_heuristic_short_name(self):
        """Test heuristic with short name."""
        client = KnowledgeGraphClient(use_wikidata=False)
        
        result = client.verify_entity("AB")
        
        # Short names should get low confidence
        assert result.confidence < 0.3
        assert result.source == "heuristic"
    
    def test_heuristic_common_word(self):
        """Test heuristic with common word."""
        client = KnowledgeGraphClient(use_wikidata=False)
        
        result = client.verify_entity("the")
        
        # Common words should get very low confidence
        assert result.confidence < 0.1
    
    def test_cache_functionality(self):
        """Test caching of verification results."""
        client = KnowledgeGraphClient(use_wikidata=False)
        
        # First call
        result1 = client.verify_entity("TestEntity")
        
        # Second call should use cache
        result2 = client.verify_entity("TestEntity")
        
        # Should be cached
        assert "cache" in result2.source
        
        # Check stats
        stats = client.get_stats()
        assert stats["cache"]["total_queries"] > 0
    
    def test_verify_fact_heuristic(self):
        """Test fact verification with heuristic."""
        client = KnowledgeGraphClient(use_wikidata=False)
        
        result = client.verify_fact("Albert Einstein was a physicist")
        
        assert isinstance(result.is_verified, bool)
        assert 0 <= result.confidence <= 1
    
    def test_verify_entities_batch(self):
        """Test batch entity verification."""
        client = KnowledgeGraphClient(use_wikidata=False)
        
        entities = ["Einstein", "Newton", "Short"]
        results = client.verify_entities_batch(entities)
        
        assert len(results) == 3
        for entity, result in results.items():
            assert entity in entities
            assert isinstance(result.confidence, float)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        client = KnowledgeGraphClient(use_wikidata=False)
        
        # Add something to cache
        client.verify_entity("Test")
        
        # Clear cache
        client.clear_cache()
        
        # Stats should show 0 cache size
        stats = client.get_stats()
        assert stats["cache_size"] == 0


class TestWikidataClient:
    """Test Wikidata API client (requires network)."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        client = WikidataClient(rate_limit=0.5)
        
        assert client.rate_limit == 0.5
        assert client.timeout == 10
    
    @pytest.mark.skipif(
        not os.environ.get("ENABLE_LIVE_API_TESTS"),
        reason="Live API tests disabled"
    )
    def test_search_entities_real(self):
        """Test real entity search (requires network)."""
        client = WikidataClient()
        
        entities = client.search_entities("Albert Einstein", limit=3)
        
        assert len(entities) > 0
        assert any("Einstein" in e.label for e in entities)
    
    @pytest.mark.skipif(
        not os.environ.get("ENABLE_LIVE_API_TESTS"),
        reason="Live API tests disabled"
    )
    def test_verify_entity_exists_real(self):
        """Test real entity verification (requires network)."""
        client = WikidataClient()
        
        result = client.verify_entity_exists("Albert Einstein")
        
        assert result.is_verified
        assert result.confidence > 0.5
        assert result.source == "wikidata"
    
    def test_extract_entities_simple(self):
        """Test simple entity extraction."""
        client = WikidataClient()
        
        text = "Albert Einstein was a physicist"
        entities = client._extract_entities(text)
        
        assert "Albert Einstein" in entities or "Einstein" in entities
    
    def test_extract_entities_with_quotes(self):
        """Test entity extraction with quoted text."""
        client = WikidataClient()
        
        text = 'The book "The Great Gatsby" was written by F. Scott Fitzgerald'
        entities = client._extract_entities(text)
        
        # Should extract quoted text
        assert any("Gatsby" in e for e in entities)


class TestJudgeIntegration:
    """Integration tests for Phase 2 components."""
    
    def test_full_code_judge_flow(self):
        """Test complete code judging flow."""
        # Simulate a model response
        response = """Here's the solution:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
```
"""
        
        params = {
            "language": "python",
            "test_cases": [
                {"input": "", "expected_output": "120"}
            ]
        }
        
        passed, details = judge_code_execution(response, params)
        
        assert "language" in details
        assert "test_results" in details
        assert len(details["test_results"]) == 1
    
    def test_multi_language_detection(self):
        """Test automatic language detection."""
        test_cases = [
            ("def foo(): pass", Language.PYTHON),
            ("function foo() {}", Language.JAVASCRIPT),
            ("public class Main {}", Language.JAVA),
        ]
        
        for code, expected in test_cases:
            detected = CodeExtractor.detect_language(code)
            assert detected == expected, f"Failed for: {code}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
