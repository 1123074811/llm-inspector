"""
验证并生成 tokenizer 探针的正确 token 数。
v6 fix: 2.7 Tokenizer探针验证脚本

依赖: pip install tiktoken sentencepiece
运行: python -m backend.scripts.verify_tokenizer_probes

输出到 fixtures/tokenizer_probes.json 供 pipeline.py 加载
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# 探针词列表（与 pipeline.py 保持一致）
PROBE_WORDS = [
    "supercalifragilistic",
    "cryptocurrency",
    "counterintuitive",
    "hallucination",
    "multimodality"
]

# Tokenizer配置
TOKENIZERS = {
    "cl100k_base": "OpenAI (GPT-4, GPT-3.5)",
    "o200k_base": "OpenAI (GPT-4o, GPT-4o-mini)",
    "p50k_base": "OpenAI (Legacy GPT-3)",
    # SentencePiece tokenizers would need manual verification
}


def verify_tiktoken(encoding_name: str) -> dict[str, int]:
    """使用 tiktoken 验证指定编码的 token 数。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return {word: len(enc.encode(word)) for word in PROBE_WORDS}
    except ImportError:
        print(f"Warning: tiktoken not installed, skipping {encoding_name}")
        return {}
    except Exception as e:
        print(f"Error verifying {encoding_name}: {e}")
        return {}


def generate_tokenizer_probes() -> dict:
    """生成完整的 tokenizer 探针数据。"""
    import datetime
    
    result = {
        "_meta": {
            "generated_at": datetime.datetime.now().isoformat(),
            "generator": "verify_tokenizer_probes.py",
            "note": "Token counts for model fingerprinting probes. Update when tokenizer versions change."
        },
        "probe_words": PROBE_WORDS,
        "tokenizers": {}
    }
    
    # 验证每个 tiktoken 编码
    for encoding_name, description in TOKENIZERS.items():
        counts = verify_tiktoken(encoding_name)
        if counts:
            result["tokenizers"][encoding_name] = {
                "description": description,
                "token_counts": counts,
                "source": "tiktoken"
            }
    
    # 手动维护的 SentencePiece tokenizers (无法自动验证)
    result["tokenizers"]["llama"] = {
        "description": "Meta LLaMA models",
        "token_counts": {
            "supercalifragilistic": 6,
            "cryptocurrency": 4,
            "counterintuitive": 4,
            "hallucination": 5,
            "multimodality": 5
        },
        "source": "manual",
        "note": "SentencePiece tokenizer - verify with actual model"
    }
    
    result["tokenizers"]["gemini"] = {
        "description": "Google Gemini models",
        "token_counts": {
            "supercalifragilistic": 7,
            "cryptocurrency": 4,
            "counterintuitive": 5,
            "hallucination": 6,
            "multimodality": 6
        },
        "source": "manual",
        "note": "Gemini tokenizer - verify with actual model"
    }
    
    return result


def main():
    """主函数：生成并保存 tokenizer 探针数据。"""
    # 生成数据
    data = generate_tokenizer_probes()
    
    # 确定输出路径 (指向 app/fixtures 目录)
    fixtures_dir = Path(__file__).parent.parent / "app" / "fixtures"
    output_file = fixtures_dir / "tokenizer_probes.json"
    
    # 保存文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Tokenizer probes saved to: {output_file}")
    print(f"\nGenerated at: {data['_meta']['generated_at']}")
    print("\nToken counts by tokenizer:")
    
    for name, info in data["tokenizers"].items():
        print(f"\n  {name} ({info.get('description', 'N/A')}) [{info.get('source', 'unknown')}]")
        counts = info.get("token_counts", {})
        for word, count in counts.items():
            print(f"    {word}: {count} tokens")


if __name__ == "__main__":
    main()
