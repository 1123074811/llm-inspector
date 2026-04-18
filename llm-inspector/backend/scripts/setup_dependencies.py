"""
LLM Inspector v12.0 - Dependency Setup Script

Checks and installs required dependencies.
Skips packages that are already installed with compatible versions.

Usage:
    python scripts/setup_dependencies.py [--upgrade]
    python scripts/setup_dependencies.py --check-only
    python scripts/setup_dependencies.py --skip-optional

Optional dependencies will only be installed if requested or available.
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dependency:
    """Dependency specification."""
    name: str
    import_name: str  # Name used in import statement
    min_version: str = "0.0.0"
    max_version: Optional[str] = None
    optional: bool = False
    description: str = ""

    def __repr__(self) -> str:
        version_spec = f">={self.min_version}"
        if self.max_version:
            version_spec += f",<={self.max_version}"
        opt_marker = " [optional]" if self.optional else ""
        return f"{self.name}{version_spec}{opt_marker}"


# ============================================================
# v13.0 Core Dependencies (always required)
# ============================================================
CORE_DEPENDENCIES = [
    Dependency("numpy", "numpy", "1.24.0", description="Numerical computing (IRT, scoring, similarity)"),
    Dependency("scipy", "scipy", "1.10.0", description="Scientific computing (optimization, stats)"),
    Dependency("cryptography", "cryptography", "41.0.0", description="AES-GCM encryption for API keys"),
    Dependency("PyYAML", "yaml", "6.0", description="SOURCES.yaml provenance registry"),
]

# ============================================================
# Recommended Dependencies (used in main code paths, with fallbacks)
# ============================================================
RECOMMENDED_DEPENDENCIES = [
    Dependency("requests", "requests", "2.28.0", optional=True,
               description="HTTP client for Wikidata / DBpedia API calls"),
    Dependency("scikit-learn", "sklearn", "1.3.0", optional=True,
               description="Factor analysis and PCA"),
    # v13: promoted from optional — needed for dual-source KG queries
    Dependency("SPARQLWrapper", "SPARQLWrapper", "2.0.0", optional=True,
               description="DBpedia + Wikidata SPARQL dual-source knowledge graph"),
    # v13: promoted from optional — needed for tokenizer fingerprinting
    Dependency("tiktoken", "tiktoken", "0.7.0", optional=True,
               description="Accurate token counting + tokenizer fingerprint probes"),
]

# ============================================================
# Optional Dependencies (conditional imports, graceful degradation)
# ============================================================
OPTIONAL_DEPENDENCIES = [
    # PyTorch (required by sentence-transformers)
    Dependency("torch", "torch", "2.0.0", optional=True,
               description="Deep learning framework for embeddings"),

    # Semantic similarity (falls back to rule-based if missing)
    Dependency("sentence-transformers", "sentence_transformers",
               "2.2.0", optional=True,
               description="Local semantic similarity without API calls"),

    # Distributed task queue (uses ThreadPoolExecutor fallback)
    Dependency("celery", "celery", "5.3.6", optional=True,
               description="Distributed task queue"),
    Dependency("redis", "redis", "5.0.1", optional=True,
               description="Broker for Celery"),

    # v13: HuggingFace datasets — suite_v13 benchmark fetching (Phase 3)
    Dependency("datasets", "datasets", "2.19.0", optional=True,
               description="HuggingFace datasets library for GPQA/MMLU-Pro/SWE-bench fetching"),

    # v13: async HTTP for parallel KG queries (Phase 5)
    Dependency("httpx", "httpx", "0.27.0", optional=True,
               description="Async HTTP client for parallel Wikidata+DBpedia queries"),

    # REST API routes (optional FastAPI integration)
    Dependency("fastapi", "fastapi", "0.104.0", optional=True,
               description="REST API (alternative to stdlib HTTP)"),
    Dependency("uvicorn", "uvicorn", "0.24.0", optional=True,
               description="ASGI server for FastAPI"),

    # YAML config parsing (uses json fallback)
    Dependency("PyYAML", "yaml", "6.0", optional=True,
               description="YAML configuration file parsing"),

    # Report templates (plain text fallback)
    Dependency("jinja2", "jinja2", "3.1.0", optional=True,
               description="Template engine for report generation"),

    # Advanced NLP (optional enhancement)
    Dependency("spacy", "spacy", "3.7.0", optional=True,
               description="Entity extraction and NLP pipelines"),

    # Advanced DB (SQLite fallback)
    Dependency("sqlalchemy", "sqlalchemy", "2.0.0", optional=True,
               description="SQL toolkit for advanced database operations"),
]

ALL_DEPENDENCIES = (
    CORE_DEPENDENCIES +
    RECOMMENDED_DEPENDENCIES +
    OPTIONAL_DEPENDENCIES
)


def get_installed_version(package_name: str) -> Optional[str]:
    """Get installed version of a package using importlib.metadata."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(package_name)
        except PackageNotFoundError:
            return None
    except ImportError:
        # Python < 3.8 (extremely unlikely for v11.0)
        pass

    # Last resort: try importing and check __version__
    try:
        module = importlib.import_module(package_name.replace("-", "_"))
        return getattr(module, "__version__", None)
    except Exception:
        return None


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to comparable tuple."""
    parts = version_str.split(".")
    result = []
    for part in parts:
        # Extract leading digits
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            result.append(int(digits))
        else:
            result.append(0)
    return tuple(result)


def version_satisfies(installed: str, required_min: str, required_max: Optional[str] = None) -> bool:
    """Check if installed version satisfies requirements."""
    try:
        inst = parse_version(installed)
        req_min = parse_version(required_min)

        if inst < req_min:
            return False

        if required_max:
            req_max = parse_version(required_max)
            if inst > req_max:
                return False

        return True
    except Exception:
        return False


def check_dependency(dep: Dependency) -> Tuple[bool, str]:
    """
    Check if dependency is satisfied.

    Returns:
        Tuple of (is_satisfied, status_message)
    """
    installed_version = get_installed_version(dep.name)

    if installed_version is None:
        return False, "Not installed"

    if version_satisfies(installed_version, dep.min_version, dep.max_version):
        status = f"OK (v{installed_version})"
        if dep.optional:
            status += " [optional]"
        return True, status
    else:
        return False, f"Version mismatch (installed: v{installed_version}, required: >={dep.min_version})"


def install_package(dep: Dependency, upgrade: bool = False) -> bool:
    """Install a package using pip."""
    cmd = [sys.executable, "-m", "pip", "install"]

    if upgrade:
        cmd.append("--upgrade")

    # Build version specifier
    version_spec = f">={dep.min_version}"
    if dep.max_version:
        version_spec += f",<={dep.max_version}"

    cmd.append(f"{dep.name}{version_spec}")

    print(f"  Installing {dep}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print(f"    [OK] Successfully installed {dep.name}")
            return True
        else:
            print(f"    [FAIL] Installation failed:")
            print(f"    {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    [FAIL] Installation timed out")
        return False
    except Exception as e:
        print(f"    [FAIL] Installation error: {e}")
        return False


def check_and_install(
    dependencies: List[Dependency],
    upgrade: bool = False,
    skip_optional: bool = False
) -> Dict[str, bool]:
    """
    Check and install dependencies.

    Returns:
        Dict mapping package name to installation status
    """
    results = {}

    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)

    to_install = []

    # First pass: check all dependencies
    for dep in dependencies:
        if skip_optional and dep.optional:
            print(f"  {dep.name}: [SKIP] Optional dependency")
            results[dep.name] = True  # Mark as OK since it's optional
            continue

        is_satisfied, status = check_dependency(dep)

        if is_satisfied:
            print(f"  {dep.name}: [OK] {status}")
            results[dep.name] = True
        else:
            print(f"  {dep.name}: [NEED] {status}")
            to_install.append(dep)

    # Second pass: install missing dependencies
    if to_install:
        print("\n" + "=" * 60)
        print("Installing Missing Dependencies")
        print("=" * 60)

        for dep in to_install:
            if install_package(dep, upgrade):
                results[dep.name] = True
            else:
                results[dep.name] = False
                if not dep.optional:
                    print(f"    WARNING: Required dependency {dep.name} failed to install!")
    else:
        print("\n  All dependencies satisfied!")

    return results


def generate_requirements_txt(output_path: str = "requirements.txt"):
    """Generate requirements.txt file matching v12.0 actual dependencies."""
    lines = [
        "# LLM Inspector v12.0 - Required Dependencies",
        "# Auto-generated by setup_dependencies.py - do not edit manually",
        "",
        "# === Core (required) ===",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "cryptography>=41.0.0",
        "",
        "# === Recommended (used with fallbacks) ===",
        "requests>=2.28.0",
        "scikit-learn>=1.3.0",
        "",
        "# === Optional (conditional imports, graceful degradation) ===",
        "# sentence-transformers>=2.2.0  # Local semantic similarity",
        "# celery>=5.3.6                # Distributed task queue",
        "# redis>=5.0.1                 # Celery broker",
        "# SPARQLWrapper>=2.0.0         # DBpedia/Wikidata SPARQL",
        "# tiktoken>=0.6.0              # Accurate token counting",
        "# fastapi>=0.104.0             # REST API (alternative)",
        "# uvicorn>=0.24.0              # ASGI server",
        "# PyYAML>=6.0                  # YAML config parsing",
        "# jinja2>=3.1.0                # Report templates",
        "# spacy>=3.7.0                 # Advanced NLP",
        "# sqlalchemy>=2.0.0            # Advanced DB operations",
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n[OK] Generated {output_path}")


def print_summary(results: Dict[str, bool], dependencies: List[Dependency]):
    """Print installation summary."""
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)

    required_failed = []
    optional_failed = []

    for dep in dependencies:
        success = results.get(dep.name, False)
        status = "[OK]" if success else "[FAIL]"
        opt_marker = " (optional)" if dep.optional else ""

        if not success:
            if dep.optional:
                optional_failed.append(dep.name)
            else:
                required_failed.append(dep.name)

        print(f"  {status} {dep.name}{opt_marker}")

    print("\n" + "-" * 60)

    if not required_failed and not optional_failed:
        print("[SUCCESS] All dependencies installed and ready!")
        return True
    elif not required_failed:
        print(f"[WARNING] Optional packages failed: {', '.join(optional_failed)}")
        print("          Core functionality is available.")
        return True
    else:
        print(f"[ERROR] Required packages failed: {', '.join(required_failed)}")
        print("        Please install these manually:")
        for pkg in required_failed:
            print(f"          pip install {pkg}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup LLM Inspector v12.0 dependencies"
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade packages even if already installed"
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional dependencies"
    )
    parser.add_argument(
        "--generate-requirements",
        action="store_true",
        help="Generate requirements.txt and exit"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check dependencies without installing"
    )

    args = parser.parse_args()

    # Generate requirements.txt and exit
    if args.generate_requirements:
        base_dir = Path(__file__).parent.parent
        req_path = base_dir / "requirements.txt"
        generate_requirements_txt(str(req_path))
        return 0

    # Deduplicate dependencies (keep most specific requirement)
    dep_dict: Dict[str, Dependency] = {}
    for dep in ALL_DEPENDENCIES:
        if dep.name in dep_dict:
            # Keep the one with higher minimum version
            existing = dep_dict[dep.name]
            if parse_version(dep.min_version) > parse_version(existing.min_version):
                dep_dict[dep.name] = dep
        else:
            dep_dict[dep.name] = dep

    unique_deps = list(dep_dict.values())

    print("\n" + "=" * 60)
    print("LLM Inspector v12.0 - Dependency Setup")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Packages to check: {len(unique_deps)}")

    if args.check_only:
        # Just check, don't install
        print("\n[CHECK-ONLY MODE]")
        all_ok = True
        for dep in unique_deps:
            if args.skip_optional and dep.optional:
                continue
            is_satisfied, status = check_dependency(dep)
            marker = "[OK]" if is_satisfied else "[MISSING]"
            opt = " [optional]" if dep.optional else ""
            print(f"  {marker} {dep.name}{opt}: {status}")
            if not is_satisfied and not dep.optional:
                all_ok = False
        return 0 if all_ok else 1

    # Check and install
    results = check_and_install(
        unique_deps,
        upgrade=args.upgrade,
        skip_optional=args.skip_optional
    )

    # Print summary
    success = print_summary(results, unique_deps)

    # Generate requirements.txt for reference
    if not args.check_only:
        base_dir = Path(__file__).parent.parent
        req_path = base_dir / "requirements.txt"
        generate_requirements_txt(str(req_path))

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
