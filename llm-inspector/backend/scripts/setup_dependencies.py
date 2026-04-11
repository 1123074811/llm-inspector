"""
LLM Inspector v7.0 - Dependency Setup Script

Checks and installs required dependencies.
Skips packages that are already installed with compatible versions.

Usage:
    python scripts/setup_dependencies.py [--upgrade]

Optional dependencies marked with [optional] will only be installed if requested.
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


# Core dependencies (required for basic functionality)
CORE_DEPENDENCIES = [
    Dependency("numpy", "numpy", "1.24.0", description="Numerical computing"),
    Dependency("scipy", "scipy", "1.10.0", description="Scientific computing (optimization, stats)"),
]

# v7 Phase 1 dependencies (scientific scoring)
PHASE1_DEPENDENCIES = [
    Dependency("numpy", "numpy", "1.24.0", description="IRT calibration matrix operations"),
    Dependency("scipy", "scipy", "1.10.0", description="Optimization for IRT MLE"),
]

# v7 Phase 2 dependencies (core algorithms)
PHASE2_DEPENDENCIES = [
    # CAT and Bayesian fusion use numpy/scipy already in CORE
    
    # Semantic Judge v3 Tier 2 (optional - falls back to rules if not installed)
    Dependency(
        "sentence-transformers", "sentence_transformers",
        "2.2.0", optional=True,
        description="Local semantic similarity without API calls (recommended)"
    ),
    
    # For knowledge graph integration in hallucination detection (optional)
    Dependency(
        "requests", "requests",
        "2.28.0", optional=True,
        description="HTTP client for Wikidata API calls"
    ),
]

# v7 Phase 3+ dependencies (future phases)
FUTURE_DEPENDENCIES = [
    # For advanced NLP features
    Dependency(
        "spacy", "spacy",
        "3.7.0", optional=True,
        description="Entity extraction and NLP pipelines"
    ),
    
    # For data validation YAML parsing
    Dependency(
        "PyYAML", "yaml",
        "6.0", optional=True,
        description="YAML configuration file parsing"
    ),
    
    # For report generation
    Dependency(
        "jinja2", "jinja2",
        "3.1.0", optional=True,
        description="Template engine for report generation"
    ),
    
    # For database (if not using sqlite3)
    Dependency(
        "sqlalchemy", "sqlalchemy",
        "2.0.0", optional=True,
        description="SQL toolkit for advanced database operations"
    ),
]

ALL_DEPENDENCIES = (
    CORE_DEPENDENCIES +
    PHASE1_DEPENDENCIES +
    PHASE2_DEPENDENCIES +
    FUTURE_DEPENDENCIES
)


def get_installed_version(package_name: str) -> Optional[str]:
    """Get installed version of a package."""
    try:
        # First try importlib.metadata (Python 3.8+)
        try:
            from importlib.metadata import version
            return version(package_name)
        except ImportError:
            pass
        
        # Fallback to pkg_resources
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except:
            pass
        
        # Last resort: try importing and check __version__
        try:
            module = importlib.import_module(package_name.replace("-", "_"))
            return getattr(module, "__version__", None)
        except:
            pass
        
        return None
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
        return False, f"Not installed"
    
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
    """Generate requirements.txt file."""
    lines = [
        "# LLM Inspector v7.0 - Required Dependencies",
        "# Generated automatically - do not edit manually",
        "",
        "# Core dependencies",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "",
        "# Optional but recommended",
        "sentence-transformers>=2.2.0  # For local semantic similarity",
        "requests>=2.28.0  # For knowledge graph API",
        "",
        "# Optional dependencies",
        "# PyYAML>=6.0  # For YAML config parsing",
        "# spacy>=3.7.0  # For advanced NLP",
        "# jinja2>=3.1.0  # For report templates",
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
        description="Setup LLM Inspector v7.0 dependencies"
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
    print("LLM Inspector v7.0 - Dependency Setup")
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
