"""
LLM Inspector v9.0 - Smart Startup Script

Automatically checks and installs dependencies before starting the service.

Usage:
    python start.py [--port 8080] [--host 0.0.0.0] [--skip-deps]

Features:
- Dependency check with auto-install
- Server health check
- Graceful error handling
- Environment validation
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add app to path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))


def run_dependency_check():
    """Run dependency setup script."""
    setup_script = Path(__file__).parent / "scripts" / "setup_dependencies.py"
    
    if not setup_script.exists():
        print("[WARN] Dependency setup script not found, skipping dependency check")
        return True
    
    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(setup_script), "--skip-optional"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("[WARN] Dependency check warnings:")
            print(result.stderr[:500])
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("[WARN] Dependency check timed out, continuing anyway")
        return True
    except Exception as e:
        print(f"[WARN] Dependency check failed: {e}, continuing anyway")
        return True


def check_environment(strict_provenance: bool = False):
    """Check environment requirements."""
    print("\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    else:
        print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check critical imports
    critical_modules = ["numpy", "scipy"]
    for module in critical_modules:
        try:
            __import__(module)
            print(f"[OK] {module} available")
        except ImportError:
            issues.append(f"Critical module '{module}' not available")
            print(f"[FAIL] {module} not available")
    
    # Check app structure
    app_dir = Path(__file__).parent / "app"
    required_dirs = ["analysis", "core", "handlers", "judge", "predetect"]
    
    for dir_name in required_dirs:
        dir_path = app_dir / dir_name
        if dir_path.exists():
            print(f"[OK] Directory {dir_name}/ exists")
        else:
            issues.append(f"Required directory '{dir_name}' not found")
            print(f"[FAIL] Directory {dir_name}/ not found")

    # v9 Phase A: provenance strict-mode validation
    try:
        from app.analysis.metric_registry import validate_required_metric_sources
        p_issues = validate_required_metric_sources(strict=False)
        if p_issues:
            for issue in p_issues:
                print(f"[WARN] Provenance: {issue}")
            if strict_provenance:
                issues.extend([f"Provenance validation failed: {x}" for x in p_issues])
        else:
            print("[OK] Metric provenance registry validation passed")
    except Exception as e:
        msg = f"Provenance check error: {e}"
        print(f"[WARN] {msg}")
        if strict_provenance:
            issues.append(msg)
    
    if issues:
        print("\n[ERROR] Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("\n[OK] Environment check passed")
    return True


def run_server(host: str, port: int, reload: bool = False):
    """Run the HTTP server."""
    print("\n" + "=" * 60)
    print("Starting LLM Inspector v9.0")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Reload: {reload}")
    print(f"Working directory: {Path(__file__).parent.absolute()}")
    print("=" * 60)
    print()
    
    try:
        # Import and run the main server
        from app.main import run_server as start_http_server
        start_http_server(host=host, port=port)
    except ImportError as e:
        print(f"[ERROR] Failed to import server module: {e}")
        print("\nTrying alternative import...")
        
        try:
            # Alternative: run as module
            import app.main
            if hasattr(app.main, 'main'):
                app.main.main()
            else:
                print("[ERROR] No main() function found in app.main")
                return 1
        except Exception as e2:
            print(f"[ERROR] Failed to start server: {e2}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Inspector v9.0 Startup Script"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--host", "-H",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency check"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (if supported)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Fail startup when required metric provenance is missing"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print("LLM Inspector v9.0 - Startup")
    print("=" * 60)
    
    # Check dependencies
    if not args.skip_deps:
        if not run_dependency_check():
            print("\n[ERROR] Dependency check failed!")
            print("Run with --skip-deps to skip this check")
            return 1
    else:
        print("\n[NOTE] Skipping dependency check (--skip-deps)")
    
    # Check environment
    strict_provenance = args.strict_provenance or os.getenv("STRICT_PROVENANCE", "false").lower() == "true"
    if not check_environment(strict_provenance=strict_provenance):
        print("\n[ERROR] Environment check failed!")
        return 1
    
    # Set environment variables
    if args.debug:
        os.environ["DEBUG"] = "1"
        print("\n[NOTE] Debug mode enabled")
    
    # Run server
    return run_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    sys.exit(main())
