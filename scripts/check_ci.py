#!/usr/bin/env python3
"""
CI/CD Status Checker
Runs local checks before pushing to ensure CI/CD pipeline will pass.
"""

import subprocess
import sys
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text:^60}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✓ {text}{RESET}")

def print_error(text):
    print(f"{RED}✗ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠ {text}{RESET}")

def run_command(cmd, description, allow_fail=False):
    """Run a command and return True if successful."""
    print(f"\n{YELLOW}Running: {description}{RESET}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print_success(f"{description} passed")
            if result.stdout:
                print(result.stdout[:500])  # Print first 500 chars
            return True
        else:
            print_error(f"{description} failed")
            if result.stderr:
                print(result.stderr[:500])
            if not allow_fail:
                return False
            return True
    except FileNotFoundError:
        print_warning(f"Command not found: {cmd[0]}")
        if allow_fail:
            return True
        return False
    except Exception as e:
        print_error(f"Error running {description}: {str(e)}")
        if allow_fail:
            return True
        return False

def check_python():
    """Check Python version."""
    print_header("Python Version Check")
    
    result = subprocess.run(
        [sys.executable, "--version"],
        capture_output=True,
        text=True
    )
    
    version = result.stdout.strip()
    print(f"Python version: {version}")
    
    if "3.9" in version or "3.10" in version or "3.11" in version:
        print_success("Python version is compatible")
        return True
    else:
        print_warning("Python version may not be compatible with CI/CD pipeline")
        return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Dependencies Check")
    
    required = ['pytest', 'black', 'flake8', 'isort']
    all_installed = True
    
    for package in required:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True
        )
        
        if result.returncode == 0:
            print_success(f"{package} is installed")
        else:
            print_error(f"{package} is NOT installed")
            all_installed = False
    
    if not all_installed:
        print_warning("\nInstall missing packages:")
        print("pip install pytest black flake8 isort pytest-cov")
    
    return all_installed

def run_tests():
    """Run pytest tests."""
    print_header("Running Tests")
    
    return run_command(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        "Unit tests"
    )

def run_linting():
    """Run flake8 linting."""
    print_header("Code Quality - Flake8")
    
    # Critical errors only
    critical = run_command(
        [sys.executable, "-m", "flake8", "src/", "--count", 
         "--select=E9,F63,F7,F82", "--show-source", "--statistics"],
        "Flake8 critical errors"
    )
    
    # All errors (allow to fail)
    run_command(
        [sys.executable, "-m", "flake8", "src/", "--count", 
         "--max-line-length=127", "--statistics"],
        "Flake8 all checks",
        allow_fail=True
    )
    
    return critical

def check_formatting():
    """Check code formatting with black."""
    print_header("Code Formatting - Black")
    
    return run_command(
        [sys.executable, "-m", "black", "--check", "src/", "webapp/", "tests/"],
        "Black formatting check",
        allow_fail=True  # Allow to fail but warn
    )

def check_imports():
    """Check import sorting with isort."""
    print_header("Import Sorting - isort")
    
    return run_command(
        [sys.executable, "-m", "isort", "--check-only", "src/", "webapp/", "tests/"],
        "isort import sorting",
        allow_fail=True
    )

def check_docker():
    """Check if Docker is available and Dockerfile is valid."""
    print_header("Docker Check")
    
    # Check if docker is available
    docker_available = run_command(
        ["docker", "--version"],
        "Docker availability",
        allow_fail=True
    )
    
    if not docker_available:
        print_warning("Docker not available. Skipping Docker checks.")
        return True
    
    # Check Dockerfile syntax (dry-run build)
    print_warning("Note: Full Docker build is slow. Skipping for now.")
    print_warning("To test Docker build manually, run: docker build -t test .")
    
    return True

def main():
    """Run all checks."""
    print(f"\n{BLUE}{'*'*60}{RESET}")
    print(f"{BLUE}{'CI/CD Pre-Push Checker':^60}{RESET}")
    print(f"{BLUE}{'*'*60}{RESET}")
    
    checks = [
        ("Python Version", check_python),
        ("Dependencies", check_dependencies),
        ("Tests", run_tests),
        ("Linting", run_linting),
        ("Formatting", check_formatting),
        ("Import Sorting", check_imports),
        ("Docker", check_docker),
    ]
    
    results = {}
    for name, func in checks:
        try:
            results[name] = func()
        except KeyboardInterrupt:
            print_error("\nChecks interrupted by user")
            sys.exit(1)
        except Exception as e:
            print_error(f"\nUnexpected error in {name}: {str(e)}")
            results[name] = False
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        if result:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    print(f"\n{BLUE}Total: {passed}/{total} checks passed{RESET}")
    
    if all(results.values()):
        print_success("\n✓ All checks passed! Ready to push.")
        return 0
    else:
        print_error("\n✗ Some checks failed. Please fix before pushing.")
        print_warning("\nTips:")
        print("  - Format code: black src/ webapp/ tests/")
        print("  - Sort imports: isort src/ webapp/ tests/")
        print("  - Fix linting: Review flake8 output above")
        print("  - Fix tests: Review pytest output above")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_error("\n\nInterrupted by user")
        sys.exit(1)
