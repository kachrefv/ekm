#!/usr/bin/env python3
"""Run tests for EKM."""

import subprocess
import sys
import argparse


def run_tests(test_type="all", verbose=False):
    """Run tests based on type."""
    cmd = [sys.executable, "-m", "pytest"]
    
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "all":
        cmd.append("tests/")
    
    if verbose:
        cmd.append("-v")
    
    # Add coverage if available
    cmd.extend(["--cov=ekm", "--cov-report=html"])
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EKM tests")
    parser.add_argument("--type", choices=["unit", "integration", "all"], default="all",
                       help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    sys.exit(run_tests(args.type, args.verbose))