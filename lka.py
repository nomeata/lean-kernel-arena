#!/usr/bin/env python3
"""Lean Kernel Arena - Tool for managing Lean kernel tests and checkers."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.resolve()


def load_yaml_files(directory: Path) -> list[dict]:
    """Load all YAML files from a directory."""
    items = []
    if not directory.exists():
        return items
    for file in directory.glob("*.yaml"):
        with open(file, "r") as f:
            data = yaml.safe_load(f)
            data["_file"] = file.name
            items.append(data)
    return items


def load_tests() -> list[dict]:
    """Load all test definitions."""
    return load_yaml_files(get_project_root() / "tests")


def load_checkers() -> list[dict]:
    """Load all checker definitions."""
    return load_yaml_files(get_project_root() / "checkers")


def find_test_by_name(name: str) -> dict | None:
    """Find a test by name."""
    for test in load_tests():
        if test.get("name") == name:
            return test
    return None


def find_checker_by_name(name: str) -> dict | None:
    """Find a checker by name."""
    for checker in load_checkers():
        if checker.get("name") == name:
            return checker
    return None


# =============================================================================
# create-test command
# =============================================================================


def create_test_lake(test: dict, output_dir: Path) -> bool:
    """Create a test from a lake project."""
    name = test["name"]
    url = test["url"]
    ref = test.get("ref")
    rev = test["rev"]
    module = test["module"]
    pre_build = test.get("pre-build")

    work_dir = output_dir / "work" / name
    work_dir.mkdir(parents=True, exist_ok=True)

    # Clone the repository
    print(f"  Cloning {url}...")
    clone_cmd = ["git", "clone", "--depth=1"]
    if ref:
        clone_cmd.extend(["--branch", ref])
    clone_cmd.extend([url, str(work_dir / "repo")])

    result = subprocess.run(clone_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error cloning: {result.stderr}")
        return False

    repo_dir = work_dir / "repo"

    # Checkout specific revision
    result = subprocess.run(
        ["git", "checkout", rev],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Error checking out {rev}: {result.stderr}")
        return False

    # Run pre-build command if specified
    if pre_build:
        print(f"  Running pre-build: {pre_build}")
        result = subprocess.run(
            pre_build,
            shell=True,
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Pre-build failed: {result.stderr}")
            return False

    # Build and export the module
    print(f"  Building and exporting module {module}...")
    output_file = output_dir / f"{name}.leantar"
    export_cmd = f"lake env lean -o {output_file} --export={output_file} {module}"

    result = subprocess.run(
        export_cmd,
        shell=True,
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Export failed: {result.stderr}")
        return False

    print(f"  Created {output_file}")
    return True


def create_test_script(test: dict, output_dir: Path) -> bool:
    """Create a test by running a script."""
    name = test["name"]
    run_cmd = test["run"]
    output_file = output_dir / f"{name}.leantar"

    print(f"  Running: {run_cmd} {output_file}")
    result = subprocess.run(
        f"{run_cmd} {output_file}",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Script failed: {result.stderr}")
        return False

    print(f"  Created {output_file}")
    return True


def create_test_static(test: dict, output_dir: Path) -> bool:
    """Create a test by copying a static file."""
    name = test["name"]
    source_file = get_project_root() / test["file"]
    output_file = output_dir / f"{name}.leantar"

    if not source_file.exists():
        print(f"  Source file not found: {source_file}")
        return False

    shutil.copy(source_file, output_file)
    print(f"  Copied {source_file} to {output_file}")
    return True


def create_test(test: dict, output_dir: Path) -> bool:
    """Create a single test."""
    name = test["name"]
    test_type = test["type"]
    print(f"Creating test: {name} (type: {test_type})")

    output_dir.mkdir(parents=True, exist_ok=True)

    if test_type == "lake":
        return create_test_lake(test, output_dir)
    elif test_type == "script":
        return create_test_script(test, output_dir)
    elif test_type == "static":
        return create_test_static(test, output_dir)
    else:
        print(f"  Unknown test type: {test_type}")
        return False


def cmd_create_test(args: argparse.Namespace) -> int:
    """Handle the create-test command."""
    output_dir = get_project_root() / "_build" / "tests"

    if args.name:
        test = find_test_by_name(args.name)
        if test is None:
            print(f"Test not found: {args.name}")
            return 1
        tests = [test]
    else:
        tests = load_tests()

    if not tests:
        print("No tests found.")
        return 0

    success = 0
    failed = 0
    for test in tests:
        if create_test(test, output_dir):
            success += 1
        else:
            failed += 1

    print(f"\nResults: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


# =============================================================================
# build-checker command
# =============================================================================


def build_checker(checker: dict, build_dir: Path) -> bool:
    """Build a single checker."""
    name = checker["name"]
    version = checker.get("version", "unknown")
    url = checker.get("url")
    ref = checker.get("ref")
    rev = checker.get("rev")
    build_cmd = checker.get("build")

    print(f"Building checker: {name} (version: {version})")

    checker_dir = build_dir / name
    checker_dir.mkdir(parents=True, exist_ok=True)

    # Clone repository if URL is provided
    if url:
        repo_dir = checker_dir / "repo"
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

        print(f"  Cloning {url}...")
        clone_cmd = ["git", "clone", "--depth=1"]
        if ref:
            clone_cmd.extend(["--branch", ref])
        clone_cmd.extend([url, str(repo_dir)])

        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error cloning: {result.stderr}")
            return False

        # Checkout specific revision if specified
        if rev:
            result = subprocess.run(
                ["git", "checkout", rev],
                cwd=repo_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  Error checking out {rev}: {result.stderr}")
                return False

        work_dir = repo_dir
    else:
        work_dir = checker_dir

    # Run build command if specified
    if build_cmd:
        print(f"  Building: {build_cmd}")
        result = subprocess.run(
            build_cmd,
            shell=True,
            cwd=work_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Build failed: {result.stderr}")
            return False

    print(f"  Checker {name} built successfully")
    return True


def cmd_build_checker(args: argparse.Namespace) -> int:
    """Handle the build-checker command."""
    build_dir = get_project_root() / "_build" / "checkers"

    if args.name:
        checker = find_checker_by_name(args.name)
        if checker is None:
            print(f"Checker not found: {args.name}")
            return 1
        checkers = [checker]
    else:
        checkers = load_checkers()

    if not checkers:
        print("No checkers found.")
        return 0

    success = 0
    failed = 0
    for checker in checkers:
        if build_checker(checker, build_dir):
            success += 1
        else:
            failed += 1

    print(f"\nResults: {success} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


# =============================================================================
# run-checker command
# =============================================================================


def run_checker_on_test(checker: dict, test: dict, build_dir: Path, tests_dir: Path) -> dict:
    """Run a checker on a test and return the result."""
    checker_name = checker["name"]
    test_name = test["name"]
    run_cmd = checker["run"]

    test_file = tests_dir / f"{test_name}.leantar"
    if not test_file.exists():
        return {
            "checker": checker_name,
            "test": test_name,
            "status": "error",
            "message": f"Test file not found: {test_file}",
            "exit_code": -1,
        }

    # Determine working directory
    checker_dir = build_dir / checker_name
    if checker.get("url"):
        work_dir = checker_dir / "repo"
    else:
        work_dir = checker_dir

    work_dir.mkdir(parents=True, exist_ok=True)

    # Run the checker
    full_cmd = f"{run_cmd} {test_file}"
    result = subprocess.run(
        full_cmd,
        shell=True,
        cwd=work_dir,
        capture_output=True,
        text=True,
    )

    exit_code = result.returncode
    if exit_code == 0:
        status = "accepted"
    elif exit_code == 1:
        status = "rejected"
    elif exit_code == 2:
        status = "declined"
    else:
        status = "error"

    return {
        "checker": checker_name,
        "test": test_name,
        "status": status,
        "exit_code": exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def cmd_run_checker(args: argparse.Namespace) -> int:
    """Handle the run-checker command."""
    build_dir = get_project_root() / "_build" / "checkers"
    tests_dir = get_project_root() / "_build" / "tests"

    # Determine which checkers to run
    if args.checker:
        checker = find_checker_by_name(args.checker)
        if checker is None:
            print(f"Checker not found: {args.checker}")
            return 1
        checkers = [checker]
    else:
        checkers = load_checkers()

    # Determine which tests to run
    if args.test:
        test = find_test_by_name(args.test)
        if test is None:
            print(f"Test not found: {args.test}")
            return 1
        tests = [test]
    else:
        tests = load_tests()

    if not checkers:
        print("No checkers found.")
        return 0

    if not tests:
        print("No tests found.")
        return 0

    results = []
    for checker in checkers:
        for test in tests:
            print(f"Running {checker['name']} on {test['name']}...", end=" ")
            result = run_checker_on_test(checker, test, build_dir, tests_dir)
            results.append(result)
            print(f"[{result['status']}]")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    status_counts = {"accepted": 0, "rejected": 0, "declined": 0, "error": 0}
    for r in results:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    for status, count in status_counts.items():
        if count > 0:
            print(f"  {status}: {count}")

    return 0


# =============================================================================
# build-site command
# =============================================================================


def cmd_build_site(args: argparse.Namespace) -> int:
    """Handle the build-site command."""
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    templates_dir = get_project_root() / "templates"
    if not templates_dir.exists():
        print(f"Templates directory not found: {templates_dir}")
        return 1

    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(),
    )

    tests = load_tests()
    checkers = load_checkers()

    # Build context data
    data = {
        "tests": tests,
        "checkers": checkers,
    }

    # Render index.html
    try:
        template = env.get_template("index.html")
        output_file = output_dir / "index.html"
        template.stream(data).dump(str(output_file))
        print(f"Generated: {output_file}")
    except Exception as e:
        print(f"Error rendering template: {e}")
        return 1

    # Copy static files if they exist
    static_dir = templates_dir / "static"
    if static_dir.exists():
        shutil.copytree(static_dir, output_dir / "static", dirs_exist_ok=True)
        print(f"Copied static files to: {output_dir / 'static'}")

    print(f"\nSite built successfully in: {output_dir}")
    return 0


# =============================================================================
# Main entry point
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="lka",
        description="Lean Kernel Arena - Tool for managing Lean kernel tests and checkers",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # create-test command
    create_test_parser = subparsers.add_parser(
        "create-test",
        help="Create test files from test definitions",
    )
    create_test_parser.add_argument(
        "name",
        nargs="?",
        help="Name of the test to create (default: all tests)",
    )

    # build-checker command
    build_checker_parser = subparsers.add_parser(
        "build-checker",
        help="Build checkers from checker definitions",
    )
    build_checker_parser.add_argument(
        "name",
        nargs="?",
        help="Name of the checker to build (default: all checkers)",
    )

    # run-checker command
    run_checker_parser = subparsers.add_parser(
        "run-checker",
        help="Run checkers on tests",
    )
    run_checker_parser.add_argument(
        "--checker",
        help="Name of the checker to run (default: all checkers)",
    )
    run_checker_parser.add_argument(
        "--test",
        help="Name of the test to run (default: all tests)",
    )

    # build-site command
    build_site_parser = subparsers.add_parser(
        "build-site",
        help="Build the website",
    )
    build_site_parser.add_argument(
        "--outdir",
        default="_out",
        help="Output directory for the website (default: _out)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "create-test":
        return cmd_create_test(args)
    elif args.command == "build-checker":
        return cmd_build_checker(args)
    elif args.command == "run-checker":
        return cmd_run_checker(args)
    elif args.command == "build-site":
        return cmd_build_site(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
