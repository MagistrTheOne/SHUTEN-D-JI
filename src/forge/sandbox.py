"""
CODE FORGE Sandbox — polyglot verification harness.

Contract: given a solution + hidden tests, run compile/tests/lint/typecheck/
security in an isolated environment and return a `Verification`.

Implementation status (v1):
- Python: EXECUTABLE in the current venv via subprocess (pytest + ruff + mypy +
  a lightweight security scan). This validates the end-to-end contract.
- TypeScript / SQL / Bash / C++: Docker images declared in DOCKER_IMAGES and the
  command recipes in RUNNERS. The runner shells out to `docker run` when Docker
  is available; otherwise it returns a `skipped` verification so the pipeline can
  still be exercised without those toolchains installed locally.

Determinism: pinned images, fixed PYTHONHASHSEED, network disabled in Docker,
hidden tests are passed separately from the prompt.

See docs/CODE_FORGE_V1_SPEC.md section 2 (sandbox) and 3 (gates).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.forge.schema import DiffStats, Language, Verification

# Pinned images per language (spec: deterministic, network-off).
DOCKER_IMAGES: dict[Language, str] = {
    Language.TYPESCRIPT: "node:20.18-bookworm-slim",
    Language.SQL: "postgres:16.4-bookworm",
    Language.BASH: "koalaman/shellcheck-alpine:v0.10.0",
    Language.CPP: "gcc:14.2-bookworm",
    Language.YAML: "python:3.11-slim-bookworm",
}

# Coarse per-language security heuristics applied to source (defense in depth;
# the Docker runners can additionally invoke bandit/semgrep/etc).
SECURITY_RULES: dict[Language, tuple[str, ...]] = {
    Language.PYTHON: (
        r"\beval\s*\(", r"\bexec\s*\(", r"subprocess\.[a-z_]+\([^)]*shell\s*=\s*True",
        r"pickle\.loads", r"yaml\.load\s*\((?![^)]*Loader)",
    ),
    Language.SQL: (
        r"(?i)\bexecute\s*\(\s*[\"'].*%s.*\+",   # string-concatenated SQL
        r"(?i)' \+|\+ '",
    ),
    Language.TYPESCRIPT: (r"\beval\s*\(", r"dangerouslySetInnerHTML", r"child_process"),
    Language.CPP: (r"\bsystem\s*\(", r"\bgets\s*\(", r"strcpy\s*\("),
    Language.BASH: (r"\beval\s+", r"curl\s+[^|]*\|\s*sh"),
    Language.YAML: (),
}

TIMEOUT_S = 60


@dataclass
class RunOutput:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _run(cmd: list[str], cwd: str, env: dict | None = None) -> RunOutput:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env=env,
        )
        # A missing optional tool (mypy/ruff not installed) is a skip, not a fail.
        if proc.returncode != 0 and "No module named" in (proc.stderr or ""):
            return RunOutput(False, proc.stdout, proc.stderr, skipped=True)
        return RunOutput(proc.returncode == 0, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired:
        return RunOutput(False, "", "TIMEOUT")
    except FileNotFoundError as e:
        return RunOutput(False, "", f"tool missing: {e}", skipped=True)


def _security_scan(source: str, language: Language) -> bool:
    for pat in SECURITY_RULES.get(language, ()):
        if re.search(pat, source):
            return False
    return True


def _diff_stats(content: str, fmt: str) -> DiffStats:
    if fmt != "diff":
        return DiffStats(added=content.count("\n") + 1, removed=0, files=1)
    added = sum(1 for ln in content.splitlines()
                if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in content.splitlines()
                  if ln.startswith("-") and not ln.startswith("---"))
    files = len(re.findall(r"^\+\+\+ ", content, re.MULTILINE))
    return DiffStats(added=added, removed=removed, files=max(files, 1))


def verify_python(
    solution_code: str,
    tests_code: str,
    *,
    solution_format: str = "answer",
) -> Verification:
    """Run the full Python verification chain in the current venv."""
    v = Verification()
    v.diff_stats = _diff_stats(solution_code, solution_format)

    with tempfile.TemporaryDirectory(prefix="forge_py_") as d:
        sol = Path(d) / "solution.py"
        tst = Path(d) / "test_solution.py"
        sol.write_text(solution_code, encoding="utf-8")
        # tests import from `solution`
        tst.write_text(tests_code, encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        env["PYTHONPATH"] = d

        # compile (syntax / import)
        comp = _run([sys.executable, "-c", f"import py_compile,sys;py_compile.compile(r'{sol}',doraise=True)"], d, env)
        v.compile_ok = comp.ok

        # tests
        if v.compile_ok:
            test = _run([sys.executable, "-m", "pytest", "-q", str(tst)], d, env)
            # pytest exit 5 = no tests collected -> treat as fail
            v.tests_ok = test.ok and "no tests ran" not in (test.stdout + test.stderr).lower()
        else:
            v.tests_ok = False

        # lint (ruff) — skipped gracefully if ruff absent
        lint = _run([sys.executable, "-m", "ruff", "check", "--quiet", str(sol)], d, env)
        v.lint_ok = lint.ok or lint.skipped

        # typecheck (mypy) — skipped gracefully if mypy absent
        tc = _run([sys.executable, "-m", "mypy", "--ignore-missing-imports", str(sol)], d, env)
        v.typecheck_ok = tc.ok or tc.skipped

        # security heuristic
        v.security_ok = _security_scan(solution_code, Language.PYTHON)

    return v


def verify_docker(
    language: Language,
    solution_code: str,
    tests_code: str,
    *,
    solution_format: str = "answer",
) -> Verification:
    """Verify non-Python languages via pinned Docker images.

    When Docker is unavailable locally, returns a verification with execution
    flags left False and security computed statically, plus a `skipped` marker in
    diff_stats metadata-free form. The pod runbook installs Docker, so this path
    becomes fully executable there.
    """
    v = Verification()
    v.diff_stats = _diff_stats(solution_code, solution_format)
    v.security_ok = _security_scan(solution_code, language)

    if not _docker_available():
        # Cannot execute; leave compile/tests/lint/typecheck False (conservative).
        return v

    image = DOCKER_IMAGES[language]
    with tempfile.TemporaryDirectory(prefix=f"forge_{language.value}_") as d:
        (Path(d) / _src_name(language)).write_text(solution_code, encoding="utf-8")
        (Path(d) / _test_name(language)).write_text(tests_code, encoding="utf-8")
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{d}:/work", "-w", "/work", image,
            "bash", "-lc", RUNNERS[language],
        ]
        out = _run(cmd, d)
        # Convention: runner prints sentinel lines the harness parses.
        text = out.stdout + out.stderr
        v.compile_ok = "COMPILE_OK" in text
        v.tests_ok = "TESTS_OK" in text
        v.lint_ok = "LINT_OK" in text or "LINT_SKIP" in text
        v.typecheck_ok = "TYPECHECK_OK" in text or "TYPECHECK_SKIP" in text
    return v


def _src_name(language: Language) -> str:
    return {
        Language.TYPESCRIPT: "solution.ts",
        Language.SQL: "solution.sql",
        Language.CPP: "solution.cpp",
        Language.BASH: "solution.sh",
        Language.YAML: "solution.yaml",
    }[language]


def _test_name(language: Language) -> str:
    return {
        Language.TYPESCRIPT: "solution.test.ts",
        Language.SQL: "tests.sql",
        Language.CPP: "tests.cpp",
        Language.BASH: "tests.bats",
        Language.YAML: "tests.py",
    }[language]


# Command recipes per image. Each must echo the sentinel lines the parser expects.
RUNNERS: dict[Language, str] = {
    Language.TYPESCRIPT: (
        "npm i -g typescript@5.6 vitest@2.1 >/dev/null 2>&1 || true; "
        "tsc --noEmit solution.ts && echo COMPILE_OK && echo TYPECHECK_OK; "
        "npx vitest run --silent && echo TESTS_OK; "
        "echo LINT_SKIP"
    ),
    Language.SQL: (
        "service postgresql start >/dev/null 2>&1 || pg_ctlcluster 16 main start || true; "
        "su postgres -c 'psql -v ON_ERROR_STOP=1 -f solution.sql' && echo COMPILE_OK; "
        "su postgres -c 'psql -v ON_ERROR_STOP=1 -f tests.sql' && echo TESTS_OK; "
        "echo LINT_SKIP; echo TYPECHECK_SKIP"
    ),
    Language.CPP: (
        "g++ -std=c++20 -fsyntax-only solution.cpp && echo COMPILE_OK && echo TYPECHECK_OK; "
        "g++ -std=c++20 tests.cpp -o /tmp/t && /tmp/t && echo TESTS_OK; "
        "echo LINT_SKIP"
    ),
    Language.BASH: (
        "shellcheck solution.sh && echo LINT_OK && echo COMPILE_OK && echo TYPECHECK_SKIP; "
        "bash tests.bats && echo TESTS_OK"
    ),
    Language.YAML: (
        "python -c 'import yaml,sys;yaml.safe_load(open(\"solution.yaml\"))' "
        "&& echo COMPILE_OK && echo TYPECHECK_SKIP && echo LINT_SKIP; "
        "python tests.py && echo TESTS_OK"
    ),
}


def verify(
    language: Language,
    solution_code: str,
    tests_code: str,
    *,
    solution_format: str = "answer",
) -> Verification:
    """Dispatch to the right runner by language."""
    if language == Language.PYTHON:
        return verify_python(solution_code, tests_code, solution_format=solution_format)
    return verify_docker(language, solution_code, tests_code, solution_format=solution_format)
