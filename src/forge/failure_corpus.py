"""
Layer C — Failure Recovery seed corpus.

Each episode is a recovery trajectory:
    task -> incorrect attempt -> failing test output -> ROOT CAUSE ->
    CORRECTED PATCH -> passing tests.

The corrected code (`exec_code`) is genuinely executed by the judge against the
hidden tests, so the recorded verification is real. One episode per error class
in the spec taxonomy; the generator scales these toward the ~100-150 target.
"""

from __future__ import annotations

from src.forge.schema import (
    BugClass,
    Difficulty,
    EditScope,
    Episode,
    Language,
    Layer,
    Solution,
    TaskPrompt,
    TaskType,
)

CANARY = "NULLXES-FORGE-FAIL-2D9C"


def _fail(**kw) -> Episode:
    kw.setdefault("layer", Layer.FAILURE)
    kw.setdefault("task_type", TaskType.REPAIR_TRAJECTORY)
    kw.setdefault("language", Language.PYTHON)
    kw.setdefault("edit_scope", EditScope.SINGLE_FUNC)
    kw.setdefault("difficulty", Difficulty.MEDIUM)
    kw.setdefault("metadata", {})
    kw["metadata"].setdefault("source", "hand_authored_failure")
    kw["metadata"].setdefault("canary", CANARY)
    return Episode(**kw)


def _answer(root_cause: str, diff: str, tests_note: str) -> str:
    return (
        f"## ROOT CAUSE\n{root_cause}\n\n"
        f"## CORRECTED PATCH\n```diff\n{diff}\n```\n\n"
        f"## TESTS\n{tests_note}\n"
    )


def failure_episodes() -> list[Episode]:
    eps: list[Episode] = []

    # off_by_one
    eps.append(_fail(
        id="cf-fail-py-0001", bug_class=BugClass.OFF_BY_ONE,
        prompt=TaskPrompt(
            task="`countdown(n)` should return [n, n-1, ..., 1]. First attempt drops 1. Diagnose and fix.",
            repo_context=(
                "Attempt:\n```python\ndef countdown(n: int) -> list[int]:\n"
                "    return list(range(n, 1, -1))\n```\n"
                "Failing: countdown(3) -> [3, 2], expected [3, 2, 1]."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "`range(n, 1, -1)` stops before 1, so the final element is excluded (off-by-one on the stop bound).",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def countdown(n: int) -> list[int]:\n"
            "-    return list(range(n, 1, -1))\n"
            "+    return list(range(n, 0, -1))\n",
            "Covers n=3 and n=1.")),
        exec_code="def countdown(n: int) -> list[int]:\n    return list(range(n, 0, -1))\n",
        tests=(
            "from solution import countdown\n\n"
            "def test_three():\n    assert countdown(3) == [3, 2, 1]\n\n"
            "def test_one():\n    assert countdown(1) == [1]\n"
        ),
    ))

    # null_handling
    eps.append(_fail(
        id="cf-fail-py-0002", bug_class=BugClass.NULL_HANDLING,
        prompt=TaskPrompt(
            task="`deep_get(d, keys)` should return None when any key is missing. First attempt raises KeyError.",
            repo_context=(
                "Attempt:\n```python\ndef deep_get(d, keys):\n"
                "    for k in keys:\n        d = d[k]\n    return d\n```\n"
                "Failing: deep_get({'a': {}}, ['a', 'b']) raises KeyError."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "Direct subscripting assumes every key exists; a missing intermediate key raises KeyError instead of returning None.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def deep_get(d, keys):\n"
            "     for k in keys:\n"
            "-        d = d[k]\n"
            "+        if not isinstance(d, dict) or k not in d:\n"
            "+            return None\n"
            "+        d = d[k]\n"
            "     return d\n",
            "Covers present path, missing leaf, and missing branch.")),
        exec_code=(
            "def deep_get(d, keys):\n"
            "    for k in keys:\n"
            "        if not isinstance(d, dict) or k not in d:\n"
            "            return None\n"
            "        d = d[k]\n"
            "    return d\n"
        ),
        tests=(
            "from solution import deep_get\n\n"
            "def test_present():\n    assert deep_get({'a': {'b': 1}}, ['a', 'b']) == 1\n\n"
            "def test_missing():\n    assert deep_get({'a': {}}, ['a', 'b']) is None\n"
        ),
    ))

    # race_condition (modeled deterministically: non-idempotent init)
    eps.append(_fail(
        id="cf-fail-py-0003", bug_class=BugClass.RACE_CONDITION,
        difficulty=Difficulty.HARD,
        prompt=TaskPrompt(
            task="`get_or_create(cache, key, factory)` must return a stable value per key. The attempt overwrites on every call (lost-update under concurrency).",
            repo_context=(
                "Attempt:\n```python\ndef get_or_create(cache, key, factory):\n"
                "    cache[key] = factory()\n    return cache[key]\n```\n"
                "Failing: a second call with a new factory replaces the first value."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "The attempt always writes before checking, so concurrent or repeated calls clobber an existing value (lost update). Use an atomic check-and-set via setdefault.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def get_or_create(cache, key, factory):\n"
            "-    cache[key] = factory()\n"
            "-    return cache[key]\n"
            "+    return cache.setdefault(key, factory())\n",
            "Covers first creation and stability on a second call.")),
        exec_code=(
            "def get_or_create(cache, key, factory):\n"
            "    return cache.setdefault(key, factory())\n"
        ),
        tests=(
            "from solution import get_or_create\n\n"
            "def test_create():\n    c = {}\n    assert get_or_create(c, 'k', lambda: 1) == 1\n\n"
            "def test_stable():\n    c = {}\n    get_or_create(c, 'k', lambda: 1)\n"
            "    assert get_or_create(c, 'k', lambda: 2) == 1\n"
        ),
    ))

    # sql_injection
    eps.append(_fail(
        id="cf-fail-py-0004", bug_class=BugClass.SQL_INJECTION,
        difficulty=Difficulty.HARD, edit_scope=EditScope.SINGLE_FILE,
        prompt=TaskPrompt(
            task="`find_user(conn, name)` builds SQL via string formatting and is injectable. Parameterize it.",
            repo_context=(
                "Attempt:\n```python\ndef find_user(conn, name):\n"
                "    cur = conn.cursor()\n"
                "    cur.execute(\"SELECT id FROM users WHERE name = '\" + name + \"'\")\n"
                "    row = cur.fetchone()\n    return row[0] if row else None\n```\n"
                "Failing: name=\"x'; DROP TABLE users;--\" corrupts the query."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "User input is concatenated into the SQL string, allowing injection. Use a parameterized query so the driver binds the value safely.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            "     cur = conn.cursor()\n"
            "-    cur.execute(\"SELECT id FROM users WHERE name = '\" + name + \"'\")\n"
            "+    cur.execute(\"SELECT id FROM users WHERE name = ?\", (name,))\n"
            "     row = cur.fetchone()\n",
            "Covers a normal lookup and a malicious name that must not drop the table.")),
        exec_code=(
            "def find_user(conn, name):\n"
            "    cur = conn.cursor()\n"
            "    cur.execute(\"SELECT id FROM users WHERE name = ?\", (name,))\n"
            "    row = cur.fetchone()\n"
            "    return row[0] if row else None\n"
        ),
        tests=(
            "import sqlite3\nfrom solution import find_user\n\n"
            "def _db():\n"
            "    c = sqlite3.connect(':memory:')\n"
            "    c.execute('CREATE TABLE users (id INTEGER, name TEXT)')\n"
            "    c.execute(\"INSERT INTO users VALUES (1, 'alice')\")\n"
            "    return c\n\n"
            "def test_lookup():\n    assert find_user(_db(), 'alice') == 1\n\n"
            "def test_injection_safe():\n"
            "    c = _db()\n"
            "    assert find_user(c, \"x'; DROP TABLE users;--\") is None\n"
            "    assert c.execute('SELECT COUNT(*) FROM users').fetchone()[0] == 1\n"
        ),
    ))

    # incompatible_api
    eps.append(_fail(
        id="cf-fail-py-0005", bug_class=BugClass.INCOMPATIBLE_API,
        prompt=TaskPrompt(
            task="`has_key_check(d, k)` uses a removed Python 2 API. Port it to Python 3.",
            repo_context=(
                "Attempt:\n```python\ndef has_key_check(d, k):\n"
                "    return d.has_key(k)\n```\n"
                "Failing: AttributeError: 'dict' object has no attribute 'has_key'."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "`dict.has_key` was removed in Python 3; membership must use the `in` operator.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def has_key_check(d, k):\n"
            "-    return d.has_key(k)\n"
            "+    return k in d\n",
            "Covers present and absent keys.")),
        exec_code="def has_key_check(d, k):\n    return k in d\n",
        tests=(
            "from solution import has_key_check\n\n"
            "def test_present():\n    assert has_key_check({'a': 1}, 'a')\n\n"
            "def test_absent():\n    assert not has_key_check({'a': 1}, 'b')\n"
        ),
    ))

    # broken_typing
    eps.append(_fail(
        id="cf-fail-py-0006", bug_class=BugClass.BROKEN_TYPING,
        prompt=TaskPrompt(
            task="`to_cents(amount)` must return an int number of cents. The attempt returns a float, breaking downstream equality.",
            repo_context=(
                "Attempt:\n```python\ndef to_cents(amount: float) -> int:\n"
                "    return amount * 100\n```\n"
                "Failing: to_cents(1.99) -> 198.99999999999997, not 199."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "Float multiplication leaves a float with rounding error and violates the int return type; round and cast to int.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def to_cents(amount: float) -> int:\n"
            "-    return amount * 100\n"
            "+    return int(round(amount * 100))\n",
            "Covers a value with float-rounding risk and a whole number.")),
        exec_code="def to_cents(amount: float) -> int:\n    return int(round(amount * 100))\n",
        tests=(
            "from solution import to_cents\n\n"
            "def test_rounding():\n    assert to_cents(1.99) == 199\n\n"
            "def test_whole():\n    assert to_cents(2.0) == 200\n"
        ),
    ))

    # bad_migration (modeled as a reversible in-memory schema migration)
    eps.append(_fail(
        id="cf-fail-py-0007", bug_class=BugClass.BAD_MIGRATION,
        edit_scope=EditScope.SINGLE_FILE,
        prompt=TaskPrompt(
            task="`migrate(rows)` renames field 'fullname' to 'name' but the attempt drops rows missing the old field. It must keep all rows.",
            repo_context=(
                "Attempt:\n```python\ndef migrate(rows):\n"
                "    return [{'name': r['fullname']} for r in rows]\n```\n"
                "Failing: a row without 'fullname' raises KeyError and the migration aborts."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "The migration assumes every row has the old field, so legacy/partial rows raise KeyError and are lost. Default the old value and preserve already-migrated rows.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def migrate(rows):\n"
            "-    return [{'name': r['fullname']} for r in rows]\n"
            "+    out = []\n"
            "+    for r in rows:\n"
            "+        name = r.get('fullname', r.get('name', ''))\n"
            "+        out.append({'name': name})\n"
            "+    return out\n",
            "Covers old-field rows, already-migrated rows, and empty rows.")),
        exec_code=(
            "def migrate(rows):\n"
            "    out = []\n"
            "    for r in rows:\n"
            "        name = r.get('fullname', r.get('name', ''))\n"
            "        out.append({'name': name})\n"
            "    return out\n"
        ),
        tests=(
            "from solution import migrate\n\n"
            "def test_rename():\n    assert migrate([{'fullname': 'A'}]) == [{'name': 'A'}]\n\n"
            "def test_keep_all():\n"
            "    assert migrate([{'name': 'B'}, {}]) == [{'name': 'B'}, {'name': ''}]\n"
        ),
    ))

    # flaky_pass (nondeterministic ordering)
    eps.append(_fail(
        id="cf-fail-py-0008", bug_class=BugClass.FLAKY_PASS,
        prompt=TaskPrompt(
            task="`top_tags(counts)` should return tag names ordered by count desc, ties broken alphabetically. The attempt relies on dict/set ordering and is flaky.",
            repo_context=(
                "Attempt:\n```python\ndef top_tags(counts):\n"
                "    return [t for t in set(counts)][:2]\n```\n"
                "Failing: order varies between runs; ties not handled."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "Iterating a set has no defined order, so results are nondeterministic. Sort by (-count, name) for a stable, specified order.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def top_tags(counts):\n"
            "-    return [t for t in set(counts)][:2]\n"
            "+    ordered = sorted(counts, key=lambda t: (-counts[t], t))\n"
            "+    return ordered[:2]\n",
            "Covers strict ordering and an alphabetical tie-break.")),
        exec_code=(
            "def top_tags(counts):\n"
            "    ordered = sorted(counts, key=lambda t: (-counts[t], t))\n"
            "    return ordered[:2]\n"
        ),
        tests=(
            "from solution import top_tags\n\n"
            "def test_order():\n    assert top_tags({'a': 5, 'b': 2, 'c': 9}) == ['c', 'a']\n\n"
            "def test_tie():\n    assert top_tags({'x': 1, 'a': 1, 'm': 1}) == ['a', 'm']\n"
        ),
    ))

    # public_contract_break
    eps.append(_fail(
        id="cf-fail-py-0009", bug_class=BugClass.PUBLIC_CONTRACT_BREAK,
        prompt=TaskPrompt(
            task="`split_name(full)` has a documented contract: return (first, last). A refactor made it return a dict, breaking callers. Restore the contract.",
            repo_context=(
                "Attempt:\n```python\ndef split_name(full: str):\n"
                "    first, _, last = full.partition(' ')\n"
                "    return {'first': first, 'last': last}\n```\n"
                "Failing: callers unpack `first, last = split_name(x)` and now get dict keys."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "The public contract returns a (first, last) tuple; the refactor changed the return type, breaking unpacking at every call site. Restore the tuple.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def split_name(full: str):\n"
            "     first, _, last = full.partition(' ')\n"
            "-    return {'first': first, 'last': last}\n"
            "+    return first, last\n",
            "Covers a two-part name and a single token.")),
        exec_code=(
            "def split_name(full: str):\n"
            "    first, _, last = full.partition(' ')\n"
            "    return first, last\n"
        ),
        tests=(
            "from solution import split_name\n\n"
            "def test_pair():\n    first, last = split_name('Ada Lovelace')\n"
            "    assert (first, last) == ('Ada', 'Lovelace')\n\n"
            "def test_single():\n    first, last = split_name('Cher')\n"
            "    assert (first, last) == ('Cher', '')\n"
        ),
    ))

    # over_refactor
    eps.append(_fail(
        id="cf-fail-py-0010", bug_class=BugClass.OVER_REFACTOR,
        prompt=TaskPrompt(
            task="A 'clever' one-liner refactor of `first_even(nums)` broke the empty/no-match case (should return None). Restore correct behavior minimally.",
            repo_context=(
                "Attempt:\n```python\ndef first_even(nums):\n"
                "    return [n for n in nums if n % 2 == 0][0]\n```\n"
                "Failing: first_even([1, 3]) raises IndexError instead of returning None."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "The comprehension+`[0]` assumes a match exists; with no even number it indexes an empty list and raises. Use next() with a default.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def first_even(nums):\n"
            "-    return [n for n in nums if n % 2 == 0][0]\n"
            "+    return next((n for n in nums if n % 2 == 0), None)\n",
            "Covers a match, no match, and empty input.")),
        exec_code=(
            "def first_even(nums):\n"
            "    return next((n for n in nums if n % 2 == 0), None)\n"
        ),
        tests=(
            "from solution import first_even\n\n"
            "def test_match():\n    assert first_even([1, 2, 3]) == 2\n\n"
            "def test_none():\n    assert first_even([1, 3]) is None\n\n"
            "def test_empty():\n    assert first_even([]) is None\n"
        ),
    ))

    # hallucinated_package
    eps.append(_fail(
        id="cf-fail-py-0011", bug_class=BugClass.HALLUCINATED_PACKAGE,
        prompt=TaskPrompt(
            task="`load_config(text)` imports a non-existent package `fastjson`. Replace it with the standard library; parse a JSON string to a dict.",
            repo_context=(
                "Attempt:\n```python\nimport fastjson\n\n"
                "def load_config(text: str) -> dict:\n"
                "    return fastjson.parse(text)\n```\n"
                "Failing: ModuleNotFoundError: No module named 'fastjson'."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "`fastjson` does not exist; the standard library `json` module provides the needed parser.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            "-import fastjson\n"
            "+import json\n\n"
            " def load_config(text: str) -> dict:\n"
            "-    return fastjson.parse(text)\n"
            "+    return json.loads(text)\n",
            "Covers a simple object and a nested object.")),
        exec_code=(
            "import json\n\n"
            "def load_config(text: str) -> dict:\n"
            "    return json.loads(text)\n"
        ),
        tests=(
            "from solution import load_config\n\n"
            "def test_flat():\n    assert load_config('{\"a\": 1}') == {'a': 1}\n\n"
            "def test_nested():\n    assert load_config('{\"a\": {\"b\": 2}}') == {'a': {'b': 2}}\n"
        ),
    ))

    # symptom_not_cause
    eps.append(_fail(
        id="cf-fail-py-0012", bug_class=BugClass.SYMPTOM_NOT_CAUSE,
        prompt=TaskPrompt(
            task="`percent(part, whole)` crashed on whole=0, so the attempt wraps it in a broad try/except returning 0 — hiding real errors. Fix the actual cause.",
            repo_context=(
                "Attempt:\n```python\ndef percent(part, whole):\n"
                "    try:\n        return part / whole * 100\n"
                "    except Exception:\n        return 0\n```\n"
                "Failing: passing a string for `part` silently returns 0 instead of raising."
            ),
        ),
        solution=Solution(format="answer", content=_answer(
            "The broad except swallows every error (including type errors), treating the symptom (a crash) not the cause (division by zero). Guard the zero case explicitly and let real errors surface.",
            "--- a/solution.py\n+++ b/solution.py\n@@\n"
            " def percent(part, whole):\n"
            "-    try:\n"
            "-        return part / whole * 100\n"
            "-    except Exception:\n"
            "-        return 0\n"
            "+    if whole == 0:\n"
            "+        return 0.0\n"
            "+    return part / whole * 100\n",
            "Covers a normal ratio, the zero-denominator guard, and a type error that must propagate.")),
        exec_code=(
            "def percent(part, whole):\n"
            "    if whole == 0:\n"
            "        return 0.0\n"
            "    return part / whole * 100\n"
        ),
        tests=(
            "import pytest\nfrom solution import percent\n\n"
            "def test_ratio():\n    assert percent(1, 4) == 25\n\n"
            "def test_zero():\n    assert percent(1, 0) == 0.0\n\n"
            "def test_type_error_propagates():\n"
            "    with pytest.raises(TypeError):\n        percent('x', 4)\n"
        ),
    ))

    return eps


if __name__ == "__main__":
    print(f"failure seed episodes: {len(failure_episodes())}")
