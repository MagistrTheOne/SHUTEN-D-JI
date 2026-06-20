"""
Layer A — Verified Gold seed corpus (hand-authored, genuinely executable).

Every Python episode here carries a real reference implementation (`exec_code`),
real hidden tests (`tests`), and a Constitution-formatted assistant answer
(`solution.content`). They are run through the judge by
`scripts/forge_build_gold.py`, which records the *real* verification result — no
fabricated pass marks.

This is the executable seed; the synthetic generator (Layer B) and additional
hand-authoring scale it toward the ~150-200 target on the pod.
"""

from __future__ import annotations

from src.forge.schema import (
    Difficulty,
    EditScope,
    Episode,
    Language,
    Layer,
    Solution,
    TaskPrompt,
    TaskType,
)

CANARY = "NULLXES-FORGE-GOLD-7F3A"


def _ep(**kw) -> Episode:
    kw.setdefault("metadata", {})
    kw["metadata"].setdefault("source", "hand_authored_gold")
    kw["metadata"].setdefault("canary", CANARY)
    return Episode(**kw)


def gold_episodes() -> list[Episode]:
    eps: list[Episode] = []

    # 1 — function_impl: clamp
    eps.append(_ep(
        id="cf-gold-py-0001",
        layer=Layer.GOLD, task_type=TaskType.FUNCTION_IMPL, language=Language.PYTHON,
        difficulty=Difficulty.EASY, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Implement `clamp(x, lo, hi)` returning x bounded to the closed interval [lo, hi].",
            constraints=["no new third-party dependency", "O(1) time"],
        ),
        solution=Solution(format="answer", content=(
            "## ASSUMPTIONS\n- lo <= hi; inputs are comparable numerics.\n\n"
            "## IMPLEMENTATION\n```python\n"
            "def clamp(x: float, lo: float, hi: float) -> float:\n"
            "    return max(lo, min(x, hi))\n```\n\n"
            "## TESTS\nCovers values below, inside, and above the range, plus boundary equality.\n\n"
            "## COMPLEXITY\nO(1) time, O(1) space.\n"
        )),
        exec_code="def clamp(x: float, lo: float, hi: float) -> float:\n    return max(lo, min(x, hi))\n",
        tests=(
            "from solution import clamp\n\n"
            "def test_inside():\n    assert clamp(5, 0, 10) == 5\n\n"
            "def test_below():\n    assert clamp(-3, 0, 10) == 0\n\n"
            "def test_above():\n    assert clamp(42, 0, 10) == 10\n\n"
            "def test_boundary():\n    assert clamp(0, 0, 10) == 0 and clamp(10, 0, 10) == 10\n"
        ),
    ))

    # 2 — function_impl: run-length encode
    eps.append(_ep(
        id="cf-gold-py-0002",
        layer=Layer.GOLD, task_type=TaskType.FUNCTION_IMPL, language=Language.PYTHON,
        difficulty=Difficulty.EASY, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Implement `rle(s)` that run-length-encodes a string as a list of (char, count) tuples in order.",
            constraints=["no new third-party dependency"],
        ),
        solution=Solution(format="answer", content=(
            "## ASSUMPTIONS\n- Empty string returns an empty list; order is preserved.\n\n"
            "## IMPLEMENTATION\n```python\n"
            "def rle(s: str) -> list[tuple[str, int]]:\n"
            "    out: list[tuple[str, int]] = []\n"
            "    for ch in s:\n"
            "        if out and out[-1][0] == ch:\n"
            "            out[-1] = (ch, out[-1][1] + 1)\n"
            "        else:\n"
            "            out.append((ch, 1))\n"
            "    return out\n```\n\n"
            "## TESTS\nCovers empty input, single run, alternating chars, and repeated runs.\n\n"
            "## COMPLEXITY\nO(n) time, O(k) space for k distinct runs.\n"
        )),
        exec_code=(
            "def rle(s: str) -> list[tuple[str, int]]:\n"
            "    out: list[tuple[str, int]] = []\n"
            "    for ch in s:\n"
            "        if out and out[-1][0] == ch:\n"
            "            out[-1] = (ch, out[-1][1] + 1)\n"
            "        else:\n"
            "            out.append((ch, 1))\n"
            "    return out\n"
        ),
        tests=(
            "from solution import rle\n\n"
            "def test_empty():\n    assert rle('') == []\n\n"
            "def test_single():\n    assert rle('aaa') == [('a', 3)]\n\n"
            "def test_mixed():\n    assert rle('aabbbc') == [('a', 2), ('b', 3), ('c', 1)]\n"
        ),
    ))

    # 3 — function_impl: merge intervals
    eps.append(_ep(
        id="cf-gold-py-0003",
        layer=Layer.GOLD, task_type=TaskType.FUNCTION_IMPL, language=Language.PYTHON,
        difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Implement `merge_intervals(intervals)` merging overlapping [start, end] pairs, returned sorted by start.",
            constraints=["no new third-party dependency", "treat touching intervals (end == next start) as overlapping"],
        ),
        solution=Solution(format="answer", content=(
            "## ASSUMPTIONS\n- Each interval is [start, end] with start <= end; touching intervals merge.\n\n"
            "## IMPLEMENTATION\n```python\n"
            "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    if not intervals:\n"
            "        return []\n"
            "    ordered = sorted(intervals, key=lambda iv: iv[0])\n"
            "    merged = [list(ordered[0])]\n"
            "    for start, end in ordered[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n```\n\n"
            "## TESTS\nCovers empty, disjoint, overlapping, touching, and unsorted inputs.\n\n"
            "## COMPLEXITY\nO(n log n) time from the sort, O(n) space.\n"
        )),
        exec_code=(
            "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
            "    if not intervals:\n"
            "        return []\n"
            "    ordered = sorted(intervals, key=lambda iv: iv[0])\n"
            "    merged = [list(ordered[0])]\n"
            "    for start, end in ordered[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
        ),
        tests=(
            "from solution import merge_intervals\n\n"
            "def test_empty():\n    assert merge_intervals([]) == []\n\n"
            "def test_overlap():\n    assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]\n\n"
            "def test_touching():\n    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]\n\n"
            "def test_unsorted():\n    assert merge_intervals([[8, 10], [1, 3]]) == [[1, 3], [8, 10]]\n"
        ),
    ))

    # 4 — function_impl: chunked
    eps.append(_ep(
        id="cf-gold-py-0004",
        layer=Layer.GOLD, task_type=TaskType.FUNCTION_IMPL, language=Language.PYTHON,
        difficulty=Difficulty.EASY, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Implement `chunked(seq, n)` yielding consecutive lists of length n (last may be shorter). Raise ValueError if n <= 0.",
            constraints=["no new third-party dependency"],
        ),
        solution=Solution(format="answer", content=(
            "## ASSUMPTIONS\n- seq is indexable; n must be a positive integer.\n\n"
            "## IMPLEMENTATION\n```python\n"
            "def chunked(seq: list, n: int) -> list[list]:\n"
            "    if n <= 0:\n"
            "        raise ValueError('n must be positive')\n"
            "    return [seq[i:i + n] for i in range(0, len(seq), n)]\n```\n\n"
            "## TESTS\nCovers exact division, remainder, empty input, and the n <= 0 error path.\n\n"
            "## COMPLEXITY\nO(n) time, O(n) space.\n"
        )),
        exec_code=(
            "def chunked(seq: list, n: int) -> list[list]:\n"
            "    if n <= 0:\n"
            "        raise ValueError('n must be positive')\n"
            "    return [seq[i:i + n] for i in range(0, len(seq), n)]\n"
        ),
        tests=(
            "import pytest\nfrom solution import chunked\n\n"
            "def test_exact():\n    assert chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n\n"
            "def test_remainder():\n    assert chunked([1, 2, 3], 2) == [[1, 2], [3]]\n\n"
            "def test_empty():\n    assert chunked([], 3) == []\n\n"
            "def test_bad_n():\n    \n    with pytest.raises(ValueError):\n        chunked([1], 0)\n"
        ),
    ))

    # 5 — function_impl: roman to int
    eps.append(_ep(
        id="cf-gold-py-0005",
        layer=Layer.GOLD, task_type=TaskType.FUNCTION_IMPL, language=Language.PYTHON,
        difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Implement `roman_to_int(s)` converting a valid uppercase Roman numeral to an integer.",
            constraints=["no new third-party dependency", "handle subtractive pairs (IV, IX, XL, XC, CD, CM)"],
        ),
        solution=Solution(format="answer", content=(
            "## ASSUMPTIONS\n- Input is a valid, uppercase Roman numeral.\n\n"
            "## IMPLEMENTATION\n```python\n"
            "def roman_to_int(s: str) -> int:\n"
            "    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    total = 0\n"
            "    prev = 0\n"
            "    for ch in reversed(s):\n"
            "        cur = vals[ch]\n"
            "        total += -cur if cur < prev else cur\n"
            "        prev = cur\n"
            "    return total\n```\n\n"
            "## TESTS\nCovers single symbols, subtractive pairs, and a composite numeral.\n\n"
            "## COMPLEXITY\nO(n) time, O(1) space.\n"
        )),
        exec_code=(
            "def roman_to_int(s: str) -> int:\n"
            "    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n"
            "    total = 0\n"
            "    prev = 0\n"
            "    for ch in reversed(s):\n"
            "        cur = vals[ch]\n"
            "        total += -cur if cur < prev else cur\n"
            "        prev = cur\n"
            "    return total\n"
        ),
        tests=(
            "from solution import roman_to_int\n\n"
            "def test_simple():\n    assert roman_to_int('III') == 3\n\n"
            "def test_subtractive():\n    assert roman_to_int('IV') == 4 and roman_to_int('IX') == 9\n\n"
            "def test_composite():\n    assert roman_to_int('MCMXciv'.upper()) == 1994\n"
        ),
    ))

    # 6 — function_impl: balanced brackets
    eps.append(_ep(
        id="cf-gold-py-0006",
        layer=Layer.GOLD, task_type=TaskType.FUNCTION_IMPL, language=Language.PYTHON,
        difficulty=Difficulty.EASY, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Implement `is_balanced(s)` returning True iff (), [], {} are correctly nested and matched.",
            constraints=["no new third-party dependency", "ignore non-bracket characters"],
        ),
        solution=Solution(format="answer", content=(
            "## ASSUMPTIONS\n- Only (), [], {} are brackets; other characters are ignored.\n\n"
            "## IMPLEMENTATION\n```python\n"
            "def is_balanced(s: str) -> bool:\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    stack: list[str] = []\n"
            "    for ch in s:\n"
            "        if ch in '([{':\n"
            "            stack.append(ch)\n"
            "        elif ch in pairs:\n"
            "            if not stack or stack.pop() != pairs[ch]:\n"
            "                return False\n"
            "    return not stack\n```\n\n"
            "## TESTS\nCovers balanced, unbalanced, wrong order, and embedded text.\n\n"
            "## COMPLEXITY\nO(n) time, O(n) space.\n"
        )),
        exec_code=(
            "def is_balanced(s: str) -> bool:\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    stack: list[str] = []\n"
            "    for ch in s:\n"
            "        if ch in '([{':\n"
            "            stack.append(ch)\n"
            "        elif ch in pairs:\n"
            "            if not stack or stack.pop() != pairs[ch]:\n"
            "                return False\n"
            "    return not stack\n"
        ),
        tests=(
            "from solution import is_balanced\n\n"
            "def test_ok():\n    assert is_balanced('a(b[c]{d})e')\n\n"
            "def test_unbalanced():\n    assert not is_balanced('(]')\n\n"
            "def test_unclosed():\n    assert not is_balanced('(((')\n"
        ),
    ))

    # 7 — function_impl: flatten
    eps.append(_ep(
        id="cf-gold-py-0007",
        layer=Layer.GOLD, task_type=TaskType.FUNCTION_IMPL, language=Language.PYTHON,
        difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Implement `flatten(nested)` that fully flattens arbitrarily nested lists into a single list, preserving order.",
            constraints=["no new third-party dependency", "treat only list as nestable; keep tuples/strings intact"],
        ),
        solution=Solution(format="answer", content=(
            "## ASSUMPTIONS\n- Only list instances are recursed; other types are leaves.\n\n"
            "## IMPLEMENTATION\n```python\n"
            "def flatten(nested: list) -> list:\n"
            "    out: list = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            out.extend(flatten(item))\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n```\n\n"
            "## TESTS\nCovers flat, deeply nested, empty sublists, and non-list leaves.\n\n"
            "## COMPLEXITY\nO(n) over total elements, O(d) recursion depth.\n"
        )),
        exec_code=(
            "def flatten(nested: list) -> list:\n"
            "    out: list = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            out.extend(flatten(item))\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
        tests=(
            "from solution import flatten\n\n"
            "def test_flat():\n    assert flatten([1, 2, 3]) == [1, 2, 3]\n\n"
            "def test_deep():\n    assert flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]\n\n"
            "def test_keep_tuple():\n    assert flatten([[(1, 2)], 'ab']) == [(1, 2), 'ab']\n"
        ),
    ))

    # 8 — test_gen: author tests for a given parser
    eps.append(_ep(
        id="cf-gold-py-0008",
        layer=Layer.GOLD, task_type=TaskType.TEST_GEN, language=Language.PYTHON,
        difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FILE,
        prompt=TaskPrompt(
            task="Write a focused pytest suite for `parse_kv(line)` which parses 'k1=v1;k2=v2' into a dict, trimming whitespace and ignoring empty segments.",
            repo_context=(
                "```python\n"
                "def parse_kv(line: str) -> dict[str, str]:\n"
                "    out = {}\n"
                "    for seg in line.split(';'):\n"
                "        seg = seg.strip()\n"
                "        if not seg:\n"
                "            continue\n"
                "        k, _, v = seg.partition('=')\n"
                "        out[k.strip()] = v.strip()\n"
                "    return out\n```"
            ),
            constraints=["tests only; do not modify the implementation"],
        ),
        solution=Solution(format="answer", content=(
            "## COVERAGE TARGETS\n- Single pair, multiple pairs, surrounding whitespace, empty segments, missing value.\n\n"
            "## TESTS\n```python\n"
            "from solution import parse_kv\n\n"
            "def test_single():\n    assert parse_kv('a=1') == {'a': '1'}\n\n"
            "def test_multi_and_trim():\n    assert parse_kv(' a = 1 ; b=2 ') == {'a': '1', 'b': '2'}\n\n"
            "def test_empty_segments():\n    assert parse_kv('a=1;;') == {'a': '1'}\n\n"
            "def test_missing_value():\n    assert parse_kv('a=') == {'a': ''}\n```\n\n"
            "## VERIFICATION\nAll four cases execute against the supplied implementation in the sandbox.\n"
        )),
        exec_code=(
            "def parse_kv(line: str) -> dict[str, str]:\n"
            "    out: dict[str, str] = {}\n"
            "    for seg in line.split(';'):\n"
            "        seg = seg.strip()\n"
            "        if not seg:\n"
            "            continue\n"
            "        k, _, v = seg.partition('=')\n"
            "        out[k.strip()] = v.strip()\n"
            "    return out\n"
        ),
        tests=(
            "from solution import parse_kv\n\n"
            "def test_single():\n    assert parse_kv('a=1') == {'a': '1'}\n\n"
            "def test_multi_and_trim():\n    assert parse_kv(' a = 1 ; b=2 ') == {'a': '1', 'b': '2'}\n\n"
            "def test_empty_segments():\n    assert parse_kv('a=1;;') == {'a': '1'}\n\n"
            "def test_missing_value():\n    assert parse_kv('a=') == {'a': ''}\n"
        ),
    ))

    # 9 — repo_bugfix (diff): off-by-one in range sum
    eps.append(_ep(
        id="cf-gold-py-0009",
        layer=Layer.GOLD, task_type=TaskType.REPO_BUGFIX, language=Language.PYTHON,
        bug_class=None, difficulty=Difficulty.EASY, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="`inclusive_sum(a, b)` should sum all integers from a to b inclusive but the last value is dropped. Fix it with a minimal patch.",
            repo_context=(
                "```python\n"
                "def inclusive_sum(a: int, b: int) -> int:\n"
                "    total = 0\n"
                "    for i in range(a, b):\n"
                "        total += i\n"
                "    return total\n```"
            ),
            constraints=["minimal patch", "keep the function signature"],
        ),
        solution=Solution(format="answer", content=(
            "## DIAGNOSIS\n`inclusive_sum(1, 5)` returns 10 instead of 15 — the upper bound is excluded.\n\n"
            "## ROOT CAUSE\n`range(a, b)` stops at b-1, so b is never added (off-by-one on the inclusive bound).\n\n"
            "## PATCH\n```diff\n"
            "--- a/solution.py\n+++ b/solution.py\n"
            "@@\n"
            "     total = 0\n"
            "-    for i in range(a, b):\n"
            "+    for i in range(a, b + 1):\n"
            "         total += i\n"
            "     return total\n```\n\n"
            "## TESTS\nAdds inclusive-bound and single-element cases.\n\n"
            "## VERIFICATION\nSandbox runs pytest on the patched file; all cases pass.\n"
        )),
        exec_code=(
            "def inclusive_sum(a: int, b: int) -> int:\n"
            "    total = 0\n"
            "    for i in range(a, b + 1):\n"
            "        total += i\n"
            "    return total\n"
        ),
        tests=(
            "from solution import inclusive_sum\n\n"
            "def test_range():\n    assert inclusive_sum(1, 5) == 15\n\n"
            "def test_single():\n    assert inclusive_sum(3, 3) == 3\n"
        ),
    ))

    # 10 — repo_bugfix (diff): null/KeyError handling
    eps.append(_ep(
        id="cf-gold-py-0010",
        layer=Layer.GOLD, task_type=TaskType.REPO_BUGFIX, language=Language.PYTHON,
        bug_class=None, difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="`get_price(cart, item)` raises KeyError for unknown items; it should return 0.0 instead. Fix minimally.",
            repo_context=(
                "```python\n"
                "def get_price(cart: dict[str, float], item: str) -> float:\n"
                "    return cart[item]\n```"
            ),
            constraints=["minimal patch", "do not change the signature"],
        ),
        solution=Solution(format="answer", content=(
            "## DIAGNOSIS\nLooking up a missing key raises KeyError instead of returning a default.\n\n"
            "## ROOT CAUSE\nDirect subscript `cart[item]` assumes the key exists; unknown items are unhandled.\n\n"
            "## PATCH\n```diff\n"
            "--- a/solution.py\n+++ b/solution.py\n"
            "@@\n"
            " def get_price(cart: dict[str, float], item: str) -> float:\n"
            "-    return cart[item]\n"
            "+    return cart.get(item, 0.0)\n```\n\n"
            "## TESTS\nCovers present and absent keys.\n\n"
            "## VERIFICATION\nSandbox runs pytest on the patched file; both cases pass.\n"
        )),
        exec_code=(
            "def get_price(cart: dict[str, float], item: str) -> float:\n"
            "    return cart.get(item, 0.0)\n"
        ),
        tests=(
            "from solution import get_price\n\n"
            "def test_present():\n    assert get_price({'a': 2.5}, 'a') == 2.5\n\n"
            "def test_absent():\n    assert get_price({'a': 2.5}, 'b') == 0.0\n"
        ),
    ))

    # 11 — refactor (diff): no behavior change
    eps.append(_ep(
        id="cf-gold-py-0011",
        layer=Layer.GOLD, task_type=TaskType.REFACTOR, language=Language.PYTHON,
        difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Refactor `unique_sorted(items)` to be clearer and O(n log n) without changing behavior (returns sorted unique values).",
            repo_context=(
                "```python\n"
                "def unique_sorted(items: list[int]) -> list[int]:\n"
                "    result = []\n"
                "    for x in items:\n"
                "        if x not in result:\n"
                "            result.append(x)\n"
                "    result.sort()\n"
                "    return result\n```"
            ),
            constraints=["no behavior change", "no new third-party dependency"],
        ),
        solution=Solution(format="answer", content=(
            "## INVARIANTS\nOutput is the ascending list of distinct input values; duplicates removed.\n\n"
            "## PLAN\nReplace the O(n^2) membership loop with a set, then sort once.\n\n"
            "## PATCH\n```diff\n"
            "--- a/solution.py\n+++ b/solution.py\n"
            "@@\n"
            " def unique_sorted(items: list[int]) -> list[int]:\n"
            "-    result = []\n"
            "-    for x in items:\n"
            "-        if x not in result:\n"
            "-            result.append(x)\n"
            "-    result.sort()\n"
            "-    return result\n"
            "+    return sorted(set(items))\n```\n\n"
            "## TESTS\nReuses behavior tests: duplicates, ordering, empty.\n\n"
            "## VERIFICATION\nSandbox runs the unchanged behavior tests on the refactored file; all pass.\n"
        )),
        exec_code="def unique_sorted(items: list[int]) -> list[int]:\n    return sorted(set(items))\n",
        tests=(
            "from solution import unique_sorted\n\n"
            "def test_dups():\n    assert unique_sorted([3, 1, 2, 3, 1]) == [1, 2, 3]\n\n"
            "def test_empty():\n    assert unique_sorted([]) == []\n\n"
            "def test_sorted():\n    assert unique_sorted([5, 4, 4, 0]) == [0, 4, 5]\n"
        ),
    ))

    # 12 — performance (diff): O(n^2) -> O(n)
    eps.append(_ep(
        id="cf-gold-py-0012",
        layer=Layer.GOLD, task_type=TaskType.PERFORMANCE, language=Language.PYTHON,
        difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="`has_duplicate(nums)` is O(n^2). Make it O(n) average time while preserving behavior.",
            repo_context=(
                "```python\n"
                "def has_duplicate(nums: list[int]) -> bool:\n"
                "    for i in range(len(nums)):\n"
                "        for j in range(i + 1, len(nums)):\n"
                "            if nums[i] == nums[j]:\n"
                "                return True\n"
                "    return False\n```"
            ),
            constraints=["preserve observable behavior; only improve performance"],
        ),
        solution=Solution(format="answer", content=(
            "## BOTTLENECK\nThe nested loop compares every pair, giving O(n^2) comparisons.\n\n"
            "## OPTIONS\nTrack seen values in a set (O(n) average) or sort then scan (O(n log n)). The set wins.\n\n"
            "## PATCH\n```diff\n"
            "--- a/solution.py\n+++ b/solution.py\n"
            "@@\n"
            " def has_duplicate(nums: list[int]) -> bool:\n"
            "-    for i in range(len(nums)):\n"
            "-        for j in range(i + 1, len(nums)):\n"
            "-            if nums[i] == nums[j]:\n"
            "-                return True\n"
            "-    return False\n"
            "+    seen: set[int] = set()\n"
            "+    for x in nums:\n"
            "+        if x in seen:\n"
            "+            return True\n"
            "+        seen.add(x)\n"
            "+    return False\n```\n\n"
            "## BENCHMARK\nOn 10k distinct ints the set version avoids ~50M comparisons of the pairwise loop.\n\n"
            "## VERIFICATION\nSandbox runs behavior tests on the patched file; all pass.\n"
        )),
        exec_code=(
            "def has_duplicate(nums: list[int]) -> bool:\n"
            "    seen: set[int] = set()\n"
            "    for x in nums:\n"
            "        if x in seen:\n"
            "            return True\n"
            "        seen.add(x)\n"
            "    return False\n"
        ),
        tests=(
            "from solution import has_duplicate\n\n"
            "def test_dup():\n    assert has_duplicate([1, 2, 3, 2])\n\n"
            "def test_unique():\n    assert not has_duplicate([1, 2, 3])\n\n"
            "def test_empty():\n    assert not has_duplicate([])\n"
        ),
    ))

    # 13 — security (diff): path traversal guard
    eps.append(_ep(
        id="cf-gold-py-0013",
        layer=Layer.GOLD, task_type=TaskType.SECURITY, language=Language.PYTHON,
        bug_class=None, difficulty=Difficulty.HARD, edit_scope=EditScope.SINGLE_FILE,
        prompt=TaskPrompt(
            task="`safe_read(base, name)` must read files only inside `base`. Currently it allows '../' escapes. Fix it.",
            repo_context=(
                "```python\n"
                "import os\n\n"
                "def safe_read(base: str, name: str) -> str:\n"
                "    path = os.path.join(base, name)\n"
                "    with open(path, encoding='utf-8') as f:\n"
                "        return f.read()\n```"
            ),
            constraints=["no new third-party dependency", "raise ValueError on escape attempts"],
        ),
        solution=Solution(format="answer", content=(
            "## THREAT\nA caller passing '../secret' reads files outside the intended base directory (path traversal).\n\n"
            "## DIAGNOSIS\n`os.path.join` happily resolves '..' segments, so the final path can escape `base`.\n\n"
            "## PATCH\n```diff\n"
            "--- a/solution.py\n+++ b/solution.py\n"
            "@@\n"
            " import os\n\n"
            " def safe_read(base: str, name: str) -> str:\n"
            "-    path = os.path.join(base, name)\n"
            "+    root = os.path.realpath(base)\n"
            "+    path = os.path.realpath(os.path.join(root, name))\n"
            "+    if os.path.commonpath([root, path]) != root:\n"
            "+        raise ValueError('path escapes base directory')\n"
            "     with open(path, encoding='utf-8') as f:\n"
            "         return f.read()\n```\n\n"
            "## TESTS\nCovers a valid in-base read and a rejected '../' escape.\n\n"
            "## VERIFICATION\nSandbox runs pytest on the patched file; both cases pass.\n"
        )),
        exec_code=(
            "import os\n\n"
            "def safe_read(base: str, name: str) -> str:\n"
            "    root = os.path.realpath(base)\n"
            "    path = os.path.realpath(os.path.join(root, name))\n"
            "    if os.path.commonpath([root, path]) != root:\n"
            "        raise ValueError('path escapes base directory')\n"
            "    with open(path, encoding='utf-8') as f:\n"
            "        return f.read()\n"
        ),
        tests=(
            "import os\nimport pytest\nfrom solution import safe_read\n\n"
            "def test_valid(tmp_path):\n"
            "    (tmp_path / 'a.txt').write_text('hi', encoding='utf-8')\n"
            "    assert safe_read(str(tmp_path), 'a.txt') == 'hi'\n\n"
            "def test_escape(tmp_path):\n"
            "    with pytest.raises(ValueError):\n"
            "        safe_read(str(tmp_path), os.path.join('..', 'etc_passwd'))\n"
        ),
    ))

    # 14 — repair_trajectory: binary search off-by-one
    eps.append(_ep(
        id="cf-gold-py-0014",
        layer=Layer.GOLD, task_type=TaskType.REPAIR_TRAJECTORY, language=Language.PYTHON,
        bug_class=None, difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="A first attempt at `bsearch(arr, target)` misses the last element. Diagnose the failing run and provide the corrected, passing version.",
            repo_context=(
                "First attempt:\n```python\n"
                "def bsearch(arr: list[int], target: int) -> int:\n"
                "    lo, hi = 0, len(arr) - 1\n"
                "    while lo < hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        if arr[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid - 1\n"
                "    return -1\n```\n"
                "Failing test output:\n```\n"
                "test_last: assert bsearch([1, 3, 5], 5) == 2  ->  got -1\n```"
            ),
            constraints=["minimal change", "keep return contract (index or -1)"],
        ),
        solution=Solution(format="answer", content=(
            "## ROOT CAUSE\nThe loop condition `while lo < hi` exits when `lo == hi`, so the final single-element window is never inspected — the last index is skipped.\n\n"
            "## CORRECTED PATCH\n```diff\n"
            "--- a/solution.py\n+++ b/solution.py\n"
            "@@\n"
            "     lo, hi = 0, len(arr) - 1\n"
            "-    while lo < hi:\n"
            "+    while lo <= hi:\n"
            "         mid = (lo + hi) // 2\n```\n\n"
            "## TESTS\nCovers first, middle, last, and absent targets.\n"
        )),
        exec_code=(
            "def bsearch(arr: list[int], target: int) -> int:\n"
            "    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        if arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n"
        ),
        tests=(
            "from solution import bsearch\n\n"
            "def test_first():\n    assert bsearch([1, 3, 5], 1) == 0\n\n"
            "def test_last():\n    assert bsearch([1, 3, 5], 5) == 2\n\n"
            "def test_absent():\n    assert bsearch([1, 3, 5], 4) == -1\n"
        ),
    ))

    # 15 — function_impl: moving average
    eps.append(_ep(
        id="cf-gold-py-0015",
        layer=Layer.GOLD, task_type=TaskType.FUNCTION_IMPL, language=Language.PYTHON,
        difficulty=Difficulty.MEDIUM, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="Implement `moving_average(xs, k)` returning the list of length len(xs)-k+1 of k-window means. Raise ValueError if k<=0 or k>len(xs).",
            constraints=["no new third-party dependency", "use a running sum (single pass)"],
        ),
        solution=Solution(format="answer", content=(
            "## ASSUMPTIONS\n- xs is a list of numbers; 1 <= k <= len(xs).\n\n"
            "## IMPLEMENTATION\n```python\n"
            "def moving_average(xs: list[float], k: int) -> list[float]:\n"
            "    if k <= 0 or k > len(xs):\n"
            "        raise ValueError('k out of range')\n"
            "    window = sum(xs[:k])\n"
            "    out = [window / k]\n"
            "    for i in range(k, len(xs)):\n"
            "        window += xs[i] - xs[i - k]\n"
            "        out.append(window / k)\n"
            "    return out\n```\n\n"
            "## TESTS\nCovers k=1, k=len, a sliding window, and the error path.\n\n"
            "## COMPLEXITY\nO(n) time via running sum, O(n) output.\n"
        )),
        exec_code=(
            "def moving_average(xs: list[float], k: int) -> list[float]:\n"
            "    if k <= 0 or k > len(xs):\n"
            "        raise ValueError('k out of range')\n"
            "    window = sum(xs[:k])\n"
            "    out = [window / k]\n"
            "    for i in range(k, len(xs)):\n"
            "        window += xs[i] - xs[i - k]\n"
            "        out.append(window / k)\n"
            "    return out\n"
        ),
        tests=(
            "import pytest\nfrom solution import moving_average\n\n"
            "def test_k1():\n    assert moving_average([1, 2, 3], 1) == [1, 2, 3]\n\n"
            "def test_window():\n    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]\n\n"
            "def test_bad_k():\n    with pytest.raises(ValueError):\n        moving_average([1, 2], 5)\n"
        ),
    ))

    # 16 — repo_bugfix (diff): broken typing / wrong return
    eps.append(_ep(
        id="cf-gold-py-0016",
        layer=Layer.GOLD, task_type=TaskType.REPO_BUGFIX, language=Language.PYTHON,
        bug_class=None, difficulty=Difficulty.EASY, edit_scope=EditScope.SINGLE_FUNC,
        prompt=TaskPrompt(
            task="`average(xs)` should return a float mean but returns an int via integer division, and crashes on empty input. Fix both minimally; empty returns 0.0.",
            repo_context=(
                "```python\n"
                "def average(xs: list[int]) -> float:\n"
                "    return sum(xs) // len(xs)\n```"
            ),
            constraints=["minimal patch", "empty input returns 0.0"],
        ),
        solution=Solution(format="answer", content=(
            "## DIAGNOSIS\n`average([1, 2])` returns 1 (int floor) not 1.5, and `average([])` raises ZeroDivisionError.\n\n"
            "## ROOT CAUSE\nFloor division `//` truncates, and there is no guard for the empty list.\n\n"
            "## PATCH\n```diff\n"
            "--- a/solution.py\n+++ b/solution.py\n"
            "@@\n"
            " def average(xs: list[int]) -> float:\n"
            "-    return sum(xs) // len(xs)\n"
            "+    if not xs:\n"
            "+        return 0.0\n"
            "+    return sum(xs) / len(xs)\n```\n\n"
            "## TESTS\nCovers a normal mean and the empty-input guard.\n\n"
            "## VERIFICATION\nSandbox runs pytest on the patched file; both cases pass.\n"
        )),
        exec_code=(
            "def average(xs: list[int]) -> float:\n"
            "    if not xs:\n"
            "        return 0.0\n"
            "    return sum(xs) / len(xs)\n"
        ),
        tests=(
            "from solution import average\n\n"
            "def test_mean():\n    assert average([1, 2]) == 1.5\n\n"
            "def test_empty():\n    assert average([]) == 0.0\n"
        ),
    ))

    return eps


if __name__ == "__main__":
    print(f"gold seed episodes: {len(gold_episodes())}")
