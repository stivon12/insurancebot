import asyncio
import json

from pipeline.common import memo, memo_path, note_failure


def test_result_built_on_failed_lookup_is_not_saved(tmp_path):
    async def failing():
        note_failure("example.org: HTTP 429")
        return {"candidates": []}

    async def ok():
        return {"candidates": ["x"]}

    async def go():
        r1 = await memo(tmp_path, "k", {"a": 1}, failing)
        assert r1 == {"candidates": []}
        assert not memo_path(tmp_path, "k", {"a": 1}).exists()  # retried next time, not cached as "no match"
        r2 = await memo(tmp_path, "k", {"a": 1}, ok)
        assert r2 == {"candidates": ["x"]}
        assert json.loads(memo_path(tmp_path, "k", {"a": 1}).read_text()) == r2

    asyncio.run(go())


def test_failure_in_nested_task_reaches_outer_memo(tmp_path):
    async def inner():
        note_failure("timeout")
        return 1

    async def outer():
        await asyncio.gather(memo(tmp_path, "in", 1, inner))
        return 2

    async def go():
        await memo(tmp_path, "out", 1, outer)
        assert not memo_path(tmp_path, "in", 1).exists()
        assert not memo_path(tmp_path, "out", 1).exists()

    asyncio.run(go())
