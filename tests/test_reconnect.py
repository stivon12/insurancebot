"""A phone that drops its connection reconnects to the same sweep; one that never returns still gets its packet."""

import time

from fastapi.testclient import TestClient

import server.app as A


def _client(monkeypatch, tmp_path, built):
    monkeypatch.setenv("LCA_WORKER", "0")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # no voice agent in the test
    monkeypatch.setattr(A, "RUNS", tmp_path)
    monkeypatch.setattr("server.session.RUNS", tmp_path)

    async def fake_process(run_dir, emit=None, country=None, metrics=None):
        built.append(run_dir)
        return {"totals": {"book_count": 0, "books_identified": 0, "books_unidentified": 0, "books_needs_appraisal": 0,
                           "books_replacement_cost": 0, "books_used_value": 0, "items_replacement_cost_low": 0,
                           "items_replacement_cost_high": 0, "shelf_run_m": 0},
                "room": {}, "sweep": {"currency": "USD"}, "items": [], "review_queue": []}

    monkeypatch.setattr(A, "process", fake_process)
    return TestClient(A.app)


def _until(msgs_ws, kind, n=20):
    for _ in range(n):
        m = msgs_ws.receive_json()
        if m["type"] == kind:
            return m
    raise AssertionError(f"no {kind} message")


def test_reconnect_keeps_the_sweep(monkeypatch, tmp_path):
    built = []
    c = _client(monkeypatch, tmp_path, built)
    sid = c.post("/api/sweeps", json={"country": "US", "posture": "seated"}).json()["id"]
    with c.websocket_connect(f"/ws/live?sweep_id={sid}"):
        pass  # connection drops
    with c.websocket_connect(f"/ws/live?sweep_id={sid}") as ws:
        assert _until(ws, "reattached")["finished"] is False
        ws.send_json({"type": "end"})
        _until(ws, "finished")
        assert _until(ws, "packet")["json"].endswith("claim_packet.json")
    with c.websocket_connect(f"/ws/live?sweep_id={sid}") as ws:  # back after the packet was built: it is re-sent
        _until(ws, "finished")
        _until(ws, "packet")
    assert len(built) == 1


def test_packet_built_when_phone_never_returns(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(A, "RECONNECT_GRACE_S", 0.2)
    c = _client(monkeypatch, tmp_path, built)
    with c:
        sid = c.post("/api/sweeps", json={"country": "US"}).json()["id"]
        with c.websocket_connect(f"/ws/live?sweep_id={sid}"):
            pass
        for _ in range(100):  # grace period plus the short wait for the agent's last sentence
            if built:
                break
            time.sleep(0.1)
    assert built and A.SWEEPS[sid].finished
