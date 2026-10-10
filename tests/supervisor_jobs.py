"""Test-only fault jobs for the R9 supervisor; the worker imports this module via --import."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from vplan.evaluate.worker import register


@register("fault_ok")
def ok(params: dict) -> dict:
    return {"outcome": "executed", "payload": {"echo": params.get("echo")}}


@register("fault_crash")
def crash(params: dict) -> dict:
    os._exit(70)  # no cleanup, no result file: what a native crash looks like to the parent


@register("fault_kill")
def kill(params: dict) -> dict:
    os.kill(os.getpid(), signal.SIGTERM)  # TerminateProcess on Windows
    time.sleep(60)
    return {"outcome": "executed"}


@register("fault_hang")
def hang(params: dict) -> dict:
    time.sleep(3600)
    return {"outcome": "executed"}


@register("fault_raise")
def boom(params: dict) -> dict:
    raise RuntimeError("motion layer exploded")


@register("fault_429")
def rate_limited(params: dict) -> dict:
    return {"outcome": "service_failure", "status_code": int(params.get("status", 429)),
            "retry_after_s": params.get("retry_after_s"), "payload": {}}


@register("fault_flaky_429")
def flaky(params: dict) -> dict:
    marker = Path(params["marker"])
    if not marker.exists():
        marker.write_text("seen", encoding="utf-8")
        return {"outcome": "service_failure", "status_code": 429, "retry_after_s": 0.5}
    return {"outcome": "executed", "payload": {"recovered": True}}


@register("fault_garbage_stdout")
def garbage(params: dict) -> dict:
    # The ManipulaPy banner pattern: non-ASCII and JSON-looking noise on both streams.
    print('{"outcome": "sim_error"} ✨ banner \U0001F916 \x00\xff', flush=True)
    sys.stdout.buffer.write(b"\xff\xfe\x00 raw bytes not utf-8\n")
    sys.stdout.flush()
    print("stderr noise äöü", file=sys.stderr, flush=True)
    return {"outcome": "executed", "payload": {"clean": True}}


@register("fault_bad_return")
def bad_return(params: dict) -> dict:
    return {"outcome": "teleported"}
