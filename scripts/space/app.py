"""Hugging Face Space entry point: vplan's Gradio UI in bring-your-own-key mode (UISettings.byok).

The Space root is the folder scripts/space/build_space.py assembles: src/vplan (the repository's layout, because
vplan/config.py finds config/ two levels above its package), config/, and data/ with the bundled replay runs."""
import spaces  # first, and unconditionally: off ZeroGPU hardware it does nothing

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MANIPULAPY_QUIET", "1")  # as in the container (docker/Dockerfile): no import banner in the logs

from vplan.ui.gradio_app import UISettings, launch  # noqa: E402  (after sys.path)


@spaces.GPU(duration=1)
def _zerogpu_placeholder() -> None:
    """Never called. A ZeroGPU Space refuses to start without one @spaces.GPU function, and this app needs no GPU; on
    other hardware the decorator does nothing."""


if __name__ == "__main__":
    seeds = str(ROOT / "data" / "seeds" / "test")
    launch(UISettings(
        byok=True, record_dir=None, cosmos_url=None, host="0.0.0.0", port=7860,
        replay_runs=(
            ("Nemotron 3.5 Lightning, held-out evaluation (the headline run)",
             str(ROOT / "data" / "runs" / "lightning-d70"), seeds),
            ("Nemotron 3 Ultra, held-out evaluation with the richer repair findings",
             str(ROOT / "data" / "runs" / "ultra-d85"), seeds),
        )))
