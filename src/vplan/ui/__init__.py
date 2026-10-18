"""W3.5 Gradio UI (spec §15, D22): live mode (scene or seed, instruction, planner, repair rounds, Cosmos frames and
verdicts or the outage banner, execution, Judge) and keyless replay of recorded attempts.

Only ui.gradio_app imports Gradio. views, markup and display are plain data and HTML; render and live drive PyBullet; sources
reads recorded attempts. Replay needs no keys, no GPU and no network.
"""
