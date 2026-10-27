# /// script
# requires-python = ">=3.12"
# dependencies = ["playwright==1.55.0"]
# ///
"""Open every scene of the built app in headless Chromium: it must load and run without errors.

    uv run --with playwright==1.55.0 playwright install chromium   # once
    uv run scripts/mjswan/smoke.py [--dist scripts/mjswan/dist] [--seconds 12]

Catches a broken JS runtime or mjswan upgrade, not gait: whether the robot walks is check.py's
job. Saves a screenshot per scene at load and at the end into --out.
"""

import argparse
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("--dist", type=Path, default=Path(__file__).parent / "dist")
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "screenshots")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=args.dist))
threading.Thread(target=server.serve_forever, daemon=True).start()
project = json.loads((args.dist / "manifest.json").read_text())["projects"][0]

failed = False
with sync_playwright() as p:
    browser = p.chromium.launch(
        args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--ignore-gpu-blocklist"]
    )
    for scene in project["scenes"]:
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{server.server_port}/?project={project['id']}&scene={scene['id']}&panel=0")
        page.wait_for_function("window.__mjswanReady || window.__mjswanError", timeout=120_000)
        page.screenshot(path=args.out / f"{scene['id']}_0.png")
        page.wait_for_timeout(args.seconds * 1000)
        page.screenshot(path=args.out / f"{scene['id']}_1.png")
        if page.evaluate("window.__mjswanError === true"):
            errors.append("window.__mjswanError")
        print(f"{scene['name']}: " + ("; ".join(errors[:5]) or "ok"))
        failed |= bool(errors)
        page.close()
    browser.close()
raise SystemExit(1 if failed else 0)
