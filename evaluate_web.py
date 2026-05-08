"""
DPO用ペア比較 WebUI。
ブラウザで2枚の画像を並べて表示し、クリックまたはキーボードで選択。

Usage:
    python evaluate_web.py --img_dir dpo_data/generated --out dpo_data/pairs.json --n 100
    # ブラウザで http://localhost:8501 を開く

    # 途中再開
    python evaluate_web.py --img_dir dpo_data/generated --out dpo_data/pairs.json --n 100 --resume
"""
import argparse
import json
import random
from pathlib import Path

from flask import Flask, send_from_directory, jsonify, request, Response

app = Flask(__name__)

STATE = {
    "img_dir": None,
    "out_path": None,
    "pairs": [],
    "candidates": [],
    "current_idx": 0,
    "n_pairs": 100,
    "current_left": None,
    "current_right": None,
}


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>DPO Pair Evaluation</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a1a; color: #fff; font-family: -apple-system, sans-serif;
         display: flex; flex-direction: column; align-items: center; height: 100vh; }
  .header { padding: 15px; text-align: center; width: 100%; }
  .progress { font-size: 24px; font-weight: bold; }
  .hint { color: #888; font-size: 14px; margin-top: 5px; }
  .container { display: flex; gap: 20px; flex: 1; align-items: center;
               justify-content: center; padding: 10px; max-height: calc(100vh - 120px); }
  .card { cursor: pointer; border: 4px solid transparent; border-radius: 12px;
          overflow: hidden; transition: all 0.15s; position: relative; }
  .card:hover { border-color: #4a9eff; transform: scale(1.02); }
  .card img { max-height: calc(100vh - 180px); max-width: 45vw; display: block; }
  .label { position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
           background: rgba(0,0,0,0.7); padding: 5px 16px; border-radius: 20px;
           font-size: 18px; font-weight: bold; pointer-events: none; }
  .card:first-child .label { color: #4a9eff; }
  .card:last-child .label { color: #ffaa33; }
  .footer { padding: 10px; display: flex; gap: 20px; }
  .btn { padding: 8px 24px; border-radius: 8px; border: none; cursor: pointer;
         font-size: 16px; font-weight: bold; }
  .btn-skip { background: #444; color: #fff; }
  .btn-skip:hover { background: #555; }
  .done { font-size: 32px; margin-top: 40px; }
  .flash { animation: flash 0.3s; }
  @keyframes flash { 0% { opacity: 0.5; } 100% { opacity: 1; } }
</style>
</head><body>
<div class="header">
  <div class="progress" id="progress">Loading...</div>
  <div class="hint">Click image or press ← (left) / → (right) / S (skip) / Q (save & quit)</div>
</div>
<div class="container" id="container"></div>
<div class="footer">
  <button class="btn btn-skip" onclick="skip()">Skip (S)</button>
</div>

<script>
let currentPair = null;

async function loadNext() {
  const res = await fetch('/api/next');
  const data = await res.json();
  if (data.done) {
    document.getElementById('container').innerHTML = '<div class="done">All done!</div>';
    document.getElementById('progress').textContent = `Complete: ${data.total} pairs`;
    return;
  }
  currentPair = data;
  document.getElementById('progress').textContent =
    `${data.progress} / ${data.target}`;
  document.getElementById('container').innerHTML = `
    <div class="card" onclick="choose('left')">
      <div class="label">← L</div>
      <img src="/img/${data.left}">
    </div>
    <div class="card" onclick="choose('right')">
      <div class="label">R →</div>
      <img src="/img/${data.right}">
    </div>`;
}

async function choose(side) {
  if (!currentPair) return;
  const container = document.getElementById('container');
  container.classList.remove('flash');
  void container.offsetWidth;
  container.classList.add('flash');
  await fetch('/api/choose', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({choice: side})
  });
  loadNext();
}

async function skip() {
  if (!currentPair) return;
  await fetch('/api/skip', {method: 'POST'});
  loadNext();
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft' || e.key === 'l') choose('left');
  else if (e.key === 'ArrowRight' || e.key === 'r') choose('right');
  else if (e.key === 's') skip();
  else if (e.key === 'q') {
    fetch('/api/save', {method: 'POST'}).then(() => {
      document.getElementById('container').innerHTML =
        '<div class="done">Saved & Quit</div>';
    });
  }
});

loadNext();
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/img/<path:filename>")
def serve_image(filename):
    return send_from_directory(STATE["img_dir"], filename)


@app.route("/api/next")
def api_next():
    st = STATE
    if len(st["pairs"]) >= st["n_pairs"] or st["current_idx"] >= len(st["candidates"]):
        return jsonify({"done": True, "total": len(st["pairs"])})

    a, b = st["candidates"][st["current_idx"]]
    if random.random() < 0.5:
        left, right = a, b
    else:
        left, right = b, a
    st["current_left"] = left
    st["current_right"] = right

    return jsonify({
        "done": False,
        "left": left,
        "right": right,
        "progress": len(st["pairs"]) + 1,
        "target": st["n_pairs"],
    })


@app.route("/api/choose", methods=["POST"])
def api_choose():
    st = STATE
    data = request.json
    choice = data["choice"]

    left = st["current_left"]
    right = st["current_right"]

    if choice == "left":
        st["pairs"].append({"preferred": left, "rejected": right})
    else:
        st["pairs"].append({"preferred": right, "rejected": left})

    st["current_idx"] += 1
    _save(st)
    return jsonify({"ok": True, "total": len(st["pairs"])})


@app.route("/api/skip", methods=["POST"])
def api_skip():
    STATE["current_idx"] += 1
    return jsonify({"ok": True})


@app.route("/api/save", methods=["POST"])
def api_save():
    _save(STATE)
    return jsonify({"ok": True, "total": len(STATE["pairs"])})


def _save(st):
    with open(st["out_path"], "w") as f:
        json.dump(st["pairs"], f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, default="dpo_data/generated")
    p.add_argument("--out", type=str, default="dpo_data/pairs.json")
    p.add_argument("--n", type=int, default=100, help="目標ペア数")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--port", type=int, default=8501)
    args = p.parse_args()

    img_dir = Path(args.img_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.png"))
    img_names = [img.name for img in images]
    print(f"Images: {len(img_names)}")

    evaluated_set = set()
    pairs = []
    if args.resume and out_path.exists():
        with open(out_path) as f:
            pairs = json.load(f)
        for pair in pairs:
            key = tuple(sorted([pair["preferred"], pair["rejected"]]))
            evaluated_set.add(key)
        print(f"Resumed: {len(pairs)} existing pairs")

    random.seed(42)
    candidates = []
    for i in range(len(img_names)):
        for j in range(i + 1, len(img_names)):
            candidates.append((img_names[i], img_names[j]))
    random.shuffle(candidates)
    candidates = [
        (a, b) for a, b in candidates
        if tuple(sorted([a, b])) not in evaluated_set
    ]

    STATE["img_dir"] = str(img_dir)
    STATE["out_path"] = str(out_path)
    STATE["pairs"] = pairs
    STATE["candidates"] = candidates
    STATE["n_pairs"] = args.n

    print(f"Target: {args.n} pairs")
    print(f"\n  Open http://localhost:{args.port} in your browser\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
