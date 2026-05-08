"""
セグメンテーションDPO用ペア比較WebUI。
入力画像 + 2つのマスク候補を表示し、良い方を選択。

Usage:
    python evaluate_seg_web.py --pair_dir dpo_data/seg_pairs --out dpo_data/seg_dpo_pairs.json --n 100
"""
import argparse
import json
import random
from pathlib import Path

from flask import Flask, send_from_directory, jsonify, request, Response

app = Flask(__name__)

STATE = {
    "pair_dir": None,
    "out_path": None,
    "pairs": [],
    "manifest": [],
    "current_idx": 0,
    "n_pairs": 100,
}


HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Seg DPO Evaluation</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a1a; color: #fff; font-family: -apple-system, sans-serif;
         display: flex; flex-direction: column; align-items: center; height: 100vh; }
  .header { padding: 10px; text-align: center; width: 100%; }
  .progress { font-size: 22px; font-weight: bold; }
  .hint { color: #888; font-size: 13px; margin-top: 4px; }
  .container { display: flex; flex-direction: column; align-items: center;
               flex: 1; padding: 5px; gap: 8px; }
  .ref-row { display: flex; gap: 16px; align-items: center; }
  .ref-row img { max-height: 22vh; border-radius: 8px; }
  .ref-label { color: #aaa; font-size: 12px; text-align: center; }
  .masks { display: flex; gap: 16px; }
  .card { cursor: pointer; border: 4px solid transparent; border-radius: 12px;
          overflow: hidden; transition: all 0.15s; position: relative; }
  .card:hover { border-color: #4a9eff; transform: scale(1.02); }
  .card img { max-height: 35vh; display: block; }
  .label { position: absolute; top: 8px; left: 50%; transform: translateX(-50%);
           background: rgba(0,0,0,0.7); padding: 4px 14px; border-radius: 16px;
           font-size: 16px; font-weight: bold; pointer-events: none; }
  .card:first-child .label { color: #4a9eff; }
  .card:last-child .label { color: #ffaa33; }
  .footer { padding: 8px; display: flex; gap: 16px; }
  .btn { padding: 6px 20px; border-radius: 8px; border: none; cursor: pointer;
         font-size: 14px; font-weight: bold; }
  .btn-skip { background: #444; color: #fff; }
  .btn-skip:hover { background: #555; }
  .flash { animation: flash 0.3s; }
  @keyframes flash { 0% { opacity: 0.5; } 100% { opacity: 1; } }
  .src-label { color: #aaa; font-size: 14px; }
</style>
</head><body>
<div class="header">
  <div class="progress" id="progress">Loading...</div>
  <div class="hint">Click mask or press ← (left) / → (right) / S (skip)</div>
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
    document.getElementById('container').innerHTML = '<div style="font-size:32px;margin-top:40px">Done!</div>';
    document.getElementById('progress').textContent = 'Complete: ' + data.total + ' pairs';
    return;
  }
  currentPair = data;
  document.getElementById('progress').textContent = data.progress + ' / ' + data.target;

  const swap = Math.random() < 0.5;
  const leftMask = swap ? data.mask_b : data.mask_a;
  const rightMask = swap ? data.mask_a : data.mask_b;
  currentPair._leftIs = swap ? 'b' : 'a';
  currentPair._rightIs = swap ? 'a' : 'b';

  document.getElementById('container').innerHTML =
    '<div class="ref-row">' +
    '  <div><div class="ref-label">Input</div><img src="/img/images/' + data.id + '.png"></div>' +
    '  <div><div class="ref-label">Ground Truth</div><img src="/img/gt/' + data.id + '.png"></div>' +
    '</div>' +
    '<div class="masks">' +
    '  <div class="card" onclick="choose(\\'left\\')">' +
    '    <div class="label">← L</div>' +
    '    <img src="/img/' + leftMask + '">' +
    '  </div>' +
    '  <div class="card" onclick="choose(\\'right\\')">' +
    '    <div class="label">R →</div>' +
    '    <img src="/img/' + rightMask + '">' +
    '  </div>' +
    '</div>';
}

async function choose(side) {
  if (!currentPair) return;
  const chosen = side === 'left' ? currentPair._leftIs : currentPair._rightIs;
  await fetch('/api/choose', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id: currentPair.id, chosen: chosen})
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
});

loadNext();
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/img/<path:filename>")
def serve_image(filename):
    return send_from_directory(STATE["pair_dir"], filename)


@app.route("/api/next")
def api_next():
    st = STATE
    if len(st["pairs"]) >= st["n_pairs"] or st["current_idx"] >= len(st["manifest"]):
        return jsonify({"done": True, "total": len(st["pairs"])})

    item = st["manifest"][st["current_idx"]]
    return jsonify({
        "done": False,
        "id": item["id"],
        "mask_a": f"mask_a/{item['id']}.png",
        "mask_b": f"mask_b/{item['id']}.png",
        "progress": len(st["pairs"]) + 1,
        "target": st["n_pairs"],
    })


@app.route("/api/choose", methods=["POST"])
def api_choose():
    st = STATE
    data = request.json
    item = st["manifest"][st["current_idx"]]

    if data["chosen"] == "a":
        st["pairs"].append({
            "image": f"images/{item['id']}.png",
            "preferred": f"mask_a/{item['id']}.png",
            "rejected": f"mask_b/{item['id']}.png",
        })
    else:
        st["pairs"].append({
            "image": f"images/{item['id']}.png",
            "preferred": f"mask_b/{item['id']}.png",
            "rejected": f"mask_a/{item['id']}.png",
        })

    st["current_idx"] += 1
    _save(st)
    return jsonify({"ok": True, "total": len(st["pairs"])})


@app.route("/api/skip", methods=["POST"])
def api_skip():
    STATE["current_idx"] += 1
    return jsonify({"ok": True})


def _save(st):
    with open(st["out_path"], "w") as f:
        json.dump(st["pairs"], f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair_dir", type=str, default="dpo_data/seg_pairs")
    p.add_argument("--out", type=str, default="dpo_data/seg_dpo_pairs.json")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--port", type=int, default=8502)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    pair_dir = Path(args.pair_dir).resolve()
    with open(pair_dir / "manifest.json") as f:
        manifest = json.load(f)

    pairs = []
    if args.resume and Path(args.out).exists():
        with open(args.out) as f:
            pairs = json.load(f)

    STATE["pair_dir"] = str(pair_dir)
    STATE["out_path"] = str(Path(args.out).resolve())
    STATE["pairs"] = pairs
    STATE["manifest"] = manifest
    STATE["current_idx"] = len(pairs)
    STATE["n_pairs"] = args.n

    print(f"Images: {len(manifest)}, Target: {args.n} pairs")
    print(f"\n  Open http://localhost:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
