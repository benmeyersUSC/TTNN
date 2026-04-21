import json
import re
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


# ── Load progress JSONs ────────────────────────────────────────────────────────

def load_jsons(folder):
    pat = re.compile(r"progress(?:_[a-z]+)?_(\d+)\.json")
    entries = []
    for path in Path(folder).glob("progress*.json"):
        m = pat.match(path.name)
        if not m:
            continue
        try:
            j = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        entries.append(j)
    entries.sort(key=lambda x: x["total_seen"])
    return entries


def get_config(entries):
    for e in entries:
        if "config" in e:
            return e["config"]
    return {}


def fixloss(vals):
    """Forward-fill -1 sentinel values with the last real value."""
    out = list(vals)
    last = next((v for v in out if v != -1), None)
    if last is None:
        return out
    for i, v in enumerate(out):
        if v != -1:
            last = v
        out[i] = last
    return out


# ── Curriculum helpers derived from config ────────────────────────────────────

def make_schedules(cfg):
    epe = cfg.get("examples_per_epoch", 1)
    tgt_len = cfg.get("tgt_len", 1)
    tgt_len_min = cfg.get("tgt_len_min", tgt_len)
    tgt_len_ramp = cfg.get("tgt_len_ramp", 1.0)
    tf_lr_min = cfg.get("tf_lr_min", 1e-5)
    tf_lr_max = cfg.get("tf_lr_max", 1e-4)
    tf_ramp = cfg.get("tf_ramp_size", 60.0)
    ss_start = cfg.get("ss_ramp_start", 3.0)
    ss_end = cfg.get("ss_ramp_end", 60.0)
    ss_min = cfg.get("ss_min", 0.0)
    ss_max = cfg.get("ss_max", 0.9)

    def ss_rate(n):
        epoch = n / epe
        if epoch < ss_start:
            return 0.0
        span = max(ss_end - ss_start, 1e-9)
        return float(min(ss_max, max(ss_min, (epoch - ss_start) / span)))

    def max_tgt_len(n):
        epoch = n / epe
        if epoch >= tgt_len_ramp:
            return tgt_len
        return min(tgt_len, int(tgt_len_min + (tgt_len - tgt_len_min)
                                * epoch / max(tgt_len_ramp, 1e-9)))

    return ss_rate, max_tgt_len


# ── Dataset length CDF ────────────────────────────────────────────────────────

def load_length_cdf(checkpoint_dir, cfg):
    tgt_len = cfg.get("tgt_len", 384)
    pad_id = cfg.get("pad_id", 0)
    data_dir = Path(checkpoint_dir).parent / "data"
    if not data_dir.exists():
        return np.array([], dtype=np.int32)
    lengths = []
    for subset in sorted(data_dir.glob("subset*")):
        tgt_bin = subset / "train.tgt.bin"
        if not tgt_bin.exists():
            continue
        data = np.frombuffer(tgt_bin.read_bytes(), dtype=np.uint8)
        n = len(data) // tgt_len
        rows = data[:n * tgt_len].reshape(n, tgt_len)
        lengths.extend(int(np.sum(row != pad_id)) for row in rows)
    return np.sort(np.array(lengths, dtype=np.int32))


def pct_visible(lengths_sorted, cutoff):
    if len(lengths_sorted) == 0:
        return 1.0
    return float(np.searchsorted(lengths_sorted, cutoff, side="right")) / len(lengths_sorted)


# ── Training metrics plot ──────────────────────────────────────────────────────

def make_metrics_plot(entries, cfg, lengths_sorted):
    epe = cfg.get("examples_per_epoch", 1)
    ss_fn, mtl_fn = make_schedules(cfg)
    xs = [e["total_seen"] / epe for e in entries]

    train_loss = [e.get("train_loss") for e in entries]
    test_loss = fixloss([e.get("test_loss", -1) for e in entries])
    ar_test_loss = fixloss([e.get("ar_test_loss", -1) for e in entries])
    rl_acc = fixloss([e.get("avg_rl_accuracy",
                            e.get("avg_rl_reward", -1)) * 100.0 for e in entries])
    ss_curve = [ss_fn(e["total_seen"]) * 100.0 for e in entries]
    vis_curve = [pct_visible(lengths_sorted, mtl_fn(e["total_seen"])) * 100.0
                 for e in entries]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=train_loss, name="Train Loss",
                             line=dict(color="#ff6b6b")))
    fig.add_trace(go.Scatter(x=xs, y=test_loss, name="TF Test Loss",
                             line=dict(color="#4dabf7")))
    fig.add_trace(go.Scatter(x=xs, y=ar_test_loss, name="AR Test Loss ★",
                             line=dict(color="#ffffff", width=3),
                             marker=dict(color="#ffffff", size=5)))
    fig.add_trace(go.Scatter(x=xs, y=rl_acc, name="RL Rollout Accuracy %",
                             yaxis="y2", line=dict(color="#cc5de8", dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=xs, y=ss_curve, name="Scheduled Sampling %",
                             yaxis="y2", line=dict(color="#38d9a9", dash="dashdot", width=1.5)))
    fig.add_trace(go.Scatter(x=xs, y=vis_curve, name="Dataset % Visible",
                             yaxis="y2", line=dict(color="#fd7e14", dash="dot", width=1.5)))
    fig.update_layout(
        title="Training Metrics",
        xaxis_title="Epochs",
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="Accuracy / Schedule %", overlaying="y", side="right"),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        font=dict(color="#eee"),
        legend=dict(bgcolor="rgba(30,30,30,0.8)"),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


# ── Token distribution (animated per epoch) ───────────────────────────────────

def make_dist_plot(snapshots, pred_key, n_tokens_key, div_id):
    if not snapshots:
        return None
    ref = snapshots[-1]
    names = ref.get("token_names", [])
    true_dist = np.array(ref.get("true_dist", []), dtype=np.float64)
    if len(true_dist) == 0:
        return None

    mask = true_dist > 1e-9
    order = np.where(mask)[0][np.argsort(true_dist[mask])[::-1]]
    labels = [names[i] if i < len(names) else str(i) for i in order]
    td = true_dist[order]
    xs = np.arange(len(labels), dtype=float)
    zero = np.zeros(len(labels))
    epe = snapshots[0].get("_epe", 1)

    def filled(x, ytop, ybot, color, name="", showlegend=False):
        xf = np.concatenate([x, x[::-1]])
        yf = np.concatenate([ytop, ybot[::-1]])
        return go.Scatter(x=xf.tolist(), y=yf.tolist(), fill="toself", fillcolor=color,
                          line=dict(color="rgba(0,0,0,0)", width=0, shape="spline"),
                          name=name, showlegend=showlegend, hoverinfo="skip")

    def pred_traces(snap):
        pd_ = np.array(snap.get(pred_key, [0.0] * len(order)), dtype=np.float64)[order]
        n_tok = snap.get(n_tokens_key, 0)
        epoch = snap["total_seen"] / epe
        eps = 1e-10
        ce = float(-np.sum(td * np.log(pd_ + eps)))
        h = float(-np.sum(td[td > eps] * np.log(td[td > eps] + eps)))
        kl = ce - h
        t0 = filled(xs, np.maximum(td, pd_), zero, "rgba(220,40,40,0.40)",
                    name="Mismatch", showlegend=True)
        t1 = filled(xs, np.minimum(td, pd_), zero, "rgba(60,200,110,0.30)")
        t2 = filled(xs, pd_, zero, "rgba(240,200,50,0.18)")
        t3 = go.Scatter(
            x=xs.tolist(), y=pd_.tolist(), mode="lines+markers", name="Pred q(x)",
            line=dict(color="rgba(240,200,50,1.0)", width=2.5, shape="spline"),
            marker=dict(color="rgba(240,200,50,0.95)", size=6,
                        line=dict(color="#fff", width=1)),
            customdata=list(zip([f"{v * 100:.2f}%" for v in pd_], labels)),
            hovertemplate="<b>%{customdata[1]}</b><br>pred=%{customdata[0]}<extra></extra>")
        title = (f"Token Distribution — epoch {epoch:.1f}  |  "
                 f"CE={ce:.4f}  KL={kl:.4f}  H(p)={h:.4f}  n={n_tok:,}")
        return [t0, t1, t2, t3], title

    static_fill = filled(xs, td, zero, "rgba(60,220,120,0.22)")
    static_line = go.Scatter(
        x=xs.tolist(), y=td.tolist(), mode="lines+markers", name="True p(x)",
        line=dict(color="rgba(60,220,120,1.0)", width=2.5, shape="spline"),
        marker=dict(color="rgba(60,220,120,0.95)", size=6, line=dict(color="#fff", width=1)),
        customdata=list(zip([f"{v * 100:.2f}%" for v in td], labels)),
        hovertemplate="<b>%{customdata[1]}</b><br>true=%{customdata[0]}<extra></extra>")

    init_traces, init_title = pred_traces(snapshots[-1])
    fig = go.Figure(data=init_traces + [static_fill, static_line])
    frames, frame_keys = [], []
    for snap in snapshots:
        ft, ftt = pred_traces(snap)
        key = str(snap["total_seen"])
        frames.append(go.Frame(data=ft, traces=[0, 1, 2, 3], name=key,
                               layout=go.Layout(title_text=ftt)))
        frame_keys.append(key)
    fig.frames = frames
    fig.update_layout(
        title=init_title,
        xaxis=dict(tickvals=xs.tolist(), ticktext=labels, tickangle=-60,
                   tickfont=dict(size=10), gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="probability", gridcolor="rgba(255,255,255,0.05)",
                   tickformat=".2%"),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font=dict(color="#eee"),
        legend=dict(bgcolor="rgba(30,30,30,0.8)", orientation="h",
                    x=0.5, xanchor="center", y=1.08),
        margin=dict(t=80, b=60), sliders=[], updatemenus=[],
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id), frame_keys


# ── Confidence histogram ───────────────────────────────────────────────────────

def make_conf_hist(snapshots, hist_key, div_id):
    if not snapshots:
        return ""
    N = 50
    bins = [(i + 0.5) / N for i in range(N)]
    zero = [0.0] * N
    eps = 1e-10
    epe = snapshots[0].get("_epe", 1)

    def wmean(h):
        return float(sum(bins[i] * h[i] for i in range(N)))

    def hnll(h):
        return float(-sum(h[i] * np.log(bins[i] + eps) for i in range(N)))

    def traces(h):
        m, nll = wmean(h), hnll(h)
        return [
            go.Bar(x=bins, y=h, marker_color="rgba(100,180,255,0.75)",
                   marker_line=dict(color="rgba(100,180,255,1.0)", width=0.5),
                   name="P(correct token)",
                   hovertemplate="P=%{x:.2f}: %{y:.4f}<extra></extra>"),
            go.Scatter(x=[m, m], y=[0, 1], mode="lines", name=f"mean={m:.3f}",
                       line=dict(color="rgba(255,80,80,1.0)", width=2, dash="dash"),
                       hovertemplate=f"mean={m:.3f}<extra></extra>"),
            go.Scatter(x=[0, 1], y=[nll, nll], mode="lines", yaxis="y2",
                       name=f"NLL={nll:.3f}",
                       line=dict(color="rgba(255,180,50,0.9)", width=2, dash="dot"),
                       hovertemplate=f"NLL={nll:.3f}<extra></extra>"),
        ]

    h0 = snapshots[-1].get(hist_key, zero)
    ep0 = snapshots[-1]["total_seen"] / epe
    fig = go.Figure(data=traces(h0))
    frames = []
    for snap in snapshots:
        h = snap.get(hist_key, zero)
        ep = snap["total_seen"] / epe
        frames.append(go.Frame(
            data=traces(h), traces=[0, 1, 2], name=str(snap["total_seen"]),
            layout=go.Layout(title_text=(f"P(correct token) — epoch {ep:.1f}  |  "
                                         f"mean={wmean(h):.3f}  NLL={hnll(h):.3f}"))))
    fig.frames = frames
    fig.update_layout(
        title=f"P(correct token) — epoch {ep0:.1f}  |  mean={wmean(h0):.3f}  NLL={hnll(h0):.3f}",
        xaxis=dict(title="P(correct token)", range=[0, 1],
                   tickvals=[i / 10 for i in range(11)],
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="fraction of positions", rangemode="nonnegative",
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(title=dict(text="NLL", font=dict(color="rgba(255,180,50,0.9)")),
                    overlaying="y", side="right", rangemode="nonnegative",
                    tickfont=dict(color="rgba(255,180,50,0.9)"),
                    gridcolor="rgba(255,255,255,0.0)"),
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font=dict(color="#eee"),
        sliders=[], updatemenus=[], bargap=0.05,
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)


# ── Shared epoch slider ────────────────────────────────────────────────────────

def make_shared_slider(snapshots, frame_keys, epe):
    n = len(snapshots)
    epoch_labels = json.dumps([f'{s["total_seen"] / epe:.1f}' for s in snapshots])
    keys_js = json.dumps(frame_keys)
    return f"""
<div style="position:fixed; bottom:0; left:0; right:0; z-index:9999;
            background:rgba(20,20,20,0.96); border-top:1px solid #333;
            padding:10px 24px; display:flex; align-items:center; justify-content:center;">
  <div style="display:flex; align-items:center; gap:16px; width:60%;">
    <span style="color:#4fc1ff; font-size:13px; min-width:50px;">Epoch</span>
    <input id="shared-slider" type="range" min="0" max="{n - 1}" value="{n - 1}"
           style="flex:1; accent-color:#4fc1ff;">
    <span id="shared-label"
          style="color:#4fc1ff; font-size:14px; font-weight:bold; min-width:80px;"></span>
  </div>
</div>
<div style="height:52px;"></div>
<script>
  const _FRAME_KEYS   = {keys_js};
  const _EPOCH_LABELS = {epoch_labels};

  function sharedSeek(idx) {{
    document.getElementById('shared-label').textContent = 'epoch ' + _EPOCH_LABELS[idx];
    const key  = _FRAME_KEYS[idx];
    const anim = id => {{
      if (document.getElementById(id))
        Plotly.animate(id, [key],
          {{frame:{{duration:0,redraw:true}}, mode:'immediate', transition:{{duration:0}}}});
    }};
    anim('dist-tf'); anim('dist-ar'); anim('conf-tf'); anim('conf-ar');
  }}

  document.getElementById('shared-slider').addEventListener('input', function() {{
    sharedSeek(parseInt(this.value));
  }});
  window.addEventListener('load', () => sharedSeek({n - 1}));
</script>
"""


# ── Toggle button pair ─────────────────────────────────────────────────────────

def toggle_pair(label_a, label_b, fn, div_a, div_b, plot_a, plot_b):
    def btn(bid, label, active):
        bg, col, bord = ("#1a4a6a", "#4fc1ff", "#4fc1ff") if active else ("#2a2a2a", "#ccc", "#444")
        return (f'<button id="btn-{bid}" onclick="{fn}(\'{bid}\')" '
                f'style="padding:5px 14px; border-radius:4px; font-size:12px; cursor:pointer; '
                f'background:{bg}; color:{col}; border:1px solid {bord}; '
                f'font-family:monospace;">{label}</button>')

    return f"""
<div style="display:flex; gap:8px; margin-bottom:8px;">
  {btn(label_a, label_a.upper(), True)}
  {btn(label_b, label_b.upper(), False)}
</div>
<script>
  function {fn}(m) {{
    document.getElementById('{div_a}').style.display = m==='{label_a}' ? '' : 'none';
    document.getElementById('{div_b}').style.display = m==='{label_b}' ? '' : 'none';
    ['{label_a}','{label_b}'].forEach(id => {{
      const on = id === m;
      const b  = document.getElementById('btn-'+id);
      b.style.background  = on ? '#1a4a6a' : '#2a2a2a';
      b.style.color       = on ? '#4fc1ff' : '#ccc';
      b.style.borderColor = on ? '#4fc1ff' : '#444';
    }});
    if (m==='{label_a}') Plotly.relayout('{plot_a}', {{autosize:true}});
    if (m==='{label_b}') Plotly.relayout('{plot_b}', {{autosize:true}});
  }}
</script>
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_html = sys.argv[2] if len(sys.argv) > 2 else "training_dashboard.html"

    if checkpoint_dir is None:
        candidates = sorted(Path(".").glob("checkpoints_*"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("No checkpoints_* directory found. Pass one as argument.")
            return
        checkpoint_dir = str(candidates[-1])
        print(f"Auto-selected checkpoint dir: {checkpoint_dir}")

    entries = load_jsons(checkpoint_dir)
    if not entries:
        print(f"No progress_*.json files found in {checkpoint_dir}")
        return

    cfg = get_config(entries)
    epe = cfg.get("examples_per_epoch", 1)
    for e in entries:
        e["_epe"] = epe

    print(f"Loaded {len(entries)} checkpoints  |  epe={epe}  |  vocab={cfg.get('vocab_size', '?')}")

    print("Scanning dataset lengths...", end=" ", flush=True)
    lengths = load_length_cdf(checkpoint_dir, cfg)
    print(f"{len(lengths):,} examples")

    metrics_html = make_metrics_plot(entries, cfg, lengths)

    # Distribution plots — only checkpoints that have distribution data
    dist_snaps = [e for e in entries if "tf_pred_dist" in e and "true_dist" in e]
    dist_section = ""
    slider_html = ""

    if dist_snaps:
        tf_result = make_dist_plot(dist_snaps, "tf_pred_dist", "tf_n_tokens", "dist-tf")
        ar_result = make_dist_plot(dist_snaps, "ar_pred_dist", "ar_n_tokens", "dist-ar")

        tf_html, frame_keys = tf_result if tf_result else ("", [])
        ar_html = ar_result[0] if ar_result else ""

        conf_tf = make_conf_hist(dist_snaps, "tf_conf_hist", "conf-tf")
        conf_ar = make_conf_hist(dist_snaps, "ar_conf_hist", "conf-ar")

        dist_tog = toggle_pair("tf", "ar", "showDist",
                               "div-dist-tf", "div-dist-ar", "dist-tf", "dist-ar")
        conf_tog = toggle_pair("tf", "ar", "showConf",
                               "div-conf-tf", "div-conf-ar", "conf-tf", "conf-ar")

        dist_section = f"""
<h2>Token Distribution — Test Set</h2>
{dist_tog}
<div id="div-dist-tf">{tf_html}</div>
<div id="div-dist-ar" style="display:none;">{ar_html}</div>
<h2>Confidence Histogram — P(correct token | context)</h2>
{conf_tog}
<div id="div-conf-tf">{conf_tf}</div>
<div id="div-conf-ar" style="display:none;">{conf_ar}</div>
"""
        if frame_keys:
            slider_html = make_shared_slider(dist_snaps, frame_keys, epe)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family:'Segoe UI',Tahoma,sans-serif; background:#111; color:#eee;
          margin:0; padding:20px; }}
  h2   {{ color:#4fc1ff; font-weight:normal; margin:30px 0 10px; }}
  .wrap {{ max-width:1200px; margin:auto; }}
</style>
</head>
<body>
<div class="wrap">
  <h2>TransformerTrainer Dashboard
    <span style="color:#555; font-size:13px; font-weight:normal;">
      &nbsp;·&nbsp; {checkpoint_dir}
    </span>
  </h2>
  {metrics_html}
  {dist_section}
</div>
{slider_html}
</body>
</html>"""

    Path(output_html).write_text(html)
    print(f"Dashboard written → {output_html}")


if __name__ == "__main__":
    main()
