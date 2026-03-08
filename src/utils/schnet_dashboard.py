"""Live dashboard helpers for SchNet training."""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Dict, List, Optional


def _line_chart_svg(
    history: List[Dict[str, object]],
    *,
    width: int = 760,
    height: int = 260,
) -> str:
    train_vals: List[float] = []
    val_vals: List[float] = []
    for row in history:
        try:
            train_vals.append(float(row.get("train_loss", 0.0)))
        except Exception:
            pass
        raw_val = row.get("val_loss")
        if raw_val is None:
            continue
        try:
            val_vals.append(float(raw_val))
        except Exception:
            continue

    all_vals = train_vals + val_vals
    if not all_vals:
        return "<div class='muted'>Noch keine Kurve verfuegbar.</div>"

    pad_x = 40
    pad_y = 22
    inner_w = max(20, width - 2 * pad_x)
    inner_h = max(20, height - 2 * pad_y)
    y_min = min(all_vals)
    y_max = max(all_vals)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    def _points(values: List[float]) -> str:
        if not values:
            return ""
        if len(values) == 1:
            x = pad_x + inner_w / 2.0
            y = pad_y + inner_h / 2.0
            return f"{x:.1f},{y:.1f}"
        pts: List[str] = []
        denom = float(max(1, len(values) - 1))
        for i, value in enumerate(values):
            x = pad_x + (float(i) / denom) * inner_w
            y = pad_y + (1.0 - ((value - y_min) / (y_max - y_min))) * inner_h
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    train_pts = _points(train_vals)
    val_pts = _points(val_vals)
    y_mid = (y_min + y_max) / 2.0
    val_polyline = (
        f"<polyline fill='none' stroke='#f97316' stroke-width='2.2' points='{val_pts}'/>"
        if val_pts
        else ""
    )

    return (
        f"<svg viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' class='loss-chart'>"
        f"<rect x='1' y='1' width='{width - 2}' height='{height - 2}' fill='#f8fafc' stroke='#d1d5db' rx='8'/>"
        f"<line x1='{pad_x}' y1='{pad_y}' x2='{pad_x}' y2='{height - pad_y}' stroke='#94a3b8' stroke-width='1'/>"
        f"<line x1='{pad_x}' y1='{height - pad_y}' x2='{width - pad_x}' y2='{height - pad_y}' stroke='#94a3b8' stroke-width='1'/>"
        f"<line x1='{pad_x}' y1='{pad_y + inner_h / 2.0:.1f}' x2='{width - pad_x}' y2='{pad_y + inner_h / 2.0:.1f}' stroke='#e2e8f0' stroke-width='1'/>"
        f"<polyline fill='none' stroke='#0f766e' stroke-width='2.4' points='{train_pts}'/>"
        f"{val_polyline}"
        f"<text x='{pad_x}' y='14' font-size='11' fill='#334155' font-family='Consolas, \"Courier New\", monospace'>max {y_max:.4f}</text>"
        f"<text x='{pad_x}' y='{height - 7}' font-size='11' fill='#334155' font-family='Consolas, \"Courier New\", monospace'>min {y_min:.4f}</text>"
        f"<text x='{width - 130}' y='14' font-size='11' fill='#334155' font-family='Consolas, \"Courier New\", monospace'>mid {y_mid:.4f}</text>"
        "</svg>"
    )


def _mae_bars(
    *,
    train_mae: Optional[List[float]],
    val_mae: Optional[List[float]],
    target_names: Optional[List[str]],
) -> str:
    if not train_mae and not val_mae:
        return "<div class='muted'>Keine MAE-Werte verfuegbar.</div>"

    train_mae = train_mae or []
    val_mae = val_mae or []
    dim = max(len(train_mae), len(val_mae))
    labels = target_names or []
    if len(labels) < dim:
        labels = labels + [f"target_{idx + 1}" for idx in range(len(labels), dim)]

    max_val = 0.0
    for val in train_mae + val_mae:
        try:
            max_val = max(max_val, float(val))
        except Exception:
            continue
    max_val = max(max_val, 1e-6)

    rows: List[str] = []
    for idx in range(dim):
        tr = float(train_mae[idx]) if idx < len(train_mae) else 0.0
        va = float(val_mae[idx]) if idx < len(val_mae) else 0.0
        has_val = idx < len(val_mae)
        tr_w = max(2.0, min(100.0, 100.0 * tr / max_val))
        va_w = max(2.0, min(100.0, 100.0 * va / max_val)) if has_val else 0.0
        val_bar = f"<div class='bar val' style='width:{va_w:.2f}%' title='val {va:.4f}'></div>" if has_val else ""
        val_txt = f" | va {va:.4f}" if has_val else ""
        rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{html.escape(labels[idx])}</div>"
            "<div class='bar-stack'>"
            f"<div class='bar train' style='width:{tr_w:.2f}%' title='train {tr:.4f}'></div>"
            f"{val_bar}"
            "</div>"
            f"<div class='bar-meta'>tr {tr:.4f}{val_txt}</div>"
            "</div>"
        )
    return "".join(rows)


def write_schnet_live_dashboard(
    html_path: Path,
    *,
    title: str,
    epoch: int,
    total_epochs: int,
    best_metric: float,
    lr: float,
    history: List[Dict[str, object]],
    refresh_ms: int,
    started_at: float,
    target_names: Optional[List[str]] = None,
    cli_lines: Optional[List[str]] = None,
) -> None:
    html_path = Path(html_path)
    refresh_ms = max(200, int(refresh_ms))
    total_epochs = max(1, int(total_epochs))
    epoch = max(0, int(epoch))
    progress = max(0.0, min(1.0, float(epoch) / float(total_epochs)))
    elapsed = max(0, int(time.time() - float(started_at)))
    elapsed_txt = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    latest = history[-1] if history else {}
    train_loss = float(latest.get("train_loss", 0.0)) if latest else 0.0
    val_loss_raw = latest.get("val_loss") if latest else None
    val_loss = None if val_loss_raw is None else float(val_loss_raw)
    metric = float(latest.get("metric", train_loss)) if latest else train_loss
    train_mae = latest.get("train_mae") if isinstance(latest, dict) else None
    val_mae = latest.get("val_mae") if isinstance(latest, dict) else None

    chart = _line_chart_svg(history)
    mae_html = _mae_bars(train_mae=train_mae, val_mae=val_mae, target_names=target_names)

    cli_render = list(cli_lines[-160:]) if cli_lines else []
    cli_rows: List[str] = []
    for row in cli_render:
        text = str(row)
        if len(text) > 320:
            text = text[:317] + "..."
        cli_rows.append(f"<div class='cli-line'>{html.escape(text)}</div>")

    hist_rows: List[str] = []
    for row in history[-10:]:
        ep = int(row.get("epoch", 0))
        tr = float(row.get("train_loss", 0.0))
        va = row.get("val_loss")
        met = float(row.get("metric", tr))
        best = float(row.get("best", met))
        lr_row = float(row.get("lr", lr))
        hist_rows.append(
            "<tr>"
            f"<td>{ep}</td>"
            f"<td>{tr:.4f}</td>"
            f"<td>{'n/a' if va is None else f'{float(va):.4f}'}</td>"
            f"<td>{met:.4f}</td>"
            f"<td>{best:.4f}</td>"
            f"<td>{lr_row:.2e}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      padding: 16px;
      font-family: Arial, sans-serif;
      background: #f5f7fa;
      color: #111827;
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 12px;
      align-items: start;
    }}
    .box {{
      background: #ffffff;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 12px;
    }}
    .muted {{ color: #64748b; }}
    .info {{ margin: 8px 0 12px; color: #334155; }}
    h1 {{ margin: 0; font-size: 1.5rem; }}
    h2 {{ margin: 0 0 8px; font-size: 1.05rem; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }}
    .kpi {{
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px;
      background: #f8fafc;
    }}
    .kpi .key {{ color: #64748b; font-size: 0.8rem; }}
    .kpi .val {{ font-family: Consolas, 'Courier New', monospace; font-size: 1rem; margin-top: 2px; }}
    progress {{ width: 100%; height: 16px; }}
    .loss-chart {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr) 180px;
      gap: 8px;
      align-items: center;
      margin: 6px 0;
    }}
    .bar-label {{ font-size: 0.88rem; color: #1f2937; }}
    .bar-stack {{
      position: relative;
      height: 16px;
      border-radius: 999px;
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      overflow: hidden;
    }}
    .bar {{
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      border-radius: 999px;
    }}
    .bar.train {{ background: rgba(15, 118, 110, 0.75); }}
    .bar.val {{ background: rgba(249, 115, 22, 0.75); }}
    .bar-meta {{
      font-family: Consolas, 'Courier New', monospace;
      font-size: 0.82rem;
      color: #334155;
      text-align: right;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px; text-align: left; }}
    .sidebar {{
      position: sticky;
      top: 10px;
      max-height: calc(100vh - 20px);
      overflow: hidden;
    }}
    .cli-log {{
      margin-top: 8px;
      border: 1px solid #1f2937;
      border-radius: 6px;
      background: #0f172a;
      color: #d1d5db;
      font-family: Consolas, 'Courier New', monospace;
      font-size: 0.82rem;
      line-height: 1.35;
      height: calc(100vh - 130px);
      min-height: 260px;
      overflow: auto;
    }}
    .cli-line {{
      padding: 5px 8px;
      border-bottom: 1px solid #1e293b;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    @media (max-width: 1200px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; max-height: none; }}
      .cli-log {{ height: 260px; }}
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: 1fr 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
      .bar-meta {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="layout">
      <div>
        <h1>{html.escape(title)}</h1>
        <div class="info">Epoche {epoch}/{total_epochs} | Laufzeit {elapsed_txt}</div>
        <div class="kpis">
          <div class="kpi"><div class="key">Train Loss</div><div class="val">{train_loss:.4f}</div></div>
          <div class="kpi"><div class="key">Val Loss</div><div class="val">{'n/a' if val_loss is None else f'{val_loss:.4f}'}</div></div>
          <div class="kpi"><div class="key">Aktuelles Metric</div><div class="val">{metric:.4f}</div></div>
          <div class="kpi"><div class="key">Best Metric</div><div class="val">{best_metric:.4f}</div></div>
        </div>
        <div class="box">
          <div class="muted">Trainings-Fortschritt ({progress * 100.0:.1f}%) | LR {lr:.2e}</div>
          <progress max="100" value="{progress * 100.0:.1f}"></progress>
        </div>
        <div class="grid" style="margin-top:12px;">
          <div class="box">
            <h2>Loss-Kurven</h2>
            {chart}
            <div class="muted" style="margin-top:6px;">teal = train, orange = val</div>
          </div>
          <div class="box">
            <h2>MAE pro Ziel</h2>
            {mae_html}
          </div>
        </div>
        <div class="box" style="margin-top:12px;">
          <h2>Letzte Epochen</h2>
          <table>
            <thead><tr><th>Epoche</th><th>Train</th><th>Val</th><th>Metric</th><th>Best</th><th>LR</th></tr></thead>
            <tbody>{"".join(hist_rows) if hist_rows else "<tr><td colspan='6' class='muted'>Noch keine Werte</td></tr>"}</tbody>
          </table>
        </div>
      </div>
      <aside class="box sidebar">
        <h2>CLI-Ausgabe</h2>
        <div class="muted">Letzte {len(cli_render)} Zeilen</div>
        <div class="cli-log">
          {"".join(cli_rows) if cli_rows else "<div class='cli-line'>Noch keine CLI-Ausgabe.</div>"}
        </div>
      </aside>
    </div>
  </div>
  <script>setTimeout(function() {{ window.location.reload(); }}, {refresh_ms});</script>
</body>
</html>
"""

    html_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = html_path.with_suffix(".tmp")
    tmp_path.write_text(page, encoding="utf-8")

    replace_ok = False
    last_exc: Optional[Exception] = None
    for _ in range(20):
        try:
            tmp_path.replace(html_path)
            replace_ok = True
            break
        except PermissionError as exc:
            last_exc = exc
            time.sleep(0.05)
        except Exception as exc:
            last_exc = exc
            break

    if not replace_ok:
        if last_exc is not None:
            print(f"[warn] SchNet dashboard update skipped: {last_exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
