"""Live dashboard helpers for active-learning loop visualization."""

from __future__ import annotations

import base64
import html
import io
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rdkit import Chem
    from rdkit import RDLogger
    try:
        from rdkit.Chem import Draw
    except Exception:  # pragma: no cover - optional dependency
        Draw = None  # type: ignore
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:  # pragma: no cover - optional dependency
        rdMolDraw2D = None  # type: ignore
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False
    Draw = None  # type: ignore
    rdMolDraw2D = None  # type: ignore


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt_float(value: object, digits: int = 4) -> str:
    val = _safe_float(value)
    if val is None:
        return "n/a"
    return f"{val:.{digits}f}"


def _smiles_to_data_uri(smiles: str) -> Optional[str]:
    if not smiles or not RDKit_AVAILABLE:
        return None
    try:
        RDLogger.DisableLog("rdApp.error")
    except Exception:
        pass
    try:
        mol = Chem.MolFromSmiles(smiles)
    finally:
        try:
            RDLogger.EnableLog("rdApp.error")
        except Exception:
            pass
    if mol is None:
        return None

    if Draw is not None:
        try:
            image = Draw.MolToImage(mol, size=(380, 220))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{payload}"
        except Exception:
            pass
    if rdMolDraw2D is not None:
        try:
            drawer = rdMolDraw2D.MolDraw2DSVG(380, 220)
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            if not svg:
                return None
            encoded = urllib.parse.quote(svg)
            return f"data:image/svg+xml;utf8,{encoded}"
        except Exception:
            return None
    return None


def _acq_chart_svg(history: List[Dict[str, object]], *, width: int = 760, height: int = 240) -> str:
    if not history:
        return "<div class='muted'>Noch keine Iterationen.</div>"

    best_vals: List[float] = []
    mean_vals: List[float] = []
    for row in history:
        best = _safe_float(row.get("best_acq"))
        mean = _safe_float(row.get("mean_acq"))
        if best is not None:
            best_vals.append(best)
        if mean is not None:
            mean_vals.append(mean)

    values = best_vals + mean_vals
    if not values:
        return "<div class='muted'>Keine Acquisition-Werte verfuegbar.</div>"

    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-9:
        v_max = v_min + 1.0

    pad_x = 44.0
    pad_y = 24.0
    inner_w = max(20.0, float(width) - 2.0 * pad_x)
    inner_h = max(20.0, float(height) - 2.0 * pad_y)

    def _points(seq: List[float]) -> str:
        if not seq:
            return ""
        if len(seq) == 1:
            x = pad_x + inner_w / 2.0
            y = pad_y + inner_h / 2.0
            return f"{x:.1f},{y:.1f}"
        pts: List[str] = []
        denom = float(max(1, len(seq) - 1))
        for idx, val in enumerate(seq):
            x = pad_x + (float(idx) / denom) * inner_w
            y = pad_y + (1.0 - ((val - v_min) / (v_max - v_min))) * inner_h
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    best_pts = _points(best_vals)
    mean_pts = _points(mean_vals)
    best_polyline = (
        f"<polyline fill='none' stroke='#0f766e' stroke-width='2.6' points='{best_pts}'/>"
        if best_pts
        else ""
    )
    mean_polyline = (
        f"<polyline fill='none' stroke='#f97316' stroke-width='2.2' points='{mean_pts}'/>"
        if mean_pts
        else ""
    )

    return (
        f"<svg viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' class='acq-chart'>"
        f"<rect x='1' y='1' width='{width - 2}' height='{height - 2}' fill='#f8fafc' stroke='#d1d5db' rx='8'/>"
        f"<line x1='{pad_x}' y1='{pad_y}' x2='{pad_x}' y2='{height - pad_y}' stroke='#94a3b8' stroke-width='1'/>"
        f"<line x1='{pad_x}' y1='{height - pad_y}' x2='{width - pad_x}' y2='{height - pad_y}' stroke='#94a3b8' stroke-width='1'/>"
        f"<line x1='{pad_x}' y1='{pad_y + inner_h / 2.0:.1f}' x2='{width - pad_x}' y2='{pad_y + inner_h / 2.0:.1f}' stroke='#e2e8f0' stroke-width='1'/>"
        f"{best_polyline}"
        f"{mean_polyline}"
        f"<text x='{pad_x}' y='14' font-size='11' fill='#334155' font-family='Consolas, \"Courier New\", monospace'>max {v_max:.4f}</text>"
        f"<text x='{pad_x}' y='{height - 7}' font-size='11' fill='#334155' font-family='Consolas, \"Courier New\", monospace'>min {v_min:.4f}</text>"
        "</svg>"
    )


def write_active_loop_dashboard(
    html_path: Path,
    *,
    iteration: int,
    total_iterations: int,
    labelled_count: int,
    pool_count: int,
    generated_last: int,
    selected_last: int,
    selected_rows: List[Dict[str, object]],
    history_rows: List[Dict[str, object]],
    refresh_ms: int,
    started_at: float,
    cli_lines: Optional[List[str]] = None,
) -> None:
    html_path = Path(html_path)
    refresh_ms = max(300, int(refresh_ms))
    total_iterations = max(1, int(total_iterations))
    iteration = max(0, int(iteration))
    progress = max(0.0, min(1.0, float(iteration) / float(total_iterations)))
    elapsed = max(0, int(time.time() - float(started_at)))
    elapsed_txt = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    last_best = _fmt_float(history_rows[-1].get("best_acq") if history_rows else None)
    last_mean = _fmt_float(history_rows[-1].get("mean_acq") if history_rows else None)

    cards: List[str] = []
    for row in selected_rows:
        smiles = str(row.get("smiles", "") or "")
        img_uri = _smiles_to_data_uri(smiles)
        acq_txt = _fmt_float(row.get("acquisition_score"))
        status_txt = html.escape(str(row.get("status", "n/a")))
        pred_html_parts: List[str] = []
        preds = row.get("predictions", [])
        if isinstance(preds, list):
            for pred in preds:
                if not isinstance(pred, dict):
                    continue
                name = html.escape(str(pred.get("name", "")))
                ptxt = _fmt_float(pred.get("pred"))
                stxt = _fmt_float(pred.get("std"))
                ltxt = _fmt_float(pred.get("label"))
                pred_html_parts.append(
                    f"<div class='pred-row'><span>{name}</span><span>pred {ptxt} | std {stxt} | lab {ltxt}</span></div>"
                )
        pred_html = "".join(pred_html_parts) if pred_html_parts else "<div class='muted'>Keine Property-Werte.</div>"
        image_html = (
            f"<img src='{img_uri}' alt='molecule'/>"
            if img_uri
            else "<div class='noimg'>Keine Zeichnung</div>"
        )
        cards.append(
            "<div class='mol-card'>"
            f"{image_html}"
            f"<div class='mol-meta'><div><strong>Acq:</strong> {acq_txt}</div><div><strong>Status:</strong> {status_txt}</div></div>"
            f"<div class='mono'>{html.escape(smiles or '[leer]')}</div>"
            f"<div class='preds'>{pred_html}</div>"
            "</div>"
        )

    hist_rows: List[str] = []
    for row in history_rows[-12:]:
        hist_rows.append(
            "<tr>"
            f"<td>{int(row.get('iteration', 0))}</td>"
            f"<td>{int(row.get('selected', 0))}</td>"
            f"<td>{int(row.get('generated', 0))}</td>"
            f"<td>{_fmt_float(row.get('best_acq'))}</td>"
            f"<td>{_fmt_float(row.get('mean_acq'))}</td>"
            "</tr>"
        )

    cli_render = list(cli_lines[-160:]) if cli_lines else []
    cli_html: List[str] = []
    for line in cli_render:
        text = str(line)
        if len(text) > 320:
            text = text[:317] + "..."
        cli_html.append(f"<div class='cli-line'>{html.escape(text)}</div>")

    acq_chart = _acq_chart_svg(history_rows)
    page = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Active Loop Live Dashboard</title>
  <style>
    body {{
      margin: 0;
      padding: 16px;
      font-family: Arial, sans-serif;
      background: #f5f7fa;
      color: #111827;
    }}
    .wrap {{ max-width: 1480px; margin: 0 auto; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 12px;
      align-items: start;
    }}
    h1 {{ margin: 0; font-size: 1.5rem; }}
    h2 {{ margin: 0 0 8px; font-size: 1.05rem; }}
    .muted {{ color: #64748b; }}
    .info {{ margin: 8px 0 12px; color: #334155; }}
    .box {{
      background: #fff;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 12px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }}
    .kpi {{
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px;
      background: #f8fafc;
    }}
    .kpi .key {{ color: #64748b; font-size: 0.78rem; }}
    .kpi .val {{ font-family: Consolas, 'Courier New', monospace; margin-top: 2px; font-size: 1rem; }}
    progress {{ width: 100%; height: 16px; }}
    .acq-chart {{ width: 100%; height: auto; display: block; }}
    .mol-grid {{
      margin-top: 8px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }}
    .mol-card {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 8px;
      background: #fff;
    }}
    .mol-card img {{
      width: 100%;
      height: 170px;
      object-fit: contain;
      border: 1px solid #eef2f7;
      border-radius: 6px;
      background: #fff;
    }}
    .noimg {{
      width: 100%;
      height: 170px;
      border: 1px dashed #cbd5e1;
      border-radius: 6px;
      display: grid;
      place-items: center;
      color: #64748b;
      background: #f8fafc;
    }}
    .mol-meta {{
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 0.9rem;
    }}
    .mono {{
      margin-top: 6px;
      font-family: Consolas, 'Courier New', monospace;
      font-size: 0.82rem;
      word-break: break-all;
      color: #1f2937;
    }}
    .preds {{
      margin-top: 6px;
      border-top: 1px solid #eef2f7;
      padding-top: 6px;
    }}
    .pred-row {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-family: Consolas, 'Courier New', monospace;
      font-size: 0.79rem;
      color: #334155;
      margin: 2px 0;
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
      .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="layout">
      <div>
        <h1>Active Loop Live Dashboard</h1>
        <div class="info">Iteration {iteration}/{total_iterations} | Laufzeit {elapsed_txt}</div>
        <div class="kpis">
          <div class="kpi"><div class="key">Labelled</div><div class="val">{labelled_count}</div></div>
          <div class="kpi"><div class="key">Pool</div><div class="val">{pool_count}</div></div>
          <div class="kpi"><div class="key">Generated (last)</div><div class="val">{generated_last}</div></div>
          <div class="kpi"><div class="key">Selected (last)</div><div class="val">{selected_last}</div></div>
          <div class="kpi"><div class="key">Best Acq (last)</div><div class="val">{last_best}</div></div>
          <div class="kpi"><div class="key">Mean Acq (last)</div><div class="val">{last_mean}</div></div>
        </div>
        <div class="box">
          <div class="muted">Fortschritt {progress * 100.0:.1f}%</div>
          <progress max="100" value="{progress * 100.0:.1f}"></progress>
        </div>
        <div class="box" style="margin-top:12px;">
          <h2>Acquisition Verlauf</h2>
          {acq_chart}
          <div class="muted" style="margin-top:6px;">teal = best_acq, orange = mean_acq</div>
        </div>
        <div class="box" style="margin-top:12px;">
          <h2>Ausgewaehlte Molekuele (letzte Iteration)</h2>
          <div class="mol-grid">
            {"".join(cards) if cards else "<div class='muted'>Noch keine Auswahl.</div>"}
          </div>
        </div>
        <div class="box" style="margin-top:12px;">
          <h2>Iterations-Historie</h2>
          <table>
            <thead><tr><th>Iter</th><th>Selected</th><th>Generated</th><th>Best Acq</th><th>Mean Acq</th></tr></thead>
            <tbody>{"".join(hist_rows) if hist_rows else "<tr><td colspan='5' class='muted'>Noch keine Werte</td></tr>"}</tbody>
          </table>
        </div>
      </div>
      <aside class="box sidebar">
        <h2>CLI-Ausgabe</h2>
        <div class="muted">Letzte {len(cli_render)} Zeilen</div>
        <div class="cli-log">
          {"".join(cli_html) if cli_html else "<div class='cli-line'>Noch keine CLI-Ausgabe.</div>"}
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
            print(f"[warn] Active-loop dashboard update skipped: {last_exc}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
