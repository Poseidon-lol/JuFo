#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

W = 1920
H = 1080


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str,
    stroke: str,
    rx: int = 16,
    sw: int = 2,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
    )


def text_lines(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 20,
    weight: int = 500,
    color: str = "#0f172a",
    anchor: str = "start",
    line_height: float = 1.25,
) -> str:
    out = [
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'fill="{color}" text-anchor="{anchor}" font-family="Segoe UI, Inter, Arial, sans-serif">'
    ]
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * line_height
        out.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    out.append("</text>")
    return "".join(out)


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = "#1e293b",
    width: int = 2,
    dashed: bool = False,
) -> str:
    dash_attr = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"{dash_attr} />'
    )


def routed_arrow(
    points: list[tuple[float, float]],
    *,
    color: str = "#1e293b",
    width: int = 2,
    dashed: bool = False,
) -> str:
    dash_attr = ' stroke-dasharray="10 8"' if dashed else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polyline points="{pts}" fill="none" '
        f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"{dash_attr} />'
    )


def card(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: list[str],
    *,
    accent: str,
) -> str:
    return (
        rect(x, y, w, h, fill="#ffffff", stroke=accent, rx=14, sw=2)
        + text_lines(x + 16, y + 34, [title], size=30, weight=700)
        + text_lines(x + 16, y + 72, body, size=20, weight=450, color="#334155")
    )


def build_svg() -> str:
    p: list[str] = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    p.append(
        """
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#1e293b"/>
  </marker>
</defs>
"""
    )

    # Background + title.
    p.append(rect(0, 0, W, H, fill="#f8fafc", stroke="#f8fafc", rx=0, sw=0))
    p.append(text_lines(70, 70, ["KI-Architektur fuer OSC-Entdeckung"], size=56, weight=800))
    # subtitle removed by request

    # Phase containers.
    p.append(rect(50, 145, 1820, 220, fill="#eaf4ff", stroke="#93c5fd", rx=20, sw=2))
    p.append(rect(50, 390, 1820, 250, fill="#ecfdf3", stroke="#86efac", rx=20, sw=2))
    p.append(rect(50, 665, 1820, 355, fill="#f3efff", stroke="#c4b5fd", rx=20, sw=2))
    p.append(text_lines(75, 185, ["Phase 1: Generierung"], size=34, weight=700))
    p.append(text_lines(75, 430, ["Phase 2: Surrogat-Scoring und Selektion"], size=34, weight=700))
    p.append(text_lines(75, 705, ["Phase 3: QC, Retraining und RL-Feedback"], size=34, weight=700))

    # Phase 1 cards.
    p.append(
        card(
            120,
            210,
            340,
            130,
            "1) Datensatz",
            ["OPV + QC Labels", "SMILES/MOL + Split"],
            accent="#3b82f6",
        )
    )
    p.append(
        card(
            520,
            210,
            340,
            130,
            "2) JT-VAE Generator",
            ["Sampling neuer Molekuele", "Validitaet + Diversity"],
            accent="#3b82f6",
        )
    )
    p.append(
        card(
            920,
            210,
            420,
            130,
            "3) Kandidaten-Pool",
            ["Canonical SMILES", "Basisfilter + Duplikat-Check"],
            accent="#3b82f6",
        )
    )

    # Phase 2 cards.
    p.append(
        card(
            220,
            460,
            320,
            145,
            "4) Full SchNet Primary",
            ["HOMO, LUMO, Gap", "Mean + Unsicherheit"],
            accent="#16a34a",
        )
    )
    p.append(
        card(
            580,
            460,
            320,
            145,
            "5) Full SchNet Optical",
            ["lambda_max, oscillator f", "Mean + Unsicherheit"],
            accent="#16a34a",
        )
    )
    p.append(
        card(
            940,
            460,
            340,
            145,
            "6) Multi-Objective Score",
            ["Mode: red | blue | general", "Acquisition + Risk + Diversity"],
            accent="#16a34a",
        )
    )
    p.append(
        card(
            1320,
            460,
            500,
            145,
            "7) Top-K Active Learning",
            ["Waehlt Query-Kandidaten", "fuer High-Fidelity QC"],
            accent="#16a34a",
        )
    )

    # Phase 3 cards.
    p.append(
        card(
            1260,
            740,
            530,
            155,
            "8) QC (ORCA TD-DFT)",
            ["Berechnet verlassliche Labels", "u.a. lambda_max und oscillator f"],
            accent="#7c3aed",
        )
    )
    p.append(
        card(
            740,
            740,
            420,
            155,
            "9) Retraining Surrogates",
            ["Update von Primary + Optical", "Bessere Kalibrierung"],
            accent="#7c3aed",
        )
    )
    p.append(
        card(
            210,
            740,
            420,
            155,
            "10) RL-Update Generator",
            ["Reward aus Score + QC", "REINFORCE / PG / PPO"],
            accent="#7c3aed",
        )
    )

    # Main data flow arrows (orthogonal routing).
    p.append(arrow(460, 275, 520, 275))
    p.append(arrow(860, 275, 920, 275))
    p.append(routed_arrow([(1130, 340), (1130, 454), (380, 454), (380, 460)]))
    p.append(arrow(540, 520, 580, 520))
    p.append(arrow(900, 520, 940, 520))
    p.append(arrow(1280, 520, 1320, 520))
    p.append(routed_arrow([(1570, 605), (1570, 700), (1525, 700), (1525, 740)]))
    p.append(arrow(1260, 798, 1190, 798))
    p.append(arrow(820, 798, 770, 798))

    # footer notes removed by request

    p.append("</svg>")
    return "".join(p)


def main() -> None:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    out_paths = [out_dir / "osc_pipeline_clean.svg", out_dir / "osc_pipeline_clean_v2.svg"]
    for out_path in out_paths:
        out_path.write_text(svg, encoding="utf-8")
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
