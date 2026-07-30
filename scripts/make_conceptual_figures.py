"""
make_conceptual_figures.py
===========================
Generates the two conceptual-model figures for Chapter 3 (Conceptual model):

  FIGURE 3.1  UML class diagram of the model entities, their state variables
              and operations (Worker, Producer/Firm, Task, Market).
  FIGURE 3.2  Agent-interaction (UML sequence) diagram: which entity exchanges
              what information within a single tick.

Content is derived from model/labour_market_model.py (classes Task, Worker,
Producer, LabourMarketModel and the fixed sub-step order in step()).

HOW TO EDIT
  * Figure 3.1 -> edit the CLASSES dict and the RELATIONS list.
  * Figure 3.2 -> edit the MESSAGES list (one tuple per arrow) and PHASES.
Then re-run; output is written to PNG (300 dpi) and PDF (vector).
No in-figure title is drawn: add the figure number / caption in LaTeX.

    python make_conceptual_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from matplotlib.path import Path
from matplotlib.patches import PathPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

INK      = "#1a1a1a"
HEADER   = "#2f4b7c"
HEADERTX = "#ffffff"
BODY     = "#fbfcfe"
BORDER   = "#2f4b7c"
ACCENT   = "#c1121f"
MUTED    = "#6b6b6b"
MONO     = "DejaVu Sans Mono"

COMP = {
    "C1": "#264653", "C2": "#2a9d8f", "C3": "#5a9e57", "C4": "#caa23a",
    "C5": "#e07a3f", "C6": "#d1495b", "C7": "#7d5bd6", "M1": "#6b6b6b",
}
COMP_LABEL = {
    "C1": "C1 Production", "C2": "C2 Labour market", "C3": "C3 Wage setting",
    "C4": "C4 AI adoption", "C5": "C5 Learning curve", "C6": "C6 Demand",
    "C7": "C7 Employment prot.",
}

# ===========================================================================
# FIGURE 3.1 -- UML CLASS DIAGRAM
# ===========================================================================
CLASSES = {
    "Market": {
        "pos": (3.5, 95), "w": 43,
        "stereotype": "<<environment>>",
        "name": "LabourMarketModel  (Market)",
        "attrs": [
            "workers : List[Worker]",
            "producers : List[Producer]",
            "w_h, w_l : float       # high / low-skill wage",
            "p : float              # goods-market price",
            "A, A_base, B : float   # inverse-demand schedule",
            "gamma, gamma_pi : float  # consumption propensities",
            "Y : float              # aggregate wage bill (lagged)",
            "Pi : float             # aggregate profit (lagged)",
            "k_ai : float           # AI rental cost (learning)",
            "delta, lam : float     # separation / wage-adj. rate",
            "employment_protection : bool",
        ],
        "ops": [
            ("step()  -- one tick", "M1"),
            ("separate_workers()", "C2"),
            ("match_and_hire()", "C2"),
            ("update_wages()", "C3"),
            ("calculate_wage_bill() -> Y", "C3"),
            ("calculate_profit_income() -> Pi", "C6"),
            ("calculate_market_price() -> p", "C6"),
            ("ai_cost_at(t) -> k_ai", "C5"),
            ("process_chain_limit()", "C7"),
        ],
    },
    "Producer": {
        "pos": (58, 95), "w": 38.5,
        "stereotype": "<<agent>>",
        "name": "Producer  (Firm)",
        "attrs": [
            "alpha : float        # share of routine tasks",
            "n_tasks : int",
            "tasks : List[Task]",
            "employees : List[Worker]",
            "c : Dict[id, float]  # Leontief coeff.",
            "output : float",
        ],
        "ops": [
            ("produce() -> U", "C1"),
            ("trim_ai_capacity()", "C1"),
            ("unit_cost_human() / unit_cost_ai()", "C4"),
            ("preferred_input_for_task()", "C4"),
            ("calculate_npv()", "C4"),
            ("should_automate()", "C4"),
            ("record_automation_decision()", "C4"),
            ("hiring_requests()", "C2"),
            ("on_worker_separated()", "C2"),
            ("severance_cost_for_task()", "C7"),
        ],
    },
    "Worker": {
        "pos": (3.5, 38), "w": 43,
        "stereotype": "<<agent>>",
        "name": "Worker   (passive)",
        "attrs": [
            "skill_level : {'high', 'low'}",
            "employed : bool",
            "employer : Producer",
            "task : Task",
            "wage : float",
            "tenure : int        # steps at current employer",
            "contract_type : {'flex', 'permanent'}",
        ],
        "ops": [
            ("-- no autonomous decisions --", None),
            ("state set centrally by Market:", None),
            ("hiring, separation, wages, tenure", None),
        ],
    },
    "Task": {
        "pos": (58, 38), "w": 38.5,
        "stereotype": "<<dataclass>>",
        "name": "Task  (unit of production)",
        "attrs": [
            "task_id : int",
            "task_type : {'routine', 'non_routine'}",
            "producer : Producer",
            "complexity_index : int  # 1 = simplest",
            "automated : bool",
            "employees : List[Worker]",
            "n_ai : int",
            "investment_cost : float",
            "ai_investment_step : int | None",
            "ai_investment_npv : float | None",
        ],
        "ops": [],
    },
}

RELATIONS = [
    ("Market",   "Producer", "aggregate", "manages",     "1", "1..*"),
    ("Market",   "Worker",   "aggregate", "manages",     "1", "0..*"),
    ("Producer", "Task",     "compose",   "owns",        "1", "1..*"),
    ("Task",     "Worker",   "assoc",     "assigned",    "0..*", "0..*"),
    ("Worker",   "Producer", "assoc",     "employed by", "0..*", "0..1"),
]


def _draw_class_box(ax, name, spec):
    x, y = spec["pos"]
    w = spec["w"]
    line_h = 2.15
    pad = 1.0
    attrs = spec["attrs"]
    ops = spec["ops"]
    head_h = 5.6
    attr_h = pad * 2 + max(len(attrs), 1) * line_h
    op_h = (pad * 2 + len(ops) * line_h) if ops else pad
    total_h = head_h + attr_h + op_h

    ax.add_patch(FancyBboxPatch((x + 0.5, y - total_h - 0.5), w, total_h,
        boxstyle="round,pad=0,rounding_size=0.8", linewidth=0,
        facecolor="#00000010", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y - total_h), w, total_h,
        boxstyle="round,pad=0,rounding_size=0.8", linewidth=1.5,
        edgecolor=BORDER, facecolor=BODY, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y - head_h), w, head_h,
        boxstyle="round,pad=0,rounding_size=0.8", linewidth=0,
        facecolor=HEADER, zorder=3))
    ax.add_patch(Rectangle((x, y - head_h), w, head_h * 0.5,
        facecolor=HEADER, edgecolor="none", zorder=3))
    ax.text(x + w / 2, y - head_h * 0.34, spec["stereotype"], ha="center",
        va="center", color="#cdd8ec", fontsize=9.0, style="italic", zorder=4)
    ax.text(x + w / 2, y - head_h * 0.72, spec["name"], ha="center",
        va="center", color=HEADERTX, fontsize=11.5, fontweight="bold", zorder=4)

    ay = y - head_h - pad - line_h * 0.5
    for a in attrs:
        if "#" in a:
            code, comment = a.split("#", 1)
            ax.text(x + pad, ay, code, ha="left", va="center", fontsize=8.7,
                color=INK, family=MONO, fontweight="bold", zorder=4)
            ax.text(x + pad + len(code) * 0.67, ay, "# " + comment.strip(),
                ha="left", va="center", fontsize=7.7, color=MUTED, family=MONO,
                style="italic", zorder=4)
        else:
            ax.text(x + pad, ay, a, ha="left", va="center", fontsize=8.7,
                color=INK, family=MONO, fontweight="bold", zorder=4)
        ay -= line_h

    sep_y = y - head_h - attr_h
    ax.plot([x + 0.4, x + w - 0.4], [sep_y, sep_y], color="#c7d0e0", lw=1.0, zorder=4)

    if ops:
        oy = sep_y - pad - line_h * 0.5
        for text, comp in ops:
            ax.text(x + pad, oy, text, ha="left", va="center", fontsize=8.7,
                color=INK, family=MONO, fontweight="bold", zorder=4)
            if comp:
                col = COMP[comp]
                cx = x + w - pad - 3.7
                ax.add_patch(FancyBboxPatch((cx, oy - 0.9), 3.7, 1.8,
                    boxstyle="round,pad=0,rounding_size=0.3", linewidth=0,
                    facecolor=col, zorder=4))
                ax.text(cx + 1.85, oy, comp, ha="center", va="center",
                    fontsize=7.3, color="white", fontweight="bold", zorder=5)
            oy -= line_h

    spec["_box"] = (x, y - total_h, w, total_h)


def _anchor(box, side):
    x, yb, w, h = box
    return {"top": (x + w / 2, yb + h), "bottom": (x + w / 2, yb),
            "left": (x, yb + h / 2), "right": (x + w, yb + h / 2),
            "corner_tr": (x + w, yb + h), "corner_bl": (x, yb)}[side]


def _diamond(ax, tip, direction, filled):
    ux, uy = direction
    px, py = -uy, ux
    L, Wd = 2.0, 1.15
    base = (tip[0] + ux * L, tip[1] + uy * L)
    pts = [tip,
           (tip[0] + ux * L / 2 + px * Wd, tip[1] + uy * L / 2 + py * Wd),
           base,
           (tip[0] + ux * L / 2 - px * Wd, tip[1] + uy * L / 2 - py * Wd)]
    ax.add_patch(Polygon(pts, closed=True,
        facecolor=(INK if filled else "white"), edgecolor=INK, lw=1.3, zorder=6))
    return base


def draw_figure_31(path_png, path_pdf):
    fig, ax = plt.subplots(figsize=(10.6, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 97)
    ax.axis("off")

    for name, spec in CLASSES.items():
        _draw_class_box(ax, name, spec)

    routes = {
        ("Market", "Producer"): ("right",  "left",   "straight"),
        ("Market", "Worker"):   ("bottom", "top",    "straight"),
        ("Producer", "Task"):   ("bottom", "top",    "straight"),
        ("Task", "Worker"):     ("left",   "right",  "straight"),
        ("Worker", "Producer"): ("corner_tr", "corner_bl", "straight"),
    }
    dir_for_side = {"top": (0, 1), "bottom": (0, -1), "left": (-1, 0), "right": (1, 0)}

    for (frm, to, kind, label, m_from, m_to) in RELATIONS:
        s_from, s_to, style = routes[(frm, to)]
        a = _anchor(CLASSES[frm]["_box"], s_from)
        b = _anchor(CLASSES[to]["_box"], s_to)

        line_start = a
        if kind in ("aggregate", "compose"):
            line_start = _diamond(ax, a, dir_for_side[s_from], filled=(kind == "compose"))

        if style == "elbow":
            midx = (line_start[0] + b[0]) / 2 + 4
            verts = [line_start, (midx, line_start[1]), (midx, b[1]), b]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
            ax.add_patch(PathPatch(Path(verts, codes), fill=False,
                edgecolor=INK, lw=1.3, zorder=2))
            ax.add_patch(FancyArrowPatch((b[0] + 2.2, b[1]), b, arrowstyle="-|>",
                mutation_scale=15, lw=0, color=INK, zorder=5))
            lblx, lbly = midx, (line_start[1] + b[1]) / 2
            fxx, fyy = line_start[0] + 2.5, line_start[1] + 1.3
            txx, tyy = b[0] - 2.5, b[1] + 1.6
        else:
            astyle = "-|>" if kind == "assoc" else "-"
            ax.add_patch(FancyArrowPatch(line_start, b, arrowstyle=astyle,
                mutation_scale=15, lw=1.3, color=INK, zorder=2))
            lblx = (line_start[0] + b[0]) / 2
            lbly = (line_start[1] + b[1]) / 2
            fxx = line_start[0] + (b[0] - line_start[0]) * 0.14
            fyy = line_start[1] + (b[1] - line_start[1]) * 0.14
            txx = line_start[0] + (b[0] - line_start[0]) * 0.86
            tyy = line_start[1] + (b[1] - line_start[1]) * 0.86

        # orientation: offset the label off the line; push multiplicities to the ends
        dxl, dyl = (b[0] - line_start[0]), (b[1] - line_start[1])
        if abs(dyl) >= abs(dxl):          # vertical-ish -> label to the left
            lx2, ly2 = (line_start[0] + b[0]) / 2 - 6.5, (line_start[1] + b[1]) / 2
        else:                              # horizontal-ish -> label above
            lx2, ly2 = (line_start[0] + b[0]) / 2, (line_start[1] + b[1]) / 2 + 1.7
        ax.text(lx2, ly2, label, ha="center", va="center", fontsize=9.4,
            style="italic", color=ACCENT, fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=1.0))
        ax.text(line_start[0] + dxl * 0.06, line_start[1] + dyl * 0.06 + 1.5, m_from,
            ha="center", va="center", fontsize=8.4, color=MUTED, fontweight="bold",
            zorder=8, bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=1.0))
        ax.text(line_start[0] + dxl * 0.94, line_start[1] + dyl * 0.94 + 1.5, m_to,
            ha="center", va="center", fontsize=8.4, color=MUTED, fontweight="bold",
            zorder=8, bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=1.0))

    ax.text(3.5, 3.0, "Component tags:", ha="left", va="center", fontsize=8.4,
        color=INK, fontweight="bold")
    lx = 14.0
    for comp, lab in COMP_LABEL.items():
        ax.add_patch(FancyBboxPatch((lx, 2.4), 1.7, 1.6,
            boxstyle="round,pad=0,rounding_size=0.3", linewidth=0,
            facecolor=COMP[comp], zorder=4))
        ax.text(lx + 2.3, 3.0, lab, ha="left", va="center", fontsize=7.6, color=INK)
        lx += len(lab) * 0.50 + 3.7

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_pdf)
    plt.close(fig)


# ===========================================================================
# FIGURE 3.2 -- UML SEQUENCE DIAGRAM (one tick)
# ===========================================================================
LIFELINES = ["Market", "Producer", "Task", "Worker"]
LIFELINE_LABELS = {
    "Market": "Market\n: LabourMarketModel", "Producer": "Producer\n: Firm",
    "Task": "Task", "Worker": "Worker",
}

MESSAGES = [
    ("0",  "C5", "Market", "Market",   "advance tick; k_ai(t) = ai_cost_at()  (learning curve)", "self"),
    ("1",  "C6", "Market", "Market",   "capture lagged wage bill Y and profit income Pi", "self"),
    ("2",  "C1", "Market", "Producer", "produce()", "call"),
    ("2",  "C1", "Producer", "Task",   "read automated?, employees, n_ai", "call"),
    ("2",  "C1", "Task", "Producer",   "task output", "return"),
    ("2",  "C1", "Producer", "Market", "output U;  trim_ai_capacity()", "return"),
    ("3",  "C6", "Market", "Market",   "price  p = A - B*Q,   A = A_base + g*Y + g_pi*Pi", "self"),
    ("4",  "C7", "Market", "Producer", "process_chain_limit(): convert / non-renew flex", "call"),
    ("4",  "C7", "Producer", "Worker", "contract_type -> 'permanent'  or  dismiss", "call"),
    ("5",  "C2", "Market", "Worker",   "separate_workers() at rate delta:  employed -> False", "call"),
    ("5",  "C2", "Worker", "Producer", "on_worker_separated()  callback", "call"),
    ("5",  "C4", "Producer", "Task",   "reactive automation: NPV / ULC -> automated, n_ai", "call"),
    ("6",  "C3", "Market", "Market",   "update_wages(): wage curve + partial adjustment -> w_h, w_l", "self"),
    ("7",  "C3", "Market", "Market",   "adaptive expectations: smooth dw_h, dw_l", "self"),
    ("8",  "C4", "Market", "Producer", "hiring_requests()   (gate: should_produce_more)", "call"),
    ("8",  "C4", "Producer", "Task",   "proactive automation -> automated, n_ai", "call"),
    ("8",  "C2", "Market", "Worker",   "match_and_hire(): assign unemployed worker", "call"),
    ("8",  "C2", "Worker", "Task",     "set employed, employer, wage, task, tenure = 0", "call"),
    ("9",  "C7", "Market", "Worker",   "tenure += 1  (employed workers)", "call"),
    ("10", "C4", "Market", "Market",   "update displacement_flow_smoothed  (mean-field)", "self"),
    ("11", "C3", "Market", "Worker",   "propagate wage level  w.wage = w_h / w_l", "call"),
    ("12", "C1", "Market", "Producer", "produce() + price again  (end-of-tick consistency)", "call"),
    ("13", "M1", "Market", "Market",   "_compute_all_stats();  datacollector.collect()", "self"),
]

PHASES = [
    ("A\nDemand",      {"0", "1"}),
    ("B\nProduction",  {"2", "3"}),
    ("C\nProtection",  {"4"}),
    ("D\nSeparation",  {"5"}),
    ("E\nWages",       {"6", "7"}),
    ("F\nMatching",    {"8"}),
    ("G\nData",        {"9", "10", "11", "12", "13"}),
]
PHASE_TINT = ["#f5f8fc", "#eef3f9"]


def _phase_of(step):
    for i, (lab, steps) in enumerate(PHASES):
        if step in steps:
            return i, lab
    return -1, ""


def draw_figure_32(path_png, path_pdf):
    n = len(LIFELINES)
    x0, x1 = 24.0, 93.0
    col_x = {name: x0 + i * (x1 - x0) / (n - 1) for i, name in enumerate(LIFELINES)}

    n_msg = len(MESSAGES)
    top = 86.0
    bottom = 12.5
    dy = (top - bottom) / (n_msg - 1)
    ys = [top - i * dy for i in range(n_msg)]

    fig, ax = plt.subplots(figsize=(10.6, 11.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 98)
    ax.axis("off")

    band_left, band_right = 9.8, 97.5
    i = 0
    while i < n_msg:
        ph_idx, ph_lab = _phase_of(MESSAGES[i][0])
        j = i
        while j + 1 < n_msg and _phase_of(MESSAGES[j + 1][0])[0] == ph_idx:
            j += 1
        y_top = ys[i] + dy * 0.62
        y_bot = ys[j] - dy * 0.62
        ax.add_patch(Rectangle((band_left, y_bot), band_right - band_left,
            y_top - y_bot, facecolor=PHASE_TINT[ph_idx % 2], edgecolor="none", zorder=0))
        # horizontal phase label in the left gutter (two lines: letter + word)
        ax.text(5.0, (y_top + y_bot) / 2, ph_lab, ha="center", va="center",
            fontsize=8.6, color="#33558a", fontweight="bold", linespacing=1.25,
            zorder=1)
        i = j + 1
    # separator between phase-label gutter and the diagram
    ax.plot([9.4, 9.4], [ys[-1] - dy * 0.62, ys[0] + dy * 0.62],
            color="#cfd8e6", lw=1.0, zorder=1)

    head_y = 93.0
    act_w = 1.5
    for name in LIFELINES:
        x = col_x[name]
        ax.add_patch(FancyBboxPatch((x - 8.0, head_y - 3.6), 16, 5.4,
            boxstyle="round,pad=0,rounding_size=0.7", linewidth=1.4,
            edgecolor=BORDER, facecolor=HEADER, zorder=4))
        ax.text(x, head_y - 0.9, LIFELINE_LABELS[name], ha="center", va="center",
            color=HEADERTX, fontsize=9.8, fontweight="bold", zorder=5)
        ax.plot([x, x], [head_y - 3.8, bottom - 2.5], color=MUTED, lw=1.0,
            ls=(0, (4, 3)), zorder=2)

    mx = col_x["Market"]
    ax.add_patch(Rectangle((mx - act_w / 2, bottom - 1.5), act_w,
        (top - bottom) + 3.0, facecolor="#dfe6f1", edgecolor=BORDER, lw=0.8, zorder=3))

    for idx, (step, comp, frm, to, text, kind) in enumerate(MESSAGES):
        y = ys[idx]
        col = COMP.get(comp, MUTED)
        xf, xt = col_x[frm], col_x[to]

        ax.add_patch(FancyBboxPatch((10.6, y - 1.1), 3.2, 2.2,
            boxstyle="round,pad=0,rounding_size=0.3", linewidth=0, facecolor=col, zorder=5))
        ax.text(12.2, y, step, ha="center", va="center", color="white",
            fontsize=8.6, fontweight="bold", zorder=6)
        ax.text(14.5, y, comp, ha="left", va="center", color=col,
            fontsize=8.0, fontweight="bold", zorder=6)

        if kind == "self":
            x = xf
            ax.add_patch(Rectangle((x + act_w / 2, y - 0.55), 2.2, 1.1,
                facecolor="none", edgecolor=col, lw=1.3, zorder=5))
            ax.add_patch(FancyArrowPatch((x + act_w / 2 + 2.2, y - 0.55),
                (x + act_w / 2, y - 0.55), arrowstyle="-|>", mutation_scale=10,
                lw=1.3, color=col, zorder=5))
            ax.text(x + act_w / 2 + 3.2, y, text, ha="left", va="center",
                fontsize=8.8, color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.95))
        else:
            if to != "Market":
                ax.add_patch(Rectangle((xt - act_w / 2, y - 1.1), act_w, 2.2,
                    facecolor="#dfe6f1", edgecolor=BORDER, lw=0.7, zorder=3))
            sgn = 1 if xt > xf else -1
            sx = xf + sgn * (act_w / 2)
            ex = xt - sgn * (act_w / 2)
            ls = "-" if kind == "call" else (0, (3.5, 2.5))
            ax.add_patch(FancyArrowPatch((sx, y), (ex, y), arrowstyle="-|>",
                mutation_scale=13, lw=1.5 if kind == "call" else 1.15,
                linestyle=ls, color=col, zorder=5))
            midx = (sx + ex) / 2
            ax.text(midx, y + 0.95, text, ha="center", va="bottom", fontsize=8.8,
                color=INK, fontweight="medium", zorder=6,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.95))

    fb_top = bottom - 3.2
    ax.add_patch(FancyBboxPatch((4, fb_top - 5.6), 92, 7.0,
        boxstyle="round,pad=0,rounding_size=0.6", linewidth=1.4,
        edgecolor=ACCENT, facecolor="#fdeff0", zorder=3))
    ax.text(50, fb_top - 0.5, "Macro-feedback loop  (M1, lagged one tick)",
        ha="center", va="center", fontsize=8.8, color=ACCENT, fontweight="bold", zorder=5)
    ax.text(50, fb_top - 2.7,
        "automation up  ->  employment down & wages down  ->  wage bill Y down  ->  "
        "demand A down (next tick)",
        ha="center", va="center", fontsize=8.2, color="#7a1419", zorder=5)
    ax.text(50, fb_top - 4.3,
        "->  price p down  ->  profitability gate closes  ->  expansion & adoption restrained",
        ha="center", va="center", fontsize=8.2, color="#7a1419", zorder=5)

    lx, ly = 6, 96.4
    ax.add_patch(FancyArrowPatch((lx, ly), (lx + 5, ly), arrowstyle="-|>",
        mutation_scale=13, lw=1.5, color=INK))
    ax.text(lx + 5.7, ly, "synchronous call / information", ha="left", va="center",
        fontsize=7.6, color=INK)
    ax.add_patch(FancyArrowPatch((lx + 33, ly), (lx + 38, ly), arrowstyle="-|>",
        mutation_scale=13, lw=1.15, linestyle=(0, (3.5, 2.5)), color=INK))
    ax.text(lx + 38.7, ly, "return / result", ha="left", va="center", fontsize=7.6, color=INK)
    ax.add_patch(Rectangle((lx + 55, ly - 0.9), 1.5, 1.8, facecolor="#dfe6f1",
        edgecolor=BORDER, lw=0.7))
    ax.text(lx + 57.2, ly, "activation (entity busy)", ha="left", va="center",
        fontsize=7.6, color=INK)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(path_png, dpi=300)
    fig.savefig(path_pdf)
    plt.close(fig)


if __name__ == "__main__":
    # Figures are stored alongside the thesis sources (scripts/ -> ../thesis_overleaf/figures/).
    outdir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "thesis_overleaf", "figures"))
    os.makedirs(outdir, exist_ok=True)
    draw_figure_31(os.path.join(outdir, "figure_3_1_uml_class_diagram.png"),
                   os.path.join(outdir, "figure_3_1_uml_class_diagram.pdf"))
    draw_figure_32(os.path.join(outdir, "figure_3_2_sequence_tick.png"),
                   os.path.join(outdir, "figure_3_2_sequence_tick.pdf"))
    print("Done: figure_3_1 and figure_3_2 (PNG + PDF)")
