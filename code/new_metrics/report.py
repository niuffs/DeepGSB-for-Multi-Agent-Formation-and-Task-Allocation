"""Write out/results.xlsx and the two figures (fig_swarm.png, fig_radius.png).

Trajectories are a regenerable cache: if a file under runs/ is missing it is
generated on the fly, so the whole pipeline runs from a clean checkout.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

import metrics as M
import rollout as RO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS, OUT = os.path.join(HERE, "runs"), os.path.join(HERE, "out")
os.makedirs(RUNS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

DIMS, SEEDS = [20, 50], [42, 43, 44, 45, 46]
NS = [20, 50, 100, 200, 500]
R_GRID = [0.001, 0.0015, 0.002, 0.003, 0.005, 0.008]
ALGOS = [("deepgsb", "DeepGSB"), ("mfgnet", "MFG-Net"), ("pinn", "PINN")]
COL = {"deepgsb": "#2196F3", "mfgnet": "#FF9800", "pinn": "#9C27B0"}
R, EPS = M.R, M.EPS

# metrics sheet: (key, header, decimals, percent?)
COLS = [("min_dist", "Min distance", 5, False), ("path_len", "Path length", 3, False),
        ("destroyed", "Destroyed", 1, False), ("dead_by_agent", "Killed by agent", 1, False),
        ("dead_by_obstacle", "Killed by obstacle", 1, False),
        ("survival", "Survival rate/%", 1, True), ("success", "Success rate/%", 1, True)]

TERMS = ["C_run", "C_cong", "C_obs", "C_term", "C_term_KL", "C_total"]
COST_GLOB = {
    "deepgsb": "DeepGSB/Experiments/*_d{}_*/deepgsb_eval/*cost_summary.csv",
    "mfgnet": "MFG-Net/Experiments/*_d{}_*/mfg_eval/*cost_summary.csv",
    "pinn": "PINN/Experiments/*_d{}_*/pinn_eval/*cost_summary.csv",
}

plt.rcParams["font.family"] = "DejaVu Sans"


def runs_for(dim, algo, n):
    return [os.path.join(RUNS, f"traj_{algo}_d{dim}_N{n}_s{s}.npz") for s in SEEDS]


_MODELS = {}


def ensure(paths):
    """Generate any missing trajectory on the fly (runs/ is a cache)."""
    for p in paths:
        if os.path.exists(p):
            continue
        algo, dim, n, seed = re.match(r"traj_(\w+)_d(\d+)_N(\d+)_s(\d+)", os.path.basename(p)).groups()
        key = (algo, int(dim))
        if key not in _MODELS:                # load each checkpoint once
            _MODELS[key] = RO.build(algo, int(dim))
        f = _MODELS[key]
        tr, u = f(RO.init_x(int(dim), int(n), int(seed)), RO.STEPS)
        np.savez_compressed(p, traj=tr.astype(np.float32), u=u.astype(np.float32),
                            algo=algo, dim=int(dim), n=int(n), seed=int(seed))
        print(f"  generated {os.path.basename(p)}", flush=True)


def mean_std(recs, key, pct=False):
    v = np.array([r[key] for r in recs], float) * (100 if pct else 1)
    return v.mean(), (v.std(ddof=1) if v.size > 1 else 0.0)


def cell(recs, key, nd, pct):
    m, s = mean_std(recs, key, pct)
    return f"{m:.{nd}f} ± {s:.{nd}f}"


# ---- data ------------------------------------------------------------------

def collect(ns, rs=(R,)):
    """recs[(dim, algo, n, r)] -> list over seeds"""
    out = {}
    for dim in DIMS:
        for algo, _ in ALGOS:
            for n in ns:
                paths = runs_for(dim, algo, n)
                ensure(paths)
                for p in paths:
                    c = M.cache(p)                    # reuse nn/od across r
                    for r in rs:
                        out.setdefault((dim, algo, n, r), []).append(M.evaluate(p, r=r, cached=c))
    return out


def read_costs():
    """C_total and terms (mean, std) recorded by each project's own evaluation.

    PINN's record stores C_cong without the 1/d normalisation used by the other
    two, so it is divided by d here and C_total is recomputed consistently.
    """
    got = {}
    for algo, _ in ALGOS:
        for dim in DIMS:
            hits = glob.glob(os.path.join(ROOT, COST_GLOB[algo].format(dim)))
            if not hits:
                continue
            with open(hits[0]) as fh:
                for line in fh:
                    m = re.match(r"#\s*(C_\w+),\s*([-\d.e+]+),\s*([-\d.e+]+)", line)
                    if m and m.group(1) in TERMS:
                        got[(algo, dim, m.group(1))] = (float(m.group(2)), float(m.group(3)))
    rows = []
    for algo, _ in ALGOS:
        for dim in DIMS:
            if (algo, dim, "C_run") not in got:
                continue
            g = lambda t: got.get((algo, dim, t),
                                  got.get((algo, dim, "C_term_KL"), (0.0, 0.0)) if t == "C_term"
                                  else (0.0, 0.0))
            cong_rec = g("C_cong")
            cong = ((cong_rec[0] / dim, cong_rec[1] / dim) if algo == "pinn" else cong_rec)
            rec_total = g("C_total")             # shift the mean by the C_cong correction
            total = (rec_total[0] - cong_rec[0] + cong[0], rec_total[1])
            for term, val in [("C_run", g("C_run")), ("C_cong", cong), ("C_obs", g("C_obs")),
                              ("C_term", g("C_term")), ("C_total", total)]:
                rows.append([algo, dim, term, val[0], val[1]])
    return rows


# ---- figures ---------------------------------------------------------------

SPEC = [("survival", "Survival rate / %"), ("success", "Success rate / %")]


def figure(recs, xvals, xlabel, fname, title, key_of, rot=0):
    fig, axes = plt.subplots(len(DIMS), 2, figsize=(10.2, 7.2))
    for i, dim in enumerate(DIMS):
        for j, (key, ylab) in enumerate(SPEC):
            ax = axes[i][j]
            for algo, lab in ALGOS:
                m, s = zip(*[mean_std(recs[key_of(dim, algo, v)], key, pct=True) for v in xvals])
                m, s = np.array(m), np.array(s)
                ax.plot(xvals, m, "o-", color=COL[algo], label=lab, lw=2.0, ms=5)
                ax.fill_between(xvals, m - s, m + s, color=COL[algo], alpha=0.22, lw=0)
            ax.set_xscale("log")
            ax.set_xticks(xvals)
            ax.set_xticklabels([f"{v:g}" for v in xvals], fontsize=12,
                               rotation=rot, ha="right" if rot else "center")
            ax.tick_params(axis="y", labelsize=12)
            ax.minorticks_off()
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylab, fontsize=14)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=12)
        ys = axes[i][0].get_position()
        fig.text(0.012, (ys.y0 + ys.y1) / 2, f"$d$={dim}", rotation=90,
                 va="center", ha="center", fontsize=15)
    fig.suptitle(title, fontsize=17, y=0.965)
    fig.tight_layout(rect=[0.025, 0.01, 1, 0.925])
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        os.remove(path)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---- excel -----------------------------------------------------------------

def write_xlsx(recs_n, loss_rows):
    wb = xlsxwriter.Workbook(os.path.join(OUT, "results.xlsx"))
    head = wb.add_format({"bold": True, "bg_color": "#DDEBF7", "border": 1, "align": "center"})

    ws = wb.add_worksheet("metrics")
    headers = ["d", "N", "Method"] + [h for _, h, _, _ in COLS]
    for c, h in enumerate(headers):
        ws.write(0, c, h, head)
    row = 1
    for dim in DIMS:
        for n in NS:
            for algo, lab in ALGOS:
                ws.write(row, 0, dim)
                ws.write(row, 1, n)
                ws.write(row, 2, lab)
                for c, (key, _, nd, pct) in enumerate(COLS, start=3):
                    ws.write(row, c, cell(recs_n[(dim, algo, n, R)], key, nd, pct))
                row += 1
    ws.set_column(0, 1, 6)
    ws.set_column(2, 2, 12)
    ws.set_column(3, len(headers) - 1, 22)

    ws2 = wb.add_worksheet("losses")
    for c, h in enumerate(["Method", "d", "Loss term", "mean", "std"]):
        ws2.write(0, c, h, head)
    for i, r in enumerate(loss_rows, start=1):
        for c in range(3):
            ws2.write(i, c, r[c])
        ws2.write_number(i, 3, r[3])
        ws2.write_number(i, 4, r[4])
    ws2.set_column(0, 0, 12)
    ws2.set_column(1, 1, 6)
    ws2.set_column(2, 2, 12)
    ws2.set_column(3, 4, 16)

    wb.close()


def main():
    recs_n = collect(NS)
    recs_r = collect([500], rs=R_GRID)
    figure(recs_n, NS, "N", "fig_swarm.png",
           f"Hyper parameter: $r$={R}, $\\epsilon$={EPS}",
           lambda d, a, v: (d, a, v, R))
    figure(recs_r, R_GRID, "$r$", "fig_radius.png",
           f"Hyper parameter: $N$=500, $\\epsilon$={EPS}",
           lambda d, a, v: (d, a, 500, v), rot=30)
    write_xlsx(recs_n, read_costs())
    print("done ->", OUT)


if __name__ == "__main__":
    main()
