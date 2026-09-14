"""
Generate figures for accuracy matrix and multi-criteria comparison.

  fig_heatmap.pdf: accuracy matrix across attack families and fractions
  fig_radar.pdf: multi-criteria comparison of the admissible rules
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
RES = HERE.parent / "results"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "font.family": "serif",
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})

METHODS = ["fedavg", "trimmed_mean", "krum", "fltrust", "feddbc", "amfta", "amfta_noq"]
LABEL = dict(zip(METHODS, ["FedAvg", "Trimmed Mean", "Krum", "FLTrust",
                           "FedDBC", "AMFTA", "AMFTA-ND"]))
RHOS = ["0.1", "0.2", "0.3"]
SEEDS = (42, 123, 456)
MAJORITY = 72.58  # majority-class accuracy of the evaluation split


def per_seed(method, byz, attack, key="accuracy", last=5):
    out = {}
    for s in SEEDS:
        for f in reversed(sorted(glob.glob(os.path.join(RES, f"{method}_byz{byz}_{attack}_seed{s}_*.json")))):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if not d:
                continue
            v = [r[key] for r in d if key in r][-last:]
            if v:
                out[s] = sum(v) / len(v) * 100
                break
    return out


def heatmap():
    fig = plt.figure(figsize=(7.3, 3.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05],
                          height_ratios=[1, 0.05], wspace=0.16, hspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[1, 0:2])
    cmap = plt.get_cmap("RdYlGn")

    for ax, attack, name in zip(axes[:2], ["label_flipping", "gaussian_noise"],
                                ["Label flipping", "Gaussian noise"]):
        M = np.array([[st.mean(list(per_seed(m, r, attack).values()) or [np.nan])
                       for r in RHOS] for m in METHODS])
        im = ax.imshow(M, cmap=cmap, vmin=40, vmax=95, aspect="auto")
        for i in range(len(METHODS)):
            for j in range(len(RHOS)):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.2,
                        color="#111111" if v > 52 else "white",
                        fontweight="bold" if v >= 90 else "normal")
        ax.set_xticks(range(len(RHOS)))
        ax.set_xticklabels([f"$\\rho$={float(r):.2f}" for r in RHOS], fontsize=8.2)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([LABEL[m] for m in METHODS] if ax is axes[0] else [],
                           fontsize=8.4)
        ax.set_title(name, fontsize=9.5, pad=6)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    ax = axes[2]
    for i in range(len(METHODS)):
        ax.axhspan(i - 0.5, i + 0.5, color="#f2f2f2" if i % 2 == 0 else "white",
                   zorder=0, lw=0)
    for i, m in enumerate(METHODS):
        for attack, off, mk, col in (("label_flipping", -0.17, "o", "#2f6fae"),
                                     ("gaussian_noise", 0.17, "s", "#c0453a")):
            vals = list(per_seed(m, "0.3", attack).values())
            ax.scatter(vals, [i + off] * len(vals), s=17, marker=mk,
                       facecolor="none", edgecolor=col, linewidth=1.0, zorder=3)
    ax.axvline(MAJORITY, color="#333333", lw=1.0, ls="--", zorder=2)
    ax.text(MAJORITY - 1.6, len(METHODS) - 0.6, "majority class 72.6%",
            rotation=90, fontsize=7.0, color="#333333", ha="right", va="bottom")
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([LABEL[m] for m in METHODS], fontsize=8.4)
    ax.yaxis.tick_right()
    ax.set_ylim(len(METHODS) - 0.5, -0.5)
    ax.set_xlim(18, 102)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xlabel("Accuracy (%) at $\\rho=0.30$", fontsize=8.4)
    ax.set_title("Per-seed values", fontsize=9.5, pad=6)
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis="x", alpha=0.3, lw=0.5, zorder=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.scatter([], [], s=17, marker="o", facecolor="none", edgecolor="#2f6fae",
               label="label flipping")
    ax.scatter([], [], s=17, marker="s", facecolor="none", edgecolor="#c0453a",
               label="Gaussian noise")
    ax.legend(frameon=False, fontsize=7.6, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.20), handletextpad=0.25, columnspacing=1.2)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Mean accuracy over three seeds (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5, length=2)
    cb.outline.set_visible(False)
    fig.savefig(FIG / "fig_heatmap.pdf")
    plt.close(fig)
    print("wrote fig_heatmap.pdf")


def radar():
    shown = ["trimmed_mean", "krum", "feddbc", "amfta", "amfta_noq"]
    worst = {}
    for m in METHODS:
        lf = st.mean(list(per_seed(m, "0.3", "label_flipping").values()))
        gn = st.mean(list(per_seed(m, "0.3", "gaussian_noise").values()))
        worst[m] = min(lf, gn)
    agg100 = {"trimmed_mean": 0.30, "krum": 33.60, "feddbc": 1.86,
              "amfta": 4.41, "amfta_noq": 4.37}
    agg500 = {"trimmed_mean": 1.83, "krum": 836.34, "feddbc": 14.00,
              "amfta": 22.58, "amfta_noq": 22.22}
    drop = {}
    for m in METHODS:
        d = []
        for attack in ("label_flipping", "gaussian_noise"):
            a = st.mean(list(per_seed(m, "0.1", attack).values()))
            b = st.mean(list(per_seed(m, "0.3", attack).values()))
            d.append(a - b)
        drop[m] = max(d)
    no_oracle = {"trimmed_mean": 0.0, "krum": 0.0, "feddbc": 1.0, "amfta": 1.0, "amfta_noq": 1.0}
    no_srvdata = {"trimmed_mean": 1.0, "krum": 1.0, "feddbc": 1.0, "amfta": 0.0, "amfta_noq": 1.0}

    axes_names = ["Worst-case\naccuracy", "Stability\nacross $\\rho$",
                  "Aggregation\nspeed ($N$=100)", "Scaling\nheadroom ($N$=500)",
                  "No oracle /\nno server data"]
    vals = {}
    for m in shown:
        v = [
            np.clip((worst[m] - 40) / (91 - 40), 0, 1),
            np.clip(1 - drop[m] / 55.0, 0, 1),
            np.clip(1 - np.log10(agg100[m] / 0.30) / np.log10(33.60 / 0.30), 0, 1),
            np.clip(1 - np.log10(agg500[m] / 1.83) / np.log10(836.34 / 1.83), 0, 1),
            0.5 * no_oracle[m] + 0.5 * no_srvdata[m],
        ]
        vals[m] = v

    n = len(axes_names)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]
    colors = {"trimmed_mean": "#c44e52", "krum": "#dd8452", "feddbc": "#8172b3",
              "amfta": "#55a868", "amfta_noq": "#1f4e8c"}
    fig, ax = plt.subplots(figsize=(5.0, 4.4), subplot_kw={"polar": True})
    for m in shown:
        v = vals[m] + vals[m][:1]
        lw = 2.0 if m == "amfta_noq" else 1.1
        ax.plot(ang, v, lw=lw, color=colors[m], label=LABEL[m],
                zorder=5 if m == "amfta_noq" else 3)
        if m == "amfta_noq":
            ax.fill(ang, v, color=colors[m], alpha=0.12, zorder=2)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(axes_names, fontsize=7.8)
    for lb, a in zip(ax.get_xticklabels(), ang[:-1]):
        lb.set_horizontalalignment("left" if 0.1 < a < np.pi - 0.1 else
                                   ("right" if a > np.pi + 0.1 else "center"))
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.35, lw=0.5)
    ax.spines["polar"].set_alpha(0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
              frameon=False, fontsize=7.6, columnspacing=1.0)
    fig.savefig(FIG / "fig_radar.pdf")
    plt.close(fig)
    print("wrote fig_radar.pdf")


if __name__ == "__main__":
    heatmap()
    radar()
