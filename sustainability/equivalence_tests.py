"""
Equivalence testing for the degradation claim.

The submitted manuscript operationalised "graceful degradation" as a
non-significant Welch contrast between rho = 0.10 and rho = 0.30. That is a
misuse of a significance test: a large p-value is absence of evidence, not
evidence of absence, and with n = 3 the design has almost no power, so a
method can be declared graceful simply by being noisy. The manuscript already
concedes this for FedAvg under Gaussian noise (the dagger footnote), which
shows the criterion does not do the work asked of it.

This script replaces the criterion with two one-sided tests (TOST). A method
degrades gracefully over [rho_1, rho_2] when the accuracy drop is
*statistically contained* inside a pre-declared practical margin epsilon:

    H01: drop >= epsilon        H02: drop <= -epsilon
    reject both  =>  equivalence established at level alpha

Only means, standard deviations and n are required, all of which the
manuscript already reports, so no re-run is needed to produce this table.

Choice of epsilon
-----------------
epsilon = 5.0 percentage points. Justification stated in the manuscript: at
the reported class balance (54.1% benign) and an operating point of roughly
90% accuracy on a link carrying 1e6 flows/day, 5 pp of accuracy is of the
order of 5e4 additional misclassified flows per day, which is the scale at
which an analyst team's triage capacity is affected. It is declared before
the tests are run and applied uniformly to every method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy import stats

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

EPSILON = 5.0     # practical margin, percentage points
ALPHA = 0.05
N = 3

FALLBACK_DATA = {
    "label flipping": {
        "FedAvg":       ((94.2606, 0.8692), (73.8122, 1.0043)),
        "Trimmed Mean": ((92.0709, 1.5320), (82.9883, 0.8384)),
        "Krum":         ((89.6588, 0.7679), (87.2989, 3.9927)),
        "FLTrust":      ((77.3573, 1.8327), (72.0531, 1.2917)),
        "FedDBC":       ((92.0753, 1.5393), (71.9868, 2.8852)),
        "AMFTA":        ((93.0639, 1.4499), (80.3176, 11.8017)),
        "AMFTA-ND":     ((92.4691, 1.3372), (91.6902, 1.2162)),
    },
    "Gaussian noise": {
        "FedAvg":       ((71.3324, 2.1685), (42.8003, 26.6524)),
        "Trimmed Mean": ((94.4296, 0.9971), (41.4878, 24.3783)),
        "Krum":         ((89.9501, 0.5448), (90.3199, 0.2456)),
        "FLTrust":      ((77.2887, 1.9810), (70.2904, 2.2885)),
        "FedDBC":       ((93.8960, 1.1892), (65.3342, 11.2593)),
        "AMFTA":        ((90.2903, 1.1174), (89.2838, 0.8323)),
        "AMFTA-ND":     ((90.9155, 1.0920), (90.5538, 1.0933)),
    },
}


def load_live_data(results_path: Path | str | None = None) -> dict:
    """Load mean and sample standard deviation (ddof=1) from paper_tables.json."""
    if results_path is None:
        target = Path(__file__).resolve().parent.parent / "results" / "paper_tables.json"
    else:
        target = Path(results_path)
        if target.is_dir():
            target = target / "paper_tables.json"
    if not target.exists():
        target = Path("results/paper_tables.json")

    if not target.exists():
        return FALLBACK_DATA

    with open(target, encoding="utf-8") as f:
        pt = json.load(f)["table"]

    methods = [
        ("FedAvg", "fedavg"),
        ("Trimmed Mean", "trimmed_mean"),
        ("Krum", "krum"),
        ("FLTrust", "fltrust"),
        ("FedDBC", "feddbc"),
        ("AMFTA", "amfta"),
        ("AMFTA-ND", "amfta_noq"),
    ]
    attacks = [
        ("label flipping", "label_flipping"),
        ("Gaussian noise", "gaussian_noise"),
    ]

    data = {}
    for att_disp, att_key in attacks:
        data[att_disp] = {}
        for m_disp, m_key in methods:
            k1 = f"{m_key}|byz0.10|{att_key}"
            k2 = f"{m_key}|byz0.30|{att_key}"
            if k1 not in pt or k2 not in pt:
                continue
            entry1 = pt[k1]
            entry2 = pt[k2]
            n1 = entry1.get("n_seeds", 3)
            n2 = entry2.get("n_seeds", 3)
            s1_factor = math.sqrt(n1 / (n1 - 1)) if n1 > 1 else 1.0
            s2_factor = math.sqrt(n2 / (n2 - 1)) if n2 > 1 else 1.0
            m1 = entry1["acc_mean"] * 100.0
            s1 = entry1["acc_std"] * s1_factor * 100.0
            m2 = entry2["acc_mean"] * 100.0
            s2 = entry2["acc_std"] * s2_factor * 100.0
            data[att_disp][m_disp] = ((m1, s1), (m2, s2))
    return data


DATA = load_live_data()


def welch(m1, s1, n1, m2, s2, n2):
    """Welch t statistic, df and two-sided p for m1 - m2."""
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    if se == 0:
        return float("inf"), n1 + n2 - 2, 0.0, 0.0
    t = (m1 - m2) / se
    df = se ** 4 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, df, p, se


def tost(m1, s1, n1, m2, s2, n2, eps=EPSILON):
    """Two one-sided tests for equivalence of m1 and m2 within +/- eps.

    Returns the larger of the two one-sided p-values, which is the TOST
    p-value: equivalence is established iff it is below alpha.
    """
    diff = m1 - m2
    _, df, _, se = welch(m1, s1, n1, m2, s2, n2)
    if se == 0:
        return diff, 0.0, df, 0.0, 0.0
    t_lo = (diff + eps) / se        # H01: diff <= -eps
    t_hi = (diff - eps) / se        # H02: diff >= +eps
    p_lo = stats.t.sf(t_lo, df)     # want small: diff is above -eps
    p_hi = stats.t.cdf(t_hi, df)    # want small: diff is below +eps
    return diff, max(p_lo, p_hi), df, p_lo, p_hi


def ci90(m1, s1, n1, m2, s2, n2):
    """90% CI on the drop; equivalent decision rule to TOST at alpha=0.05."""
    diff = m1 - m2
    _, df, _, se = welch(m1, s1, n1, m2, s2, n2)
    h = stats.t.ppf(0.95, df) * se
    return diff - h, diff + h


def main():
    parser = argparse.ArgumentParser(description="TOST equivalence testing for degradation claim")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to results directory or paper_tables.json")
    args = parser.parse_args()

    data = load_live_data(args.results) if args.results else DATA

    results = {}
    print(f"Equivalence testing, margin epsilon = {EPSILON} pp, alpha = {ALPHA}, n = {N}")
    print("Drop is accuracy at rho=0.10 minus accuracy at rho=0.30 (positive = degradation).\n")
    for attack, table in data.items():
        print(f"--- {attack} ---")
        print(f"  {'method':<14s}{'drop':>7s}{'90% CI':>18s}"
              f"{'p_NHST':>9s}{'p_TOST':>9s}  verdict")
        results[attack] = {}
        for m, ((m1, s1), (m2, s2)) in table.items():
            _, _, p_nhst, _ = welch(m1, s1, N, m2, s2, N)
            drop, p_tost, df, _, _ = tost(m1, s1, N, m2, s2, N)
            lo, hi = ci90(m1, s1, N, m2, s2, N)
            equivalent = p_tost < ALPHA
            different = p_nhst < ALPHA
            if equivalent and not different:
                verdict = "graceful (equivalent)"
            elif different and not equivalent:
                verdict = "degrades"
            elif equivalent and different:
                verdict = "degrades but trivially"
            else:
                verdict = "INCONCLUSIVE (underpowered)"
            results[attack][m] = {
                "drop": drop, "ci90": [lo, hi], "p_nhst": p_nhst,
                "p_tost": p_tost, "df": df, "verdict": verdict,
            }
            print(f"  {m:<14s}{drop:>7.1f}  [{lo:6.1f}, {hi:6.1f}]"
                  f"{p_nhst:>9.3f}{p_tost:>9.3f}  {verdict}")
        print()

    print("Summary of what changes relative to the submitted criterion:")
    for attack, table in results.items():
        for m, v in table.items():
            if v["verdict"].startswith("INCONCLUSIVE") and v["p_nhst"] >= ALPHA:
                print(f"  {attack:<16s}{m:<14s} was reported 'graceful' on p>0.05;"
                      f" equivalence is NOT established (p_TOST={v['p_tost']:.3f})")

    (OUT / "equivalence.json").write_text(json.dumps(
        {"epsilon": EPSILON, "alpha": ALPHA, "n": N, "results": results}, indent=2))
    print(f"\nwrote {OUT / 'equivalence.json'}")


if __name__ == "__main__":
    main()
