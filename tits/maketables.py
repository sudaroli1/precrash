"""
Emit the two Nexar tables as LaTeX, straight from the result CSVs.

Typing numbers into a manuscript by hand is how a paper ends up disagreeing
with its own code. These are generated, so the only way the tables can be wrong
is if the results are.

  python3 maketables.py
"""
from __future__ import annotations

import pathlib
import pandas as pd

HERE = pathlib.Path(__file__).parent

LABEL = {
    "constant_0.51": r"constant $0.51$",
    "constant_0.99": r"constant $0.99$",
    "never_crosses_0.49": r"never crosses ($0.49$)",
    "ramp_0_to_1": r"linear ramp $0\!\to\!1$",
    "step_at_50pct": r"step at mid-clip",
    "cross_then_drop": r"crosses, then drops",
    "cross_then_dip": r"crosses, dips, recovers",
    "metadata_only": r"metadata only\tnote{a}",
    "sighted_causal": r"risk model (reads video)",
}
BLIND_ORDER = ["constant_0.51", "constant_0.99", "ramp_0_to_1",
               "step_at_50pct", "cross_then_dip", "cross_then_drop",
               "never_crosses_0.49"]
HELD_ORDER = ["constant_0.51", "ramp_0_to_1", "step_at_50pct",
              "never_crosses_0.49", "metadata_only", "sighted_causal"]


def tbl_blind():
    d = pd.read_csv(HERE / "controls_scores.csv").set_index("submission")
    rows = []
    for k in BLIND_ORDER:
        r = d.loc[k]
        rows.append(f"{LABEL[k]} & {r.AP:.3f} & {r.AUC:.3f} & "
                    f"{r.TTA_s:.2f} & {r.STTA_s:.2f} \\\\")
    return r"""\begin{table}[t]
\caption{The video-blind family scored on all 1,500 labelled Nexar clips,
with earliness measured to the \emph{annotated} collision. A constant above
threshold takes 99.9\% of the \SI{19.10}{\second} the corpus admits while
scoring exactly chance on both discrimination terms.}
\label{tab:nexar-blind}
\centering
\begin{tabular}{@{}lcccc@{}}
\toprule
submission & AP & AUC & TTA@0.5 & STTA@0.5 \\
 &  &  & (s) & (s) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""


def tbl_held():
    d = pd.read_csv(HERE / "test_scores.csv").set_index("submission")
    rows = []
    for k in HELD_ORDER:
        r = d.loc[k]
        rows.append(
            f"{LABEL[k]} & {r.AUC:.3f} & {r.TTA_s:.2f} & "
            f"{r['bounded']:.3f} & {r['nTTA@FPR10']:.3f} & "
            f"{r['alarm_min_per_h']:.1f} & "
            f"\\textbf{{{r['coverage@duty10']:.3f}}} \\\\")
    return r"""\begin{table*}[t]
\caption{All submissions on the same 300 held-out clips. Under the published
metric (columns 2--3) the only submission that discriminates earns
\SI{0.04}{\second} of earliness and the one that reads nothing earns
\SI{18.86}{\second}. Bounding leaves the constant at its maximum, exactly as
Section~\ref{sec:protocol} predicts; conditioning removes it but is beaten by
the metadata-only control. Only the frame-level statistic (last column) orders
the submissions by what they can actually see.}
\label{tab:nexar-held}
\centering
\begin{tabular}{@{}lcc cc cc@{}}
\toprule
& \multicolumn{2}{c}{published metric}
& \multicolumn{2}{c}{corrected protocol}
& \multicolumn{2}{c}{frame-level alarm} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
submission & AUC & TTA@0.5 (s) & bounded & conditioned & alarm & coverage \\
 &  &  & @0.5 & @FPR 10\% & (min/h) & @duty 10\% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\item[a] Given the video file's frame rate and duration. It never decodes a
frame.
\end{tablenotes}
\end{table*}"""


if __name__ == "__main__":
    (HERE / "tab_nexar.tex").write_text(tbl_blind() + "\n\n" + tbl_held() + "\n")
    print("wrote tab_nexar.tex")
    print(tbl_blind())
    print()
    print(tbl_held())
