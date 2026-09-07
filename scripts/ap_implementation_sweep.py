"""Do torchmetrics and sklearn compute the same Average Precision on data like ours?

WHY THIS EXISTS (administrative/EXPERIMENTS.md, 2026-08-26 "Library version drift"):
six independently-configured arms all land 0.04-0.08 AP below the published numbers. A uniform
offset across every arm is the signature of something in the *measurement*, not the models.
requirements.txt pins torchmetrics 1.4.0; the environment runs 1.9.0.

WHY SYNTHETIC DATA IS ENOUGH: AP is a pure function of (scores, labels). The question here is
"do two implementations agree on this array?", NOT "what is A0's true AP". So any array with the
same statistical character -- 0.16% positives, large N, realistic tie density -- answers it, and
we skip a multi-gigabyte prediction dump entirely.

THE THREE-WAY COMPARISON, and why it is three and not two:
BaseModel.test_step calls self.test_avg_precision(y_hat, y) with RAW LOGITS
(src/models/BaseModel.py:274). torchmetrics applies sigmoid internally. Sigmoid is monotonic, so
it cannot change the ranking -- EXCEPT where it saturates, and float32 sigmoid saturates hard
past about |logit| > 17. Saturation collapses distinct logits onto one float, manufacturing ties
that did not exist in the input. So we compute:

    tm(logits)             <- exactly what the pipeline computes
    sk(logits)             <- reference definition on the raw scores
    sk(sigmoid(logits))    <- reference definition after the same transform tm applies

If tm(logits) != sk(logits) but tm(logits) == sk(sigmoid(logits)), the culprit is sigmoid
saturation, not the AP algorithm. Two comparisons could not have told those apart.

Usage:  python scripts/ap_implementation_sweep.py [--max-n 1e7] [--seed 0] [--shift 4.0]
"""
import argparse

import numpy as np
import torch
import torchmetrics
from sklearn.metrics import average_precision_score

POS_RATE = 0.0016  # the project's fire-pixel rate; held FIXED in every cell on purpose


def make_data(n, pos_rate, shift, rng):
    """Labels ~ Bernoulli(pos_rate); logits shifted by `shift` for positives.

    `shift` sets how separable the classes are, i.e. roughly what AP a correct metric would
    report. It is tuned to land near the project's real AP (~0.4) so the comparison happens in
    the regime we actually care about, not at chance.
    """
    y = (rng.random(n) < pos_rate).astype(np.int64)
    x = rng.standard_normal(n).astype(np.float32)
    x[y == 1] += shift
    return x, y


def quantize(x, n_levels):
    """Collapse scores onto `n_levels` distinct values -> controls tie density."""
    if n_levels is None:
        return x
    lo, hi = x.min(), x.max()
    step = (hi - lo) / n_levels
    return (np.round((x - lo) / step) * step + lo).astype(np.float32)


def tie_density(x):
    """Fraction of elements whose value is shared with at least one other element."""
    return 1.0 - len(np.unique(x)) / len(x)


def ap_torchmetrics(scores, labels):
    m = torchmetrics.AveragePrecision(task="binary")
    return float(m(torch.from_numpy(scores), torch.from_numpy(labels)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-n", type=float, default=1e7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shift", type=float, default=4.0)
    args = p.parse_args()

    print(f"torch {torch.__version__} | torchmetrics {torchmetrics.__version__}")
    print(f"pos_rate={POS_RATE}  shift={args.shift}  seed={args.seed}\n")

    ns = [int(x) for x in (1e4, 1e5, 1e6, 1e7) if x <= args.max_n]
    levels = [None, 1_000_000, 10_000, 100]

    hdr = (f"{'N':>10} {'levels':>9} {'tie(log)':>9} {'tie(sig)':>9} "
           f"{'tm(logit)':>10} {'sk(logit)':>10} {'sk(sigm)':>10} "
           f"{'tm-sk_log':>11} {'tm-sk_sig':>11}")
    print(hdr)
    print("-" * len(hdr))

    for n in ns:
        for lv in levels:
            rng = np.random.default_rng(args.seed)
            x, y = make_data(n, POS_RATE, args.shift, rng)
            x = quantize(x, lv)
            if y.sum() == 0:
                print(f"{n:>10} {str(lv):>9}   (no positives in this draw -- skipped)")
                continue
            sig = 1.0 / (1.0 + np.exp(-x.astype(np.float32)))
            a_tm = ap_torchmetrics(x, y)
            a_sk = average_precision_score(y, x)
            a_ss = average_precision_score(y, sig)
            print(f"{n:>10} {str(lv):>9} {tie_density(x):>9.5f} {tie_density(sig):>9.5f} "
                  f"{a_tm:>10.6f} {a_sk:>10.6f} {a_ss:>10.6f} "
                  f"{a_tm - a_sk:>+11.2e} {a_tm - a_ss:>+11.2e}")
        print()

    # --- second mechanism: saturation ties, which is how the REAL data gets its ties ---
    # Uniform quantization spreads ties evenly across the score range. Real background pixels
    # pile up at one extreme instead. Those are different distributions of tie-BLOCK SIZES, and
    # tie handling is sensitive to block size, so the two mechanisms are worth testing apart.
    print("saturation-tie check (bottom `clip_frac` of scores collapsed to one value), N=1e6")
    print(f"{'clip_frac':>10} {'tie(log)':>9} {'tm(logit)':>10} {'sk(logit)':>10} {'diff':>11}")
    n = 1_000_000
    for clip_frac in (0.0, 0.5, 0.9, 0.99):
        rng = np.random.default_rng(args.seed)
        x, y = make_data(n, POS_RATE, args.shift, rng)
        if clip_frac > 0:
            x = np.maximum(x, np.quantile(x, clip_frac)).astype(np.float32)
        a_tm = ap_torchmetrics(x, y)
        a_sk = average_precision_score(y, x)
        print(f"{clip_frac:>10.2f} {tie_density(x):>9.5f} {a_tm:>10.6f} {a_sk:>10.6f} "
              f"{a_tm - a_sk:>+11.2e}")


# ---------------------------------------------------------------------------------------
# TODO (yours -- answer these BEFORE reading the numbers, and write the answers into
# EXPERIMENTS.md so they are on the record as predictions rather than reconstructions):
#
# TODO 1 -- PREDICT FIRST. Which cell, if any, do you expect to show a nonzero difference, and
#   which of the four causes (accumulation precision / tie handling / different definition /
#   internal size switch) would that pattern imply? Committing a prediction before looking is
#   what stops this from becoming a post-hoc story.

#   I wouldn't expect the internal size switch to be a problem since the dataset is presumably unmodified compared to WSTS+'s work.
#   I also wouldn't expect accumulation precision to become a new problem how this dataset is processed (?)
#   The way you described it, different definition seems extremely unlikely in terms of something like switching to a trapezoidal definition because of how obviously drastic that change is, but a different definition seems the most likely if there's some sort of error. But then again, how can such an error not be caught across all torchmetrics users?
#   Tie handling also seems like a plausbile mistake worth checking because it's a problem with AP specifically and with highly imbalanced datasets, but it's still somewhat questionable how something like that could slip through
#   Are these thoughts good enough as a prediction? I don't predict that any of these would happen, really. Is simply avoiding making new hypotheses after these are checked good enough to not add an post-hoc contamination?

# TODO 2 -- SET THE THRESHOLD THAT MATTERS. The gap you are chasing is 0.040 AP. What size of
#   tm-vs-sk difference would you count as "this explains the gap"? As "this explains part of
#   it"? As "irrelevant"? Decide those three numbers now, not after seeing the output.\

#   <0.02 is irrelevant, between 0.02 and 0.03 explains part of it and between 0.3 and 0.04 I would be satisfied

#
# TODO 3 -- WHICH TIE MECHANISM IS REAL? The grid uses uniform quantization; the second table
#   uses low-end saturation. Your real ties come from sigmoid saturating on confident background
#   pixels. Which table is the honest model of your data, and what does that imply about which
#   rows you should be reading? Then verify it rather than assuming: dump real logits and
#   measure their tie density (scripts/dump_test_predictions.py).

#   I think the uniform quantization (aka sigmoid of -30 is 0, right?) is more real, because even with tons of pixels with super negative logits, the vegetation information and general unpredicatbility of fire spread, especially through embers, creates lots of variation in exactly how unconfident the model is.
#   So I shouldn't only look at the high N because it's not just about saturation. But which rows should I be reading?

# TODO 4 -- THE SYNTHETIC DATA IS I.I.D.; YOURS IS NOT. Real fire pixels are spatially clustered,
#   so neighbouring scores are strongly correlated. Does that break this test? Careful: separate
#   (a) does correlation change the AP *value*, from (b) does it change whether two
#   implementations *agree* on the same array. Those are not the same question, and only one of
#   them threatens the conclusion.

# (a) yes and (b) no. The test is safe

# TODO 5 -- SCOPE OF THE CONCLUSION. Suppose every cell comes back at exactly 0.000000. What have
#   you ruled out -- torchmetrics 1.9 vs the reference definition, or 1.9 vs 1.4? Under what
#   assumption does the first imply the second, and is that assumption safe enough to close the
#   suspect out, or does it still need the throwaway 1.4.0 environment?
# ---------------------------------------------------------------------------------------

#   Are we assuming that the reference definition has to be right? What is the reference definition, exactly? We can't rule out 1.9 by only running this script in 1.4. The first implies the second under the assumption that the differences between 1.9 and 1.4 are negligibe for this task. I think that if no difference is found between the reference definition and 1.9, it's worth checking 1.4 just to be safe since I am trying to find a difference, not just run a fair test on how hidden the difference is. Even if it's a little post hoc, I'm okay with a false positive rate since it's easily checked with a new trianing run. Is the procedure I'm describing post-hoc?
#   Also I'm in the 1.4 environment right now
if __name__ == "__main__":
    main()
