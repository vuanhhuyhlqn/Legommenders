"""
tester_popularity.py

Impression-level **popularity-stratified** evaluation for Legommenders.

Motivation
----------
MIND has a long-tailed click distribution: a small head of news items collects
most of the clicks. A single averaged metric hides whether a model is merely
good at re-ranking already-popular items or can also surface tail items the
user actually clicked. This script splits the evaluation by the popularity of
the *clicked* (positive) candidate.

Protocol (follows the original MIND split via the ``mindimp`` dataset)
---------------------------------------------------------------------
1. **Popularity** is computed from the **train** split: count how many times
   each news item was *clicked* (``click == 1``), rank items from most- to
   least-clicked, and walk down accumulating clicks until the running total
   reaches ``pop_mass`` (default 80%) of all clicks. Every item up to and
   including the one that crosses the threshold is **popular** (the head that
   is responsible for 80% of clicks); the rest is **unpopular** (long tail).

2. The model scores every ``(impression, candidate)`` row of the **dev** split
   (the ``test`` UT of ``mindimp`` is the full dev set, grouped by ``imp``).

3. Two variants are evaluated, each grouped per impression:
     * ``popular``   — within every impression keep its *non-clicked*
       candidates plus only the clicked items that are popular,
     * ``unpopular`` — keep the same non-clicked candidates plus only the
       clicked items that are unpopular.
   Impressions left without both a positive and a negative are dropped (a
   ranking metric such as GAUC is undefined otherwise).

4. The configured metrics (GAUC / MRR / NDCG@k) are averaged per impression
   for each variant. A large popular-vs-unpopular gap signals popularity bias.

Usage (after training a model on the ``mindimp`` dataset)::

    python tester_popularity.py \\
        --data config/data/mindimp.yaml \\
        --model config/model/fastformer.yaml \\
        --embed config/embed/bertbase.yaml \\
        --load_sign <signature-of-trained-model> \\
        --hidden_size 256 \\
        --pop_mass 0.8
"""

from __future__ import annotations

import os
from typing import Dict, Sequence, Set

import numpy as np
import pandas as pd
from pigmento import pnt

from loader.env import Env
from tester import Tester
from utils import bars, io
from utils.config_init import CommandInit
from utils.metrics import MetricPool


# =========================================================================== #
#                      Pure helpers (unit-test friendly)                      #
# =========================================================================== #
def pareto_popular_keys(counts: Dict, mass: float = 0.8) -> Set:
    """Return the head of items responsible for ``mass`` fraction of all events.

    ``counts`` maps item -> click count. Items are ranked by count descending;
    we accumulate counts and keep every item up to and including the one whose
    inclusion first pushes the cumulative total to ``>= mass * total``.
    """
    if not counts:
        return set()
    series = pd.Series(counts, dtype="int64").sort_values(ascending=False)
    total = int(series.sum())
    if total <= 0:
        return set()
    threshold = mass * total
    cum = series.cumsum()
    cum_before = cum - series          # cumulative *excluding* the current item
    # An item is popular iff the mass accumulated before it is still under the
    # threshold -> this includes exactly the crossing item and excludes the
    # rest of the tail.
    popular = series.index[cum_before.to_numpy() < threshold]
    return set(popular.tolist())


def variant_keep_masks(labels: np.ndarray, item_is_popular: np.ndarray):
    """Row masks for the popular / unpopular impression variants.

    Both variants keep all non-clicked rows (``label == 0``); they differ only
    in which positives (``label == 1``) are retained.
    """
    clicked = labels == 1
    non_clicked = labels == 0
    popular_keep = non_clicked | (clicked & item_is_popular)
    unpopular_keep = non_clicked | (clicked & ~item_is_popular)
    return popular_keep, unpopular_keep


def valid_group_mask(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Row mask keeping only groups (impressions) that contain BOTH a positive
    and a negative — required for per-group ranking metrics like GAUC."""
    df = pd.DataFrame({"g": groups, "l": labels})
    stat = df.groupby("g")["l"].agg(["min", "max"])
    good = stat.index[(stat["min"] == 0) & (stat["max"] == 1)].to_numpy()
    if good.size == 0:
        return np.zeros(len(labels), dtype=bool)
    return np.isin(groups, good)


# =========================================================================== #
#                              Tester subclass                                #
# =========================================================================== #
class PopularityTester(Tester):
    """Tester that reports popular vs unpopular impression metrics."""

    # ------------------------------------------------------------------ #
    # Train popularity -> set of popular *item-id tokens*                #
    # ------------------------------------------------------------------ #
    def popular_item_tokens(self) -> tuple[Set[int], dict]:
        train_path = os.path.join(self.data.base_dir, "train.parquet")
        df = pd.read_parquet(train_path, columns=["nid", "click"])
        click_counts = df.loc[df["click"] == 1, "nid"].value_counts()

        mass = float(self.config.pop_mass)
        popular_nids = pareto_popular_keys(click_counts.to_dict(), mass=mass)

        # Map raw news ids -> item-id tokens (the space used at eval time).
        vocab = self.manager.item_ut.key_feature.tokenizer.vocab
        o2i = vocab.o2i
        popular_tokens = {int(o2i[n]) for n in popular_nids if n in o2i}

        info = dict(
            mass=mass,
            total_clicks=int(click_counts.sum()),
            clicked_items=int(click_counts.size),
            popular_items=len(popular_nids),
        )
        return popular_tokens, info

    # ------------------------------------------------------------------ #
    # Metrics for one variant                                            #
    # ------------------------------------------------------------------ #
    def _grouped_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
        metrics: Sequence[str],
    ) -> Dict[str, float]:
        keep = valid_group_mask(labels, groups)
        n_impr = int(np.unique(groups[keep]).size) if keep.any() else 0
        if not keep.any():
            out: Dict[str, float] = {m: float("nan") for m in metrics}
            out["#impr"] = 0
            return out

        pool = MetricPool.parse(list(metrics))
        result = pool.calculate(
            scores[keep].tolist(),
            labels[keep].tolist(),
            groups[keep].tolist(),
        )
        out = dict(result)
        out["#impr"] = n_impr
        return out

    # ------------------------------------------------------------------ #
    # Override the standard test() with the stratified evaluation        #
    # ------------------------------------------------------------------ #
    def test(self) -> Dict[str, Dict[str, float]]:
        cm = self.manager.cm
        label_col, group_col, item_col = cm.label_col, cm.group_col, cm.item_col

        popular_tokens, info = self.popular_item_tokens()
        pnt(
            f"train popularity @ {info['mass'] * 100:.0f}% click-mass: "
            f"{info['popular_items']}/{info['clicked_items']} clicked items are popular "
            f"(out of {info['total_clicks']} total clicks)"
        )

        # Run the model once over the dev impressions; collect label/group/item.
        loader = self.manager.get_test_loader()
        score_series, col_series = self.base_evaluate(
            loader,
            cols=[label_col, group_col, item_col],
            bar=bars.TestBar(),
        )

        scores = score_series.numpy()
        labels = col_series[label_col].numpy()
        groups = col_series[group_col].numpy()
        items = col_series[item_col].numpy()

        # Fast popular-token lookup over the item-id space.
        size = int(items.max()) + 1 if items.size else 1
        is_popular = np.zeros(size, dtype=bool)
        if popular_tokens:
            tok = np.fromiter(popular_tokens, dtype=np.int64)
            tok = tok[tok < size]
            is_popular[tok] = True
        item_is_popular = is_popular[items]

        popular_keep, unpopular_keep = variant_keep_masks(labels, item_is_popular)

        metrics = list(self.exp.metrics)
        results = {
            "popular": self._grouped_metrics(
                scores[popular_keep], labels[popular_keep], groups[popular_keep], metrics
            ),
            "unpopular": self._grouped_metrics(
                scores[unpopular_keep], labels[unpopular_keep], groups[unpopular_keep], metrics
            ),
        }

        # Pretty print + persist.
        lines = []
        for variant in ("popular", "unpopular"):
            r = results[variant]
            parts = []
            for key, value in r.items():
                parts.append(f"{key}: {value}" if key == "#impr" else f"{key}: {value:.4f}")
            line = f"[{variant}] " + ", ".join(parts)
            pnt(line)
            lines.append(line)

        io.file_save(Env.ph.result_path, "\n".join(lines))
        return results


# ---------------------------------------------------------------------- #
# CLI entry-point (mirrors tester.py)                                    #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    configuration = CommandInit(
        required_args=["data", "model"],
        default_args=dict(
            exp="config/exp/default.yaml",
            embed="config/embed/null.yaml",
            hidden_size=256,
            item_hidden_size="${hidden_size}$",
            pop_mass=0.8,       # cumulative click-mass threshold for "popular"
            latency=False,
            num_batches=1000,
        ),
    ).parse()

    tester = PopularityTester(config=configuration)
    tester.run()
