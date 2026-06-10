"""
mind_imp_processor.py

A MIND processor variant that follows the **original MIND split** while
preserving *impression* boundaries, so that evaluation can be performed at the
impression level (one behaviour-log line = one impression).

Differences from the base :class:`MINDProcessor`:

    • ``train`` split  = the *full* ``train/behaviors.tsv``
      (no 10% user hold-out for validation),
    • ``valid`` split  = ``dev/behaviors.tsv`` (the full official dev set),
    • ``test``  split  = ``dev/behaviors.tsv`` as well, so that both the
      in-training dev evaluation and the post-hoc ``tester`` evaluate on the
      original dev set,
    • each interaction row carries its impression id (``IMP_COL``) so the
      candidate list shown together in one impression stays groupable. The
      matching data config sets ``group_col: imp`` so that the ranking metrics
      (GAUC / MRR / NDCG) are averaged **per impression**.

This is the dataset used by the popularity-stratified evaluator
(``tester_popularity.py``).
"""

import os
from typing import cast

import pandas as pd
from unitok import UniTok, EntityTokenizer, BertTokenizer

from processor.base_processor import Interactions
from processor.mind_processor import MINDProcessor


class MindImpProcessor(MINDProcessor):
    """MIND with original train/dev split + impression ids preserved."""

    IMP_COL = "imp"  # column holding the impression identifier

    # ------------------------------------------------------------------ #
    # Item tokenization: BERT only (skip Llama + GloVe)                   #
    # ------------------------------------------------------------------ #
    def config_item_tokenization(self):
        """Tokenise item text with BERT only.

        The base :class:`MINDProcessor` also registers Llama-1 (needs a large
        tokenizer download) and GloVe tokenizers; here we keep just BERT to
        speed up processing and avoid the extra dependencies. The categorical
        (sub)category entity tokenizers are kept for the ``category`` input.
        """
        self.add_item_tokenizer(BertTokenizer(vocab="bert"))

        self.item.add_feature(
            tokenizer=EntityTokenizer(vocab="category"),
            column="category",
        )
        self.item.add_feature(
            tokenizer=EntityTokenizer(vocab="subcategory"),
            column="subcategory",
        )

    # ------------------------------------------------------------------ #
    # Extra impression-id feature for the interaction UniTok             #
    # ------------------------------------------------------------------ #
    def config_inter_tokenization(self, ut: UniTok):
        """Register the impression id as an entity feature so it can be used
        as the evaluation ``group_col``."""
        ut.add_feature(
            tokenizer=EntityTokenizer(vocab="imp_id"),
            column=self.IMP_COL,
        )

    # ------------------------------------------------------------------ #
    # Interaction loader keeping the impression id                       #
    # ------------------------------------------------------------------ #
    def _load_interactions(self, path: str) -> pd.DataFrame:
        """Parse a ``behaviors.tsv`` file into one ``(imp, uid, nid, click)``
        row per candidate news (the ``predict`` column is exploded)."""
        user_set = set(self.user_df[self.UID_COL].unique())

        interactions = pd.read_csv(
            filepath_or_buffer=cast(str, path),
            sep="\t",
            names=[self.IMP_COL, self.UID_COL, "time", self.HIS_COL, "predict"],
            usecols=[self.IMP_COL, self.UID_COL, "predict"],
        )
        interactions = interactions[interactions[self.UID_COL].isin(user_set)]

        interactions["predict"] = interactions["predict"].str.split().apply(
            lambda lst: [token.split("-") for token in lst]
        )
        interactions = interactions.explode("predict")
        interactions[[self.IID_COL, self.LBL_COL]] = pd.DataFrame(
            interactions["predict"].tolist(), index=interactions.index
        )
        interactions.drop(columns=["predict"], inplace=True)
        interactions[self.LBL_COL] = interactions[self.LBL_COL].astype(int)
        return interactions

    # ------------------------------------------------------------------ #
    # Original split: train = train file, valid = test = full dev file   #
    # ------------------------------------------------------------------ #
    def load_interactions(self) -> Interactions:
        train_df = self._load_interactions(
            os.path.join(self.data_dir, "train", "behaviors.tsv")
        )
        dev_df = self._load_interactions(
            os.path.join(self.data_dir, "dev", "behaviors.tsv")
        )

        train_df = train_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)

        # valid and test are both the full dev set (independent copies so the
        # separate tokenization passes never mutate a shared frame).
        return Interactions(train_df, dev_df.copy(), dev_df.copy())
