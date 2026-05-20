#!/usr/bin/env python3
"""Compatibility façade for the partitioned Dreth Causal v28/9 package.

This file keeps the original executable/import surface intact.
"""
from dreth.functions import *
from dreth.world import HiddenMutation, CausalWorld
from dreth.ledger import (
    DEFAULT_TOLERANCE, values_match, NoiseEnvelope, Compression,
    TemporalTrassEntry, VarNethra, NoveltyNethra, ChainedLedger,
)
from dreth.fit import (
    predict_var, score_var_hypothesis, enumerate_var_hypotheses,
    enumerate_var_hypotheses_restricted, _func_apply_batch,
    score_hypotheses_batched, fit_var,
)
from dreth.sentinels import select_var_sentinels, check_var_sentinels_with_envelope
from dreth.records import CycleRecord, FitDiagnostic
from dreth.agent import ChainedAgent
from dreth.baseline import RefitBaseline
from dreth.cli import parse_args, run

if __name__ == "__main__":
    run()
