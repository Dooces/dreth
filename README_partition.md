# Dreth Causal v28/9 partitioned copy

This is a mechanical split of `dreth_causal_v28.py` into a small package.

Run it the same way as before:

```bash
python3 dreth_causal_v28.py --seed 7 --n-vars 100 --cycles 2000 --schedule incremental --mode v29-algebraic-only --quiet
```

The file `dreth_causal_v28.py` remains as a compatibility facade so `dreth_extensions.py` can keep importing `Compression` and `FUNC_LIBRARY` from the original module name.

Partition:

- `dreth/functions.py`: scalar function library and hidden SIN gate.
- `dreth/world.py`: `HiddenMutation`, `CausalWorld`.
- `dreth/ledger.py`: tolerance, envelopes, compression records, per-variable state, ledger.
- `dreth/fit.py`: prediction, hypothesis enumeration, batched scoring, fitting.
- `dreth/sentinels.py`: sentinel selection and envelope-checked sentinel validation.
- `dreth/records.py`: diagnostic dataclasses.
- `dreth/agent.py`: `ChainedAgent`, moved intact.
- `dreth/baseline.py`: `RefitBaseline`.
- `dreth/cli.py`: argparse and top-level run loop.
