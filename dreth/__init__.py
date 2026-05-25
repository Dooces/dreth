"""
dreth — streaming causal structure learning with recursive authority records.

Modules:
  functions.py      — operator vocabulary (FUNC_LIBRARY, HIDDEN_FUNC_LIBRARY)
  world.py          — hidden causal world and intervention oracle
  ledger.py         — earned authority state: VarNethra, TiedFrontier, Compression,
                      NoiseEnvelope, ChainedLedger
  fit.py            — full audit: hypothesis enumeration and scoring
  sentinels.py      — sentinel selection and cheap-path validation
  records.py        — diagnostic-only: CycleRecord, FitDiagnostic
  agent.py          — authority-record control loop: ChainedAgent
  baseline.py       — naive refit-everything agent for cost comparison
  cli.py            — CLI entrypoint and schedule runner
"""
