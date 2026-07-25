# Dreth

Dreth is an experimental economy of consideration governed by prospective predictive
authority.

A nethra is a scoped operative handle over learned or supplied structure. It binds a
provider, touched structure, an executable use claim, evidence, and repair topology. A
nethra can earn contextual permission to remove actual work. Consequential failure blocks
the exact local dispatch path and opens that same handle for repair. Higher nethras can own
subordinate failure channels and suppress subordinate checks after earning coverage of
those channels.

## Implementation source of truth

[The kernel specification](docs/KERNEL_SPEC.md) defines:

- the exact domain, provider, kernel, and evaluator ownership boundary;
- immutable prospective commitments;
- operation-, action-, target-, context-, revision-, and horizon-specific authority;
- tareth/trass role derivation;
- real work dispatch and accounting;
- harmless mismatch behavior;
- local consequential failure quarantine;
- lazy factorization, relational attribution, and prospective repair promotion;
- recurrent higher handles and channel-specific subordinate suppression;
- module APIs, pseudocode, implementation order, and acceptance tests.

## Current branch status

The current executable is a caller-driven ledger skeleton:

- callers supply `expected`, `role`, `implicated_ids`, and factorization callbacks;
- `observe()` settles every due commitment in a context against one caller-supplied value;
- `can_reuse()` returns a Boolean and changes no work plan;
- failures create caller-directed boundaries and factors;
- consolidation receives caller-selected boundary IDs.

Its tests establish consistency of those ledger rules. The next implementation pass replaces
that causal boundary according to `docs/KERNEL_SPEC.md`.

## Research success

A valid Dreth experiment executes the specified mechanism exactly and reports complete
evidence and cost. Zero work savings, missed failures, poor attribution, and negative utility
remain valid experimental results.

An effective Dreth run removes actual consideration while preserving consequential failure
detection, local repair, and positive net utility.

## Historical mechanism receipt

Conversation history records an accepted 12-run `regime_switch` experiment in which an
earned higher sentinel suppressed lower sentinel work:

```text
n = 8, 12
historical handle amortization = 13.8%
regime sentinel pass = 9,186
regime sentinel fail = 11,214
no_sentinel = 0
runs ok = 12/12
```

The new regression harness will recreate that same-object ablation and report its result.
The pass condition is mechanism fidelity and complete accounting.

## Current executable

```bash
python -m dreth
python -m pytest -q
```

The JSON demo describes the existing ledger skeleton. Implementation work begins with Pass 1
in [TODO.md](TODO.md).
