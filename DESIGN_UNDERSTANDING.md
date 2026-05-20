# What I Think The Intended Dreth Design Is

This is my honest account. I am writing it because I have flattened and
patronized this design repeatedly, and the user wants to know whether I
actually understand it or whether I am still pattern-matching to something
simpler. I will state what I think, clearly, including where I am uncertain.

---

## The Observable Mechanism

There is a hidden causal world: N variables, each with a set of parents and a
function (MEAN, SUM, PROD, LOW, HIGH, etc.). The agent cannot see this
structure directly. It can observe states and issue interventions — set a
variable to a value, observe the resulting state of all visible variables.

Each cycle, the agent must produce accurate predictions for each variable at
minimal cost. The world can drift: edges can change, functions can change,
values can shift. The agent must notice drift and recover.

This is not a batch learning problem. The agent runs continuously. Budget is
real. Every intervention costs something. The design question is: how do you
build up a structure of certified beliefs that lets you pay less on future
cycles without accumulating hidden errors?

---

## The Three Dispatch Paths

Each cycle, each variable gets one of:

1. **trass-skip**: This variable's output doesn't propagate to anything the
   operation cares about. Do no work. Not even a sentinel check.

2. **sentinel cheap path**: We have a certified hypothesis (parents, func).
   Run a small set of targeted interventions to verify it still holds. If it
   does, accept the cached prediction without re-fitting.

3. **full audit**: Enumerate the hypothesis space, score all candidates with
   interventional probes, pick the winner, update certification state.

The whole point of the framework is that paths 1 and 2 are only safe because
something earned them. You can only skip a variable if its trass verdict
survived the substitution test. You can only use sentinels if the hypothesis
has been stable across enough consecutive audits. The certification machinery
is what makes the cheap paths legitimate.

---

## What A VarNethra Actually Is

A VarNethra is not a label. It is the current certified state of the agent's
belief about one variable. It actively gates reasoning:

- `operation_role == "tareth"` means: this variable is in the available_parents
  set for downstream fits. Other variables can use it as a parent candidate.
- `operation_role == "trass"` means: this variable is excluded from all
  downstream hypothesis spaces. It contributes nothing to the operation.
- `status == "certified"` with sentinels means: the cheap path is open. No
  full audit until a sentinel fails.

The VarNethra for variable i determines what the agent will even consider when
fitting variable j (j's parents can only be tareth-certified vars). A certified
nethra is an active filter on the hypothesis space of every variable that might
depend on it. Not a description. A filter.

This is the operative part. A nethra is operative because its current state
modifies every downstream inference the agent makes.

---

## What Tareth and Trass Actually Are

`_certify_operation_role` runs a substitution test:

1. Perturb this variable's value to several spread points.
2. For each perturbation, compare the average outcome at all other visible
   variables before and after.
3. If at least half the perturbations propagate a change beyond any visible
   variable's noise tolerance → tareth.
4. Otherwise → trass.

The verdict is scoped:

- Scoped to the current set of visible variables. When new variables are
  revealed, `_retest_trass_vars` re-runs the test. A variable that was trass
  when 3 vars were visible may be tareth when 10 are visible.
- Scoped to the current noise tolerances. If the envelope certifies a tighter
  epsilon, a previously-trass variable might now propagate detectable changes.
- Scoped to the current intervention targets. If `role_salience` is set, only
  specific downstream targets count for the verdict.

Trass is not a permanent property of a variable. It is a provisional verdict
that was earned under specific conditions and can be revoked when those
conditions change.

---

## The False-Trass Problem (Already Handled At One Level)

The code at `_install_var` lines 708-723:

```python
for p in parents:
    if pn.operation_role == "trass":
        pn.operation_role = "untested"
        # force re-test
```

When variable x4 fits to `MEAN(x1, x2)`, but x1 was classified trass, this
is a contradiction. Perturbing x1 was found to not propagate in isolation —
but x4 depends on x1, so x1 does propagate through x4. The local-scope trass
verdict was earned without accounting for x4 as a mediator.

This is the false-trass problem: two local trass verdicts can jointly fail.
The code handles it reactively: when the contradiction is detected, re-test.
The reference files handle it proactively: composition triggers a joint
re-test before the composed verdict is accepted.

Dreth currently handles the first level of this. It does not handle the
problem at the form level or at higher levels of composition.

---

## What Forms Are (And What They Were Intended To Be)

A Form is a discovered pattern: multiple variables share the same operator,
the same arity, and similar behavior under similar parent distributions.

The `STATE.md` Phase 1 description says forms are "first-class objects."
The `forms.py` docstring says the [Future] direction is:

> Recursive structure: a form can be defined in terms of other forms (a form
> whose "parent" is itself a form-output). Phase 2.

My reading of the intended design is that Forms are the second level of the
certification hierarchy:

- Level 0 certifies: which (parents, func) explains variable i?
- Level 1 certifies: which operator and arity pattern recurs across multiple
  variables in a consistent way? That pattern is a form.
- Level 2 (not implemented): which patterns of patterns recur? That would be a
  form-of-form, a second-order nethra.

A form is not a certified nethra just because multiple variables happen to fit
the same operator. Structural co-occurrence is morphology. A form becomes a
certified nethra when the shared pattern survives joint intervention testing
across its instance bindings — when substituting one binding for another
leaves form-level monitored targets unchanged.

This is the same substitution-test-as-certification model, applied one level
up. The form's operation_role (does this operator pattern matter?) requires
its own substitution test at the form level. The form's sentinels are the
sentinel_template in the code — shared sentinel structure across all instances.

Currently: forms are discovered from per-variable fits (morphology observation),
and compressions discovered for a form propagate to all instances. But the
form does not yet have its own envelope, its own operation-role test, or its
own frontier. It is a bookkeeping object, not yet a certified nethra in its
own right.

---

## The Recursive Certification Vision

What I think the design intends:

Each level of certified nethras serves as the raw material for the next level.

Level 0 nethras (variable fits) → when certified, they enter the
available_parents pool that Level 1 (forms) can use as the substrate for
discovering shared patterns.

Level 1 nethras (forms) → when certified, they should gate the hypothesis
space for variables that might be instances of that form. Instead of
enumerating all (parents, func) pairs for a new variable, consult the form
registry first. If the variable looks like an instance of Form_007, use
Form_007's sentinel_template as the primary validation path. Only fall back
to per-variable audit if form validation fails.

Level 2 nethras (forms-of-forms) → not yet designed, but structurally the
same: certified Level 1 nethras become the substrate.

The attention-economy payoff is multiplicative: 5 variables that share a form
need 1 form-level sentinel check, not 5 independent per-variable checks. If
the form is certified as trass at the operation level, all 5 are skipped with
one test. The recursion is where the cost savings compound.

---

## The TiedFrontier's Role In This

A TiedFrontier records that multiple hypotheses are currently
indistinguishable under the current probe set. It exists because:

1. Score proximity is a morphological fact: these candidates give similar
   predictions on the interventions we ran. It is not a causal fact.

2. The ambiguity must be maintained until there is evidence to resolve it.
   Premature collapse (choosing one candidate because it scored slightly higher
   on one audit) discards information that might be needed later.

3. The tie might be informative: candidates that persistently tie might reveal
   that the FUNC_LIBRARY is missing an operator (library defect), that two
   operators are genuinely equivalent under this variable's scope (operational
   equivalence), or that parent ordering doesn't matter (structural symmetry).
   But that inference requires regime-survival evidence — the tie must persist
   across audits with different available_parents contexts, not just recur
   under identical conditions.

4. The tie has a direct connection to the form level: persistent ties that
   share an operator across multiple variables might be early evidence of a
   form. Ties that are structurally symmetric (same operator, different parent
   bindings, consistent across interventions) might be evidence that a form
   exists with multiple equivalent bindings.

None of this can be inferred from the tie set alone. The morphology (what shape
is the tie) must be computed first. The separating probes must be generated and
run. The regime-survival history must accumulate. Only then can cause be
assigned.

Currently, TiedFrontier accumulates the morphological facts but cannot yet
generate separating probes or track regime survival. It is the right structure
but not yet operative.

---

## Where I Think I Was Wrong Previously

**I treated nethra as a passive label.** I described VarNethra as "the agent's
current belief about a variable" — which is correct but incomplete. The missing
part: the belief gates every downstream inference. When I wrote documentation
that said "certified nethras record what the agent knows," I was
underdescribing the operative role. A certified nethra does not just record
knowledge — it changes what the agent considers for every dependent variable.

**I missed the scope-dependence of tareth/trass.** I wrote about "permanent
operation roles" and "irrelevant variables." The code shows `_retest_trass_vars`
exists precisely because the verdict is not permanent. The scope changes and
the verdict must be re-earned. I wrote documentation that implied trass
classification was final.

**I underestimated the forms recursion.** I described forms as "structural
patterns discovered from per-variable fits" and stopped there. The Phase 2
direction in `STATE.md` and the "recursive structure" note in `forms.py` both
point toward forms being certified nethras at a higher level, not just bookkeeping.
I did not push on what that would mean operationally (form-level envelopes,
form-level substitution tests, form-level operation-role certification).

**I confused morphology with evidence.** When I worked on TiedFrontier, I
repeatedly tried to use the tie set to suppress `semantic_changed` or bias
probe selection — acting as though the morphological fact (these candidates
tie) implied something about the cause (these candidates are equivalent). The
correct sequence is: observe tie → record morphology → generate separating
probes → observe regime survival → classify cause. I kept trying to skip to
the end.

**I missed that the attention economy closes a loop.** The framework is not
just "learn structure and then predict cheaply." The certified structure feeds
back into the cost of learning further structure. Certified Level-0 nethras
reduce the hypothesis space for Level-1 form discovery. Certified Level-1
nethras reduce the audit cost for new Level-0 variables. The loop compounds.
I was treating each level as independent.

---

## What I Am Still Uncertain About

1. Whether the intended "operation" is well-defined at the form level. At the
   variable level, the operation is "predict this variable's output given the
   current world state." At the form level, what is the operation? I don't
   know if this has been worked out.

2. Whether false-trass at the form level is handled by the same mechanism
   (reactive: detect contradiction when it appears) or requires a proactive
   joint re-test (as in the reference files). The reference files suggest
   proactive joint re-testing is necessary for the composition to be
   trustworthy. The current code handles it reactively. I don't know if
   reactive handling is sufficient at higher levels.

3. Whether the TiedFrontier is intended to feed directly into form discovery
   (ties that persist and are structurally symmetric might be evidence of a
   form with multiple equivalent bindings) or whether form discovery and
   frontier resolution are meant to be independent pipelines that occasionally
   intersect. The reference files suggest they should be connected, but I
   cannot tell if this was the original intention.

4. How the attention economy is supposed to close at form-level trass. If a
   Form is certified trass (perturbing this shared pattern doesn't propagate
   beyond tolerance), does that mean all variables matching the form are
   collapsed simultaneously? Or does each instance need its own trass test?
   The current code skips per-variable trass tests for instances that would
   inherit a form-level trass verdict, but the form-level test doesn't yet
   exist.

---

## Summary

Dreth is a streaming causal structure learner that builds up a recursive
hierarchy of certified nethras. Each level earns its certification by surviving
intervention tests in a specific scope. Certified nethras gate the hypothesis
space and dispatch paths for every downstream inference — they are operative,
not descriptive. Certification is provisional: it is revoked when scope
changes, when drift is detected, or when a composition contradiction appears.
Forms are intended to be the next level of this hierarchy, with their own
certification lifecycle, not just bookkeeping derived from per-variable fits.

The false-trass problem appears at every level of composition and must be
handled by joint re-testing, not by inheriting local verdicts. The TiedFrontier
is the mechanism for maintaining ambiguity as a first-class object until
regime-survival evidence justifies collapse. Collapse before that evidence
exists is morphology masquerading as cause.

That is my current understanding. It is probably still incomplete.
