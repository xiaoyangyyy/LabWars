# Causal Decompiler Paper Protocol

> LabWars' conference-facing experiment is not a 630-cell ATE grid. It is an **MRI of one factual trajectory**: freeze exogenous noise and LLM text, patch the unrolled SCM, and read a vector of public vs private outcomes.

## 1. What is supposed to surprise a reviewer

Most LLM-agent papers report "treatment X changed success rate." LabWars reports four objects that usually do not exist in those papers:

1. **Identity twin.** A no-op counterfactual must reproduce the factual actions. If this fails, later ATEs are theatre.
2. **Split-Y.** Public protest can sit at 0.03 while private divergence is 0.5. Same `do()`, two transcripts.
3. **Memory IRF.** Deleting the same cluster at t=3 vs t=45 vs t=52 is an interventional curve, not a mediation fraction `|ΔM/ΔY|`.
4. **AND-cause lie.** Skipping "promise" and skipping "draft" each look decisive; Shapley splits them. The planted SCM (Y = promise AND draft) is the oracle: knockout sums to 2, Shapley sums to 1.

Three-worlds adds the social-gating channel: W0 factual, W1 `do(op)`, W2 `do(op)` + omniscient observation. If private divergence moves and public compliance does not, the effect lived in the hidden transcript.

## 2. Commands

Scripted smoke (no API):

```bash
python -m src.experiments paper --rounds 8 --seed 11 --llm-provider scripted
```

Full 60-round MRI after a persisted DeepSeek factual run (replays the LLM sidecar, does not re-pay for the factual world):

```bash
python -m src.experiments paper --from-jsonl output/runs/run_XXXX.jsonl --full-cast
```

A/B/C/D as CRN twins of one control, not independent seeds:

```bash
python -m src.experiments paper --rounds 60 --full-cast --contrasts A --llm-provider scripted
```

Experiment-level wrappers:

```python
from src.experiments.exp_a_promise import run_paper
from src.experiments.exp_c_false_memory import run_paper as run_c
```

## 3. Tables the protocol writes

| Table | Estimand |
|---|---|
| Identity twin | CRN + LLM replay hits/misses |
| Split-Y | protest, PPD, R52 comply, cluster, promise broken/honored, trust logged |
| Memory IRF | ATE of `do_memory(t)` over story beats |
| Shapley vs skip | planted oracle + budgeted story events |
| Three worlds | total / omniscient / gated channel / hypocrisy index |
| CRN contrasts | A1→A2 honor, A1→A5 delete, C3→C2 false memory, … |

Outputs: `output/reports/paper_protocol_{run_id}.md` and `.json`.

## 4. What not to run for the paper

The 19×60 DeepSeek matrix is a cost accident. Do not treat it as Causal MRI. Validity still needs shuffled vs full memory, but as a CRN pair (`V6→V2`), not as a separate unpaid-for grid.

## 5. Identification caveats (write these in the paper)

- Memory IRF is an interventional analogue of an indirect effect, not a natural indirect effect.
- LLM replay identifies twins only when the prompt is unchanged. `λ` lesions rewrite prompts and will miss the cache.
- N=1 bootstrap CIs are vacuous; multi-seed is for the CRN contrast table, not for the MRI listing of one machine.
