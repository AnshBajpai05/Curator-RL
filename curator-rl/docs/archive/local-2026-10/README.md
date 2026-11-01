# Local work of 2026-10-09 … 2026-10-12, archived when this branch was adopted as the base (D-104)

This work was done on the owner's machine on top of commit `bd9099e`, in parallel with the
`audit-fixes-and-gpu-run` branch, and was never committed. When that branch (PR #1) was adopted as the
base, the parts it already covers were dropped. This folder keeps what is unique, so the record of
pre-registered tests stays complete.

**Numbering.** The local decision log used D-96 … D-101, which collide with this branch's D-96 … D-103.
Here they are called **L-96 … L-101** (`local_decisions_L-96_to_L-101.md`; the headings inside still read
"D-…" as written at the time). The main log refers to them as L-numbers in D-104 … D-106.

| File | What it is | Status |
|---|---|---|
| `local_decisions_L-96_to_L-101.md` | L-96: a parallel Phase F runner (superseded by this branch's runner). L-97: shared-slice diagnosis. L-98: the φ-audit pre-registration, frozen. L-99: its outcome. L-101: smoke test and learning-rate pilot on Kaggle with the local runner | historical record |
| `L-98_reward_validity_audit_prereg.md` | Frozen pre-registration of a *label-based* reward-validity audit: φ between an arm's reward and the slice verifier on calibration answers | run; Stage T negative (L-99) |
| `L-100_shared_slice_evidence_rule_proposal.md` | Follow-up proposal: audit for shared-slice arms, gain test for private ones | never frozen; superseded by this branch's label-free audit (D-98) |
| `audit_operating_characteristics.py` | Monte Carlo of the φ rule's flag rates (single and repeated looks); numpy only, runs anywhere | runnable |
| `reports/shared_slice_diagnosis.*` | L-97 numbers | data |
| `reports/audit_tune.*`, `reports/audit_diagnosis.*` | L-99 numbers (Stage T and reason attribution) | data |

The code that produced these reports (`calibration/audit.py`, joint slice credit, the local runner) is
**not** in this branch: it was written against the pre-PR code, and this branch rebuilt the same areas
differently. It is kept in `RL_project_uncommitted_backup_2026-10-22.zip` next to the repository.
