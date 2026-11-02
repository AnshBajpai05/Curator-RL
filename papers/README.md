# Papers we compared against

Only the papers whose methods or results CURATOR is directly compared with. Downloaded from arXiv on
2 November 2026; the first page of each PDF was checked against its title. The PDFs stay local
(`papers/.gitignore`): arXiv papers carry their authors' licences, so the repository links to them instead.

| File | Paper | Venue | How we use it |
|---|---|---|---|
| `DataFlex-RL_2026_arXiv2609.06107.pdf` | Liang, Chen, Feng, Qiang, Zhang. *DataFlex-RL: An Evaluation Platform for RLVR Data Policies*. [arXiv 2609.06107](https://arxiv.org/abs/2609.06107) | arXiv, September 2026 | The field's reference result: "None of the eight selection or reweighting methods has a paired 95% confidence interval excluding zero relative to uniform sampling" (Qwen2.5-7B, 12 seeds). Our accuracy null (Curator ties Uniform) agrees with it. |
| `SEC_Self-Evolving-Curriculum_COLM2026_arXiv2505.14970.pdf` | Chen, Lu, Kim, Zhang, Tang, Piché, Gontier, Bengio, Kamalloo. *Self-Evolving Curriculum for LLM Reasoning*. [arXiv 2505.14970](https://arxiv.org/abs/2505.14970) | COLM 2026 | Baseline. Re-implemented as `curator-rl/src/curator_rl/scheduler/baselines/sec.py`; compared on GPU (D-103, 3 seeds; D-108). Checked against the authors' code: `curator-rl/reports/analysis/sec_dump_fidelity.md`. |
| `DUMP_Distribution-Level-Curriculum_2025_arXiv2504.09710.pdf` | Wang, Cui, Li, Wan, Zhao. *DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training*. [arXiv 2504.09710](https://arxiv.org/abs/2504.09710) | arXiv, 2025 | Baseline. Re-implemented as `scheduler/baselines/dump.py`; compared on GPU (D-103, D-108); checked against the authors' code (same report). |
| `Graves_Automated-Curriculum-Learning_2017_arXiv1704.03003.pdf` | Graves, Bellemare, Menick, Munos, Kavukcuoglu. *Automated Curriculum Learning for Neural Networks*. [arXiv 1704.03003](https://arxiv.org/abs/1704.03003) | ICML 2017 | Learning-progress curricula: the LP baseline (`scheduler/baselines/lp.py`, compared on GPU, 3 seeds) and one of Curator's own signals. |
| `TSCL_Teacher-Student-Curriculum-Learning_arXiv1707.00183.pdf` | Matiisen, Oliver, Cohen, Schulman. *Teacher-Student Curriculum Learning*. [arXiv 1707.00183](https://arxiv.org/abs/1707.00183) | IEEE TNNLS 2020 | The \|learning progress\| Boltzmann rule our LP baseline follows. |

Cited in the documentation but **not** compared against, so not included: TAC (ICML 2026), HarnessBandit,
Prompt Replay, MoDoMoDo, RISED, Spurious Rewards (ICML 2026) and the 2026 noisy-data study.

To fetch the PDFs again: open each arXiv link and choose "View PDF".
