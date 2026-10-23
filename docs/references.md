# Sources and inspiration

llm-evals started from three papers, and most of its design choices trace back to
published work on evaluation, statistics and testing. This page lists them, what
each one informed, and where that idea lives in the code, or that it isn't built yet.

The sources were opened and checked on 19 October 2026, while the design was being
written. Wilson, McNemar, NSGA-II, Luo et al. and Landis & Koch were checked against
their bibliographic records rather than the full text.

## Where the design started

The design combines the discovery loop from one paper with the gating structure from
another, and adds paired statistics that neither has.

| Paper | What we took | What we left |
|---|---|---|
| [STELLAR: A Search-Based Testing Framework for Large Language Model Applications](https://arxiv.org/abs/2601.00497) (Sorokin et al., BMW/TUM) | Generating tests over a space of input features; deterministic checks alongside judge checks; checking the judges themselves | Single-sample labels, failure counts as the metric, and search guided by an uncalibrated judge |
| [Automated Self-Testing as a Quality Gate: Evidence-Driven Release Management for LLM Applications](https://arxiv.org/abs/2603.15676) (Maiorano) | Deterministic checks first; PROMOTE / HOLD / ROLLBACK; failures fed back into the test cases; tiered scenarios | Absolute thresholds on a single run, with no comparison against a baseline |
| [Multi-Agent LLM Committees for Beta Testing](https://arxiv.org/abs/2512.21352) (Karanam and Kennady) | The idea of diverse simulated personas, for multi-turn testing later | The headline numbers, which have methodological red flags |

## Where each idea comes from

| Design choice | Source and what it supports | In llm-evals |
|---|---|---|
| Paired difference on shared cases, uncertainty clustered by case, several samples per case, a minimum detectable difference | [Miller, "Adding Error Bars to Evals" (2024)](https://arxiv.org/abs/2411.00640) recommends paired standard errors, clustered errors for grouped questions, resampling answers to cut variance, and power analysis | The core of the gate: `stats/paired.py` (cluster bootstrap, detectable drop) and `samples` in each suite |
| A Wilson interval for each case's rate | [Wilson (1927), JASA 22:209–212](https://doi.org/10.1080/01621459.1927.10502953) | `stats/intervals.py` |
| McNemar as a second paired test | [McNemar (1947), Psychometrika 12:153–157](https://doi.org/10.1007/BF02295996) | `stats/paired.py`, shown in every report |
| Treating every output as variable, even at temperature 0 | [Ouyang et al., "An Empirical Study of the Non-determinism of ChatGPT in Code Generation" (TOSEM)](https://arxiv.org/abs/2308.02828) found temperature 0 doesn't guarantee determinism. [Atil et al. (2024)](https://arxiv.org/abs/2408.04667) found accuracy swings of up to 15% across runs in "deterministic" settings. [He / Thinking Machines (2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) traces the cause to kernels that aren't batch-invariant | Every case runs several times per version, and `llm-evals aa` measures the noise by running a version against itself |
| Pin model ids, never aliases | [Chen, Zaharia & Zou (2023)](https://arxiv.org/abs/2307.09009): the same model service behaved substantially differently between March and June 2023 | Version files name a model id, and the version id hashes it, so a model change is a new version |
| A pass rate per case, not one pass or fail | [Yao et al., τ-bench (2024)](https://arxiv.org/abs/2406.12045) defines pass^k, success on all k trials, and shows reliability falling as k grows | Each case's rate across its samples; the gate compares those rates |
| PROMOTE / HOLD / ROLLBACK, deterministic checks first, failures fed back into the cases | [Automated Self-Testing as a Quality Gate](https://arxiv.org/abs/2603.15676) | The gate's decisions (`stats/gate.py`, `engine.py`); adding a production failure as a case is the documented workflow |
| A pairwise judge run in both orders | [Zheng et al., MT-Bench (2023)](https://arxiv.org/abs/2306.05685) measured position bias: GPT-4 gave the same verdict after swapping the order in only about 65% of cases | `checks/pairwise.py` keeps a winner only when both orders agree (report only) |
| Calibrate judges against human labels; expect the criteria to shift while labeling | [Shankar et al., "Who Validates the Validators?" (UIST 2024)](https://arxiv.org/abs/2404.12272) describe criteria drift | `llm-evals calibration`. It happened here too: labeling the faithfulness judge changed what "invented" and "the actor" should mean |
| A checklist judge of yes/no items | [Cook et al., "TICKing All the Boxes" (2024)](https://arxiv.org/abs/2410.03608): generated checklists raised exact agreement with human preferences from 46.4% to 52.2% | `checks/checklist.py`, one gating unit per rubric item |
| κ of at least 0.6 before a judge can gate | [Landis & Koch (1977)](https://doi.org/10.2307/2529310) call 0.61–0.80 "substantial" agreement. Their bands are a rough guide, so the bar itself is a design choice | A judge unit gates only with at least 50 labels and κ ≥ 0.6, for that exact judge model and prompt |
| Faithfulness checked claim by claim | [Min et al., FActScore (EMNLP 2023)](https://arxiv.org/abs/2305.14251) | The research agent's `claim_support` judge checks each claim against the documents it cites |
| Metamorphic invariance tests | [Ribeiro et al., CheckList (ACL 2020)](https://arxiv.org/abs/2005.04118) | `checks/invariance.py`: rerun on a reworded input and expect the same answer |
| The answer must follow a changed fact | [Longpre et al. (EMNLP 2021)](https://arxiv.org/abs/2109.05052) swapped answers in passages and found models lean on memorized knowledge | The research suite's known-facts cases and `follows_corpus` check |
| Rerunning to separate real failures from flaky ones | [Luo et al., flaky tests (FSE 2014)](https://doi.org/10.1145/2635868.2635920) | A critical failure rolls back only if it fails on every sample and reruns confirm it |
| Repeated samples per case in an agent framework | [Inspect: epoch reducers](https://inspect.aisi.org.uk/metrics.html) | `samples` per case, for prompts and tool-using agents alike |
| A held-out split; case changes kept apart from version changes | The [Quality Gate paper](https://arxiv.org/abs/2603.15676) names test overfitting as a failure mode. [STELLAR](https://arxiv.org/abs/2601.00497) notes that static benchmark samples can leak into training data | `held_out` cases are never shown in PR comments; judge prompts are validated on a held-out labeled batch; version ids hash only what the model sees |
| Generating tests over a feature space | [STELLAR](https://arxiv.org/abs/2601.00497) | Partly: the example suites generate cases from a grid of known features (format, units, traps), recorded as `strata`. The search over that space isn't built |
| Evolutionary search for new failures | [STELLAR](https://arxiv.org/abs/2601.00497), [NSGA-II (Deb et al., 2002)](https://doi.org/10.1109/4235.996017) and [pymoo](https://arxiv.org/abs/2002.04504) | Not built yet: the planned discovery loop |
| Confirming and re-sampling borderline discoveries | [Jin & Branke (2005), IEEE TEVC 9(3)](https://pub.uni-bielefeld.de/record/2978653) survey noisy fitness and averaging remedies. [Smith & Winkler, "The Optimizer's Curse" (2006)](https://doi.org/10.1287/mnsc.1050.0451) show that options picked as the best are systematically overestimated | Not built yet: part of the discovery loop |
| Discovery guided only by deterministic or calibrated checks | [Gao, Schulman & Hilton (2022)](https://arxiv.org/abs/2210.10760): optimizing against a proxy reward model eventually lowers the true score | Not built yet: the discovery loop. The same reasoning is why uncalibrated judges can't gate |
| Correcting a failure rate for judge error | [Rogan & Gladen (1978), Am J Epidemiol](https://pubmed.ncbi.nlm.nih.gov/623091/) | Not in llm-evals. It appears in [The Eval Stack](https://www.benjaminwyndham.com/tools/eval-stack), the companion design tool |
| Canary strings in public case files | [BIG-bench (Srivastava et al., 2022)](https://arxiv.org/abs/2206.04615) puts a canary GUID in task files so they can be filtered from training sets. A [2024 report](https://www.alignmentforum.org/posts/kSmHMoaLKGcGgyWzs/big-bench-canary-contamination-in-gpt-4) found a GPT-4 base model could reproduce one, so canaries are a weak signal | Not built yet |
