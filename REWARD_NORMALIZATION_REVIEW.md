# Reward normalization correction and analysis comparison

The maximum-error normalization was corrected and the complete paper analysis was rerun. The correction changes the converted calibration and rewarded scores, but **no primary or sensitivity association changes its significance decision at .05**. Reward reliability increases from ICC(2,3) .40 to .44. Accuracy, raw calibration, Brier score, resolution and outcome variance remain unchanged.

This branch evaluates a retrospective analysis composite. It does not establish which task implementation participants saw or which formula determined their historical bonuses. The manuscript and task webapp were not edited.

## Scope and provenance

- Local branch: `fix/reward-normalization`.
- Starting point: the supplied `category-learning-metacog-analysis-main (2).zip` snapshot, previously identified with upstream revision `c893630d985484e0ffe517dcaabf9952d0e76a83`.
- The initial analysis used an archive without Git history. For publication, the correction was applied to a fresh clone of the upstream repository at the revision above. The archive baseline and upstream tree were verified to be identical, and the correction applied without conflicts. This branch preserves the upstream Git history.
- The same six raw CSV files, participants, confidence-key mapping, trial inclusion rules, statistical models, and comparison families were used in both runs.

## What was corrected

The previous denominator treated confidence key 1 as maximally miscalibrated when every outcome was correct, while treating keys 2–4 as maximally miscalibrated when outcomes were incorrect. For key 1, the mapped probability is .5625: squared error is .19140625 when correct and .31640625 when incorrect. The original denominator therefore was not the stated maximum.

For fixed confidence-bin counts, all four mapped probabilities exceed .5. Each bin's calibration error is maximized when its accuracy is zero. The corrected denominator is

`C_max = sum_j (N_j / N) * f_j^2`,

where `N_j` is the number of responses in confidence bin `j` and `f_j` is its fixed probability. Converted calibration remains `1 - C / C_max`, bounded to [0,1], and rewarded score remains the average of accuracy and converted calibration. Only the construction of `C_max` changed. It is evaluated through the existing weighted calibration function using all-incorrect outcomes.

The fixed probabilities remain .5625, .6875, .8125 and .9375. This work does not test incentive compatibility or establish that the composite is a strictly proper scoring rule.

## Changes to scores

Scores changed in **162 of 240 participant-sessions**, involving 71 of the 80 participants. All changes were increases or zero. The average absolute change in rewarded score was **.006068** on the 0–1 scale (0.607 points on a 0–100 scale); the maximum individual-session change was **.142830** (14.283 points). Thus, a small mean change does not imply negligible change for every participant.

| Mean rewarded score | Before | Corrected |
|---|---:|---:|
| Initial session | .748547 | .757148 |
| One-week retest | .785446 | .791116 |
| Final retest | .812044 | .815976 |
| Across all participant-sessions | .782012 | .788080 |

## Reliability

| Rewarded-score reliability | Before | Corrected |
|---|---:|---:|
| ICC(2,3) | .400746 | .435077 |
| 95% confidence interval | .15–.59 | .19–.62 |
| P | .002049 | .000685 |

These coefficients concern the **mean across three administrations**, not a single administration. The point estimate remains in the manuscript's “fair” category, with substantial uncertainty. The other five ICCs and intervals are unchanged.

## Strategy models

All 12 strategy models converged. The corrected rewarded-score omnibus tests were:

| Effect | Previous chi-square (df=2) | Corrected chi-square (df=2) | Previous P | Corrected P | Previous FDR P | Corrected FDR P |
|---|---:|---:|---:|---:|---:|---:|
| Classification strategy | 7.363572 | 8.534445 | .025178 | .014021 | .041200 | .025237 |
| Decision strategy | 8.508062 | 7.517301 | .014207 | .023315 | .025572 | .038152 |
| Session | 12.178064 | 12.535972 | .002268 | .001896 | .007891 | .006826 |

All three associations remain significant after correction across the 18 primary tests. The nonreward model estimates, raw P values and primary FDR-adjusted P values are unchanged. In the separately corrected 18-test two-way interaction family, no interaction is nominally significant; the minimum unadjusted P remains .372580. No pairwise strategy superiority or causal conclusion is established.

## Questionnaire associations

Both the primary N=80 and strict sensitivity N=51 analyses retain six nominal associations and **zero associations surviving either FDR or Bonferroni correction** across their respective 66-test families. No nominal or corrected significance decision changes.

All 11 reward-related correlation coefficients in each sample were recomputed. The other five metrics' raw correlations and P values are unchanged. Recalculating the complete families changes 49 FDR-adjusted values in the N=80 analysis and 51 in the N=51 analysis, including values for nonreward outcomes, without changing any significance decisions. Use the regenerated tables rather than assuming adjusted values remain unchanged. Full before/after values are in `reward_normalization_comparison/`.

## Validation completed

1. All 15 notebook code cells executed in fresh baseline and corrected processes. All 30 baseline CSV tables reproduced Warren's saved outputs within rtol 1e-6 and atol 1e-10.
2. Eleven tests passed: the six existing reviewer-correction tests and five new normalization tests. The new tests cover an explicit low-confidence counterexample, the all-incorrect maximum, perfect calibration, exhaustive binary outcomes, and invariance to uniform export-row duplication.
3. Independent calculations from deduplicated main-task testing trials verified all 240 corrected participant-session scores within absolute tolerance 1e-12. Each session contained 42 distinct trials per participant, with four identical exported records per trial.
4. The other five task metrics were unchanged within absolute tolerance 1e-12. N=80 and N=51 sample definitions, testing-only feature-distance results, calibration change tests and power calculations were preserved.
5. All strategy fits converged. Existing scale-related boundary warnings and power-approximation numerical warnings were retained in the logs; no execution failed. Model diagnostics, logs, package versions and input-file hashes are included.

Nineteen of 30 CSV outputs were unchanged; 11 changed. Dependent figures and supplemental PDFs were regenerated. The original supplied files remain preserved in the upstream base revision and in the separate local analysis archive.

## Implications for the manuscript

If this correction is adopted:

- Describe the normalized composite as calculated for analysis and specify the trial-weighted maximum-error denominator. Keep the description of actual displayed scores and payment separate until the deployed implementation is confirmed.
- Replace rewarded-score ICC .40 (95% CI .15–.59) with **.44 (95% CI .19–.62)** and P=.000685 (or P<.001 at manuscript precision). Update Figure 4.
- Replace reward coefficients/tests and Supplementary Table S1 with the corrected outputs. The existing prose listing significant strategy factors remains accurate.
- Replace the relevant questionnaire-correlation tables, including FDR values. The statement that no association survived correction remains accurate in both samples.
- Replace dependent reward plots, including the reward panels in Supplemental Figures 2–6. Supplemental Figure 1 contains only accuracy and outcome variance and is numerically unchanged. Existing nonreward plots and feature-distance figures do not require numerical changes.
- The abstract's qualitative “fair” reward reliability description and the main strategy/questionnaire conclusions remain accurate. No change to the other five measures or their interpretations is implied by this correction.

Suggested Methods wording for the analysis composite:

> For analysis, we converted calibration error to a score from 0 to 1 by subtracting the ratio of observed calibration error to its maximum possible value from 1. The maximum was calculated for each participant's fixed confidence-response distribution, with bins weighted by trial counts. Because all scoring probabilities exceeded .5, this maximum corresponded to incorrect classifications in every occupied confidence bin. The analysis composite was the mean of accuracy and converted calibration.

The separate author decision is how to describe the historical incentive. This numerical correction does not reconstruct scores displayed to participants or establish bonuses paid.

## Reproducing and reviewing this branch

Install the repository's pinned requirements into a suitable Python environment. Put the six original CSV files in `data/`; raw data are not bundled in the review package. Then run:

```sh
python -m unittest discover -s tests -v
python scripts/run_paper_notebook.py --repo . --data data --output paper_repro_outputs
```

The comparison script accepts `--baseline`, `--corrected`, `--supplied`, `--data` and `--output` paths. The first three point to output roots containing `tables/`. It verifies the baseline against the supplied tables and independently checks corrected scores against the raw trials.

The branch includes the corrected code, regression tests, regenerated outputs, and comparison results. Review the code changes against the upstream base with `git diff c893630d985484e0ffe517dcaabf9952d0e76a83..HEAD -- analysisfunctions.py paper_analysis_utils.py tests/`. The validation results above describe the completed analysis runs; transferring the correction to the upstream Git history did not change the analytical code or result files.
