# Reviewer revisions to the manuscript analysis

> Historical revision record: reward-related estimates below predate the maximum-error normalization correction. See [REWARD_NORMALIZATION_REVIEW.md](REWARD_NORMALIZATION_REVIEW.md) and the regenerated `paper_repro_outputs` for current branch results.

The strategy models, ICC reporting, and feature-distance trial inclusion have been corrected in the code, notebook, and generated outputs. The feature-distance analysis now uses testing trials only, as confirmed by the author. The manuscript file has not been edited; replacement wording below is ready to transfer to the current draft.

## Strategy analysis: BH15 and BH19–20

The previous OLS analysis treated repeated participant-sessions as independent. After excluding the one `Wrote down` report, 239 sessions from 80 participants remained. Only 24 of 27 classification-strategy by decision-strategy by session cells were occupied. The full factorial design had rank 24 with 27 columns. Removing the singleton did not make that specification identifiable. Statsmodels explicitly documents that linear models can use a generalized inverse without rejecting rank-deficient designs; the revised code checks rank before fitting and checks the covariance of each tested contrast before inference. See [Statsmodels' rank-deficiency documentation](https://www.statsmodels.org/stable/pitfalls.html#rank-deficient-exog-perfect-multicollinearity).

The revised primary analysis uses a linear mixed model for each of the six metrics:

```text
metric ~ classification strategy + decision strategy + session + participant random intercept
```

The fixed effects are categorical. Models are fitted with REML and use asymptotic Wald chi-square tests. The additive specification assumes no interactions and was adopted as a parsimonious, identifiable revision; it was not the original analysis. A separate sensitivity specification includes all two-way interactions and tests each interaction jointly. It assumes no three-way interaction. Full rank in these restricted models does not recover all effects of the original full factorial model. [Statsmodels describes random-intercept models for repeated observations](https://www.statsmodels.org/stable/mixed_linear.html).

All six additive designs have rank 7/7 and all six sensitivity designs have rank 19/19. All 12 models converged, with positive-definite fixed-effect covariance matrices. Fitting warnings are retained in the diagnostics CSV. The boundary warning is triggered by small absolute variances on the original outcome scales; random-intercept variance is approximately 0.11–0.38 times residual variance, rather than approximately zero. Sparse cells still limit interaction precision.

Unadjusted P values are retained. Benjamini-Hochberg FDR adjustment is additionally reported across the 18 primary factor tests and, separately, across the 18 sensitivity interaction tests. These are two exploratory families. Statistical decisions use full-precision values, not displayed rounding.

### Revised findings

| Task metric | Classification strategy P | Decision strategy P | Session P |
|---|---:|---:|---:|
| Accuracy | .003136 | .000233 | .004766 |
| Brier score | .032517 | .085635 | .000472 |
| Calibration | .179270 | .625883 | .002994 |
| Resolution | .086861 | .344789 | .00004995 |
| Outcome variance | .003507 | .000748 | .155229 |
| Rewarded score | .025178 | .014207 | .002268 |

Every primary result with unadjusted P < .05 also has FDR-adjusted P < .05 in the 18-test family. The classification-strategy association with Brier score has adjusted P=.048775. Calibration is associated with session (adjusted P=.007891), but neither strategy factor is associated with calibration at the .05 threshold. None of the two-way interaction tests reaches even unadjusted P < .05 (minimum P=.372580). This does not demonstrate that interactions are absent. The original three-way interaction cannot be tested as specified and must not be described as significant.

The historical accuracy/classification OLS result is P=.04950000324868886. It rounds to .050 at three decimals, but it comes from the superseded analysis. The corrected primary association is chi-square(2)=11.529929, P=.003135506, FDR-adjusted P=.007890573. The old CSV is retained in `paper_repro_outputs/archive/supplemental_table_1_strategy_anova_superseded.csv` for provenance only.

### Replacement Methods wording

We examined associations of self-reported classification strategy, decision strategy, and session with each behavioral metric using separate linear mixed models with a participant random intercept to account for repeated administrations. The primary models included additive categorical fixed effects of the two strategy factors and session and were fitted by restricted maximum likelihood. The singleton report of writing down components was excluded from these regressions, leaving 239 participant-sessions from 80 participants. We audited cell counts and design-matrix rank before inference. The original full factorial specification was rank deficient because three cells were empty; therefore, it was replaced by the additive models, with full-rank models containing all two-way interactions used as a sensitivity analysis. Omnibus effects were assessed with asymptotic Wald chi-square tests. We report unadjusted P values and Benjamini-Hochberg FDR-adjusted P values across the 18 primary tests and, separately, across the 18 sensitivity interaction tests. Strategy associations were considered exploratory.

### Replacement Results wording

In the revised primary models, classification strategy was associated with accuracy, Brier score, outcome variance, and rewarded score; decision strategy was associated with accuracy, outcome variance, and rewarded score. Session was associated with all metrics except outcome variance. These associations met the .05 threshold after FDR correction across the 18 primary tests. Neither strategy factor was significantly associated with calibration or resolution. No two-way interaction reached nominal significance in the sensitivity analysis. The full three-way interaction was not estimable under the original specification, so no three-way interaction conclusion was drawn.

These are omnibus associations. They do not establish that every pair of strategies differs, or that choosing randomly is significantly worse than every alternative. Pairwise claims require explicit contrasts and their own justified multiplicity handling. The abstract, Results paragraph beginning “We then examined the relationship between strategy,” and Discussion section “Strategy and metacognition” should lose the old three-way-interaction and strategy–calibration claims. Use “associated with,” rather than language implying that self-reported strategy caused changes in performance.

### Generated evidence

- [Corrected primary tests](paper_repro_outputs/tables/supplemental_table_1_strategy_mixed_models.csv)
- [Corrected Supplemental Table 1](paper_repro_outputs/supplement/supplemental_table_1.pdf)
- [Two-way interaction sensitivity tests](paper_repro_outputs/tables/strategy_two_way_interactions_sensitivity.csv)
- [Cell counts](paper_repro_outputs/tables/strategy_cell_counts.csv) and [design ranks](paper_repro_outputs/tables/strategy_design_audit.csv)
- [Coefficient estimates and 95% intervals](paper_repro_outputs/tables/strategy_model_coefficients.csv) and [fit diagnostics](paper_repro_outputs/tables/strategy_model_diagnostics.csv)

## ICC interpretation: BH3 and BH10

The calculation explicitly selects ICC2k. With three administrations this is ICC(2,3): a two-way random-effects, absolute-agreement estimate for the mean across three administrations. It does not estimate the reliability of a single administration. The distinction between individual and average measurements is part of the ICC specification in the [Stata ICC manual](https://www.stata.com/manuals/ricc.pdf). The point estimates and existing confidence intervals are retained; their interpretation and display are corrected. Numeric interval bounds, number of administrations, and the unit of measurement are now exported explicitly.

| Metric | ICC(2,3) | Existing 95% CI |
|---|---:|---:|
| Accuracy | .56 | .37 to .70 |
| Brier score | .55 | .35 to .69 |
| Calibration | .39 | .13 to .59 |
| Resolution | .22 | −.09 to .46 |
| Outcome variance | .56 | .36 to .70 |
| Rewarded score | .40 | .15 to .59 |

Intervals are shown at the precision returned by the installed Pingouin version. The negative resolution lower bound is retained. The existing manuscript values for Brier score (.53), calibration (.32), and rewarded score (.54) also need to be replaced by these current results. Avoid saying estimated ICCs are necessarily restricted to 0–1, and acknowledge uncertainty rather than relying only on categories such as “fair.”

### Replacement Methods wording

For each behavioral metric, we estimated a two-way random-effects, absolute-agreement intraclass correlation coefficient for the mean across the three administrations, ICC(2,3), using all 80 completers. We report 95% confidence intervals. These coefficients describe reliability of the three-administration mean and should not be interpreted as reliability of a single administration.

### Replacement Results wording

The estimated reliability of the mean across three administrations was .56 (95% CI .37–.70) for accuracy, .55 (.35–.69) for Brier score, .39 (.13–.59) for calibration, .22 (−.09–.46) for resolution, .56 (.36–.70) for outcome variance, and .40 (.15–.59) for rewarded score. The confidence intervals indicate substantial uncertainty, especially for calibration and resolution. These results concern the average over three administrations and do not establish reliability for a single use of the task.

The Results heading, figure caption, abstract, and discussion of longitudinal measurement should use the same unit of reliability. The current code does not substitute ICC(2,1); reporting single-administration reliability would be a separate explicit analysis decision.

See the [ICC table](paper_repro_outputs/tables/icc2k_all_completers.csv) and [updated figure](paper_repro_outputs/figures/test_retest_reliability.svg).

## Feature-distance analysis: testing trials only

The author confirmed that the intended analysis uses testing trials only. `add_feature_distance` now requires both `use == "task"` and `block == "test"`. Training and tutorial rows are excluded before distances and participant-by-distance accuracy means are calculated. The previous analysis pooled 22 training and 42 testing trials per participant per session. The correction aligns the analysis with the Methods and describes performance during testing without feedback.

Both samples retain 42 distinct testing trials per participant per session: 3,360 trials in each primary session (N=80) and 2,142 in each strict sensitivity session (N=51). The source CSV contains four logged rows per trial; the exported model tables distinguish source-row counts from distinct-trial counts. Participant inclusion and the session-specific random-intercept model specification are unchanged.

### Impact on the results

| Sample | Session | Previous pooled slope | Testing-only slope | Testing-only 95% CI | Testing-only P |
|---|---|---:|---:|---|---:|
| Primary N=80 | Initial | −.045536 | −.050625 | −.061777 to −.039473 | 5.71e−19 |
| Primary N=80 | 1-week retest | −.065446 | −.076500 | −.088577 to −.064423 | 2.18e−35 |
| Primary N=80 | Final retest | −.061696 | −.072875 | −.083374 to −.062376 | 3.80e−42 |
| Strict sensitivity N=51 | Initial | −.054639 | −.063529 | −.075019 to −.052040 | 2.29e−27 |
| Strict sensitivity N=51 | 1-week retest | −.091141 | −.102941 | −.116319 to −.089563 | 2.15e−51 |
| Strict sensitivity N=51 | Final retest | −.073459 | −.092549 | −.103239 to −.081859 | 1.40e−64 |

Slopes express change in proportion correct per additional feature of distance. The primary estimates correspond to accuracy decreases of approximately 5.1, 7.7, and 7.3 percentage points per additional feature in the initial, 1-week, and final sessions. Their magnitudes are about 11–18% larger than the previous pooled estimates. All six associations remain negative with P < .001; the substantive conclusion that accuracy decreases with feature distance is preserved.

All six models converged, with full-rank fixed-effect designs and positive-definite fixed-effect covariance matrices. Three scale-dependent boundary warnings occurred in the strict sensitivity fits and are retained in the CSVs. Their random-intercept variances are approximately 0.10–0.53 times residual variance.

### Replacement Methods wording

The feature-distance analysis used only main-task testing trials, excluding training and tutorial trials. For each participant and session, we averaged correctness across testing trials at each feature distance from the category prototype, with prototypes assigned distance zero. We fitted a separate linear mixed model for each session, with feature distance as a fixed predictor and a participant random intercept. The primary analysis included all 80 completers. We repeated the analysis in the 51 participants whose testing accuracy was greater than .5 in all three sessions. Each participant contributed 42 distinct testing trials per session.

### Replacement Results wording

In the primary sample of 80 completers, testing accuracy decreased with feature distance in the initial session (slope −.0506, 95% CI −.0618 to −.0395), the 1-week retest (−.0765, −.0886 to −.0644), and the final retest (−.0729, −.0834 to −.0624; all P < .001). The same pattern was observed in the strict sensitivity sample of 51 participants whose testing accuracy exceeded .5 in all three sessions (slopes −.0635, −.1029, and −.0925, respectively; all P < .001). These results support using feature distance to vary task difficulty during the testing block.

The existing manuscript's N=64 “consistently above chance” description differs from the current code's strict accuracy > .5 in every session rule (N=51). The replacement paragraph above identifies N=80 as the primary sample and N=51 as the strict sensitivity sample. Figure captions should specify testing trials only and identify which sample is plotted.

### Generated evidence

- [Primary testing-only models](paper_repro_outputs/tables/distance_models_all80_primary.csv) and [primary figure](paper_repro_outputs/figures/distance_regression_all80_primary.svg)
- [Strict sensitivity testing-only models](paper_repro_outputs/tables/distance_models_strict51_sensitivity.csv) and [sensitivity figure](paper_repro_outputs/figures/distance_regression_strict51_sensitivity.svg)
- [Primary participant-by-distance means](paper_repro_outputs/tables/distance_grouped_all80_primary.csv) and [sensitivity means](paper_repro_outputs/tables/distance_grouped_strict51_sensitivity.csv)

The four previous combined-block CSVs are retained under `paper_repro_outputs/archive/` with `_training_and_testing_superseded` in their names. They document the change and are not current analysis outputs.

## Validation

All 12 corrected strategy models and six testing-only distance models were fitted on the six local source CSVs. The rebuilt task summaries and ICC point estimates match the previously saved values to numerical precision. Six focused regression tests cover dependence between sessions, empty-cell rank deficiency, rejection of nonidentifiable designs, duplicate/missing observations, the distinction between single- and average-administration ICCs, and exclusion of training/tutorial rows in every distance-analysis session. Every included participant was verified to have 42 distinct testing trials per session, and the exported distance means match direct testing-block aggregation. All 15 notebook code cells executed sequentially in a fresh Python process, and the numeric outputs across all 30 CSV tables matched the saved results within numerical tolerance. The revised supplemental PDF, ICC figure, and both distance figures were visually inspected.
