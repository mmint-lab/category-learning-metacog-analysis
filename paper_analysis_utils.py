"""Utilities for reproducing the metacognition game paper analyses.

This module contains reusable analysis functions for the manuscript
reproducibility notebook. The notebook can be read top-to-bottom and rerun
from a fresh kernel without hidden state.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy.stats import linregress, nct, norm, pearsonr, t, ttest_rel
from statsmodels.stats.multitest import multipletests

from analysisfunctions import (
    calcBrierScore,
    calcCalibration,
    calcOutcomeVariance,
    calcResolution,
    convertCalibration,
    renameDfColumns,
)


# The task confidence scale has 4 keys mapped onto the .5-1 probability range.
# Passed explicitly to every scoring call so the key-to-probability mapping is
# fixed for all subjects rather than inferred from each subject's max key used.
N_CONFIDENCE_KEYS = 4

TASK_METRICS = [
    "accuracy",
    "brier_score",
    "calibration",
    "resolution",
    "outcome_variance",
    "rewarded_score",
]

STRATEGY_FORMULAS = {
    "additive": "C(classification_strategy) + C(decision_strategy) + C(test)",
    "two_way": "(C(classification_strategy) + C(decision_strategy) + C(test)) ** 2",
    "full_factorial": "C(classification_strategy) * C(decision_strategy) * C(test)",
}

STRATEGY_TERM_LABELS = {
    "C(classification_strategy)": "Classification strategy",
    "C(decision_strategy)": "Decision strategy",
    "C(test)": "Session",
    "C(classification_strategy):C(decision_strategy)": "Classification x Decision",
    "C(classification_strategy):C(test)": "Classification x Session",
    "C(decision_strategy):C(test)": "Decision x Session",
}

TEST_LABELS = {
    0: "Initial Test",
    1: "1 Week Retest",
    2: "Final Retest",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "brier_score": "Brier Score",
    "calibration": "Calibration",
    "resolution": "Resolution",
    "outcome_variance": "Outcome Variance",
    "rewarded_score": "Rewarded Score",
    "calibration_converted_score": "Converted Calibration",
}

STRATEGY_RESPONSE_LABELS = {
    "I paid attention to a SUBSET OF COMPONENTS and ignored the others": "Subset of components",
    "I paid attention to a ALL COMPONENTS": "All components",
    "I DID NOT HAVE a strategy": "No strategy",
    "I WROTE DOWN the components and classifications": "Wrote down",
    "I CAREFULLY EVALUATED the options": "Carefully evaluated",
    "I trusted MY GUT": "Trusted gut",
    "I chose RANDOMLY": "Chose randomly",
}

QUESTIONNAIRE_INSTRUMENT_GROUPS = {
    "supplemental_figure_2": ["bfi10_extraversion", "bfi10_agreeableness", "bfi10_conscientiousness"],
    "supplemental_figure_3": ["bfi10_neuroticism", "bfi10_openness"],
    "supplemental_figure_4": ["bapq_aloof", "bapq_pragmatic_language", "bapq_rigid"],
    "supplemental_figure_5": ["asrs_inattention", "asrs_hyperactivity_impulsivity"],
    "supplemental_figure_6": ["phq9_na"],
}

SUBSCALE_LABELS = {
    "bfi10_extraversion": "BFI-10: Extraversion",
    "bfi10_agreeableness": "BFI-10: Agreeableness",
    "bfi10_conscientiousness": "BFI-10: Conscientiousness",
    "bfi10_neuroticism": "BFI-10: Neuroticism",
    "bfi10_openness": "BFI-10: Openness",
    "bapq_aloof": "BAPQ: Aloof",
    "bapq_pragmatic_language": "BAPQ: PragLang",
    "bapq_rigid": "BAPQ: Rigidity",
    "asrs_inattention": "ASRS: Inattention",
    "asrs_hyperactivity_impulsivity": "ASRS: Hyperactivity/Impulsivity",
    "phq9_na": "PHQ9",
}

# US letter page, portrait, in inches. Supplemental PDFs are saved at exactly
# this size so each file drops into the supplement as a full page.
LETTER_PORTRAIT = (8.5, 11)

PROTOTYPE_CODES = pd.DataFrame(
    {
        "prototype": ["A", "B", "A", "B"],
        "retest": [1, 1, 2, 2],
        "name": [
            "0-A-1_1-B-1_2-C-1_3-D-2_4-E-1_5-A-3_6-B-3_7-C-3_8-D-4_9-E-3",
            "0-A-2_1-B-2_2-C-2_3-D-1_4-E-2_5-A-4_6-B-4_7-C-4_8-D-3_9-E-4",
            "0-A-1_1-B-2_2-C-1_3-D-2_4-E-1_5-A-4_6-B-4_7-C-3_8-D-4_9-E-4",
            "0-A-2_1-B-1_2-C-2_3-D-1_4-E-2_5-A-3_6-B-3_7-C-4_8-D-3_9-E-3",
        ],
    }
)


@dataclass(frozen=True)
class PaperData:
    questionnaire_items: pd.DataFrame
    questionnaire_overall: pd.DataFrame
    questionnaire_subscales: pd.DataFrame
    initial: pd.DataFrame
    retest_1: pd.DataFrame
    retest_2: pd.DataFrame
    all_task_rows: pd.DataFrame
    completer_task_rows: pd.DataFrame


def ensure_output_dirs(base_dir: str | Path = "paper_repro_outputs") -> dict[str, Path]:
    base = Path(base_dir)
    paths = {
        "base": base,
        "figures": base / "figures",
        "tables": base / "tables",
        "supplement": base / "supplement",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_paper_data(data_dir: str | Path = "data") -> PaperData:
    data_dir = Path(data_dir)
    questionnaire_items = pd.read_csv(data_dir / "questionnaire_data.csv")
    questionnaire_overall = pd.read_csv(data_dir / "questionnaire_overall_scores.csv")
    questionnaire_subscales = pd.read_csv(data_dir / "questionnaire_subscale_scores.csv")

    initial = pd.read_csv(data_dir / "task_data_initial_test.csv")
    retest_1 = pd.read_csv(data_dir / "task_data_retest_1.csv")
    retest_2 = pd.read_csv(data_dir / "task_data_retest_2.csv")
    initial["test"] = 0
    retest_1["test"] = 1
    retest_2["test"] = 2

    all_task_rows = pd.concat([initial, retest_1, retest_2], ignore_index=True)
    completer_ids = set(retest_2["subject_id"].unique())
    initial = initial.loc[initial["subject_id"].isin(completer_ids)].copy()
    retest_1 = retest_1.loc[retest_1["subject_id"].isin(completer_ids)].copy()
    completer_task_rows = pd.concat([initial, retest_1, retest_2], ignore_index=True)

    return PaperData(
        questionnaire_items=questionnaire_items,
        questionnaire_overall=questionnaire_overall,
        questionnaire_subscales=questionnaire_subscales,
        initial=initial,
        retest_1=retest_1,
        retest_2=retest_2,
        all_task_rows=all_task_rows,
        completer_task_rows=completer_task_rows,
    )


def session_completeness_table(data: PaperData) -> pd.DataFrame:
    rows = []
    for test, df in [(0, data.initial), (1, data.retest_1), (2, data.retest_2)]:
        rows.append(
            {
                "test": test,
                "test_label": TEST_LABELS[test],
                "n_subjects": df["subject_id"].nunique(),
                "n_rows": df.shape[0],
                "n_test_trials_per_subject": int(
                    df.loc[df["block"] == "test"].groupby("subject_id").size().median()
                ),
                "n_unique_test_trials_per_subject": int(
                    df.loc[df["block"] == "test"]
                    .drop_duplicates(["subject_id", "trial_number"])
                    .groupby("subject_id")
                    .size()
                    .median()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_task_session(df: pd.DataFrame, correct_rewarded_score: bool = True) -> pd.DataFrame:
    rows = []
    for subject_id in df["subject_id"].unique():
        data_subject = df.loc[df["subject_id"] == subject_id]
        test_trials = data_subject.loc[data_subject["block"] == "test"]
        estimates = test_trials["confidence"].to_numpy().astype(int)
        outcomes = test_trials["correct"].to_numpy().astype(int)
        accuracy = test_trials["correct"].mean()
        calibration_converted = convertCalibration(estimates, outcomes, n_keys=N_CONFIDENCE_KEYS)
        # Retrospective analysis composite; this does not reconstruct the
        # historical webapp display or establish the bonuses actually paid.
        rewarded_score = (accuracy + calibration_converted) / 2
        if correct_rewarded_score:
            rewarded_score = max(rewarded_score, 0)

        rows.append(
            {
                "subject_id": float(subject_id),
                "accuracy": accuracy,
                "brier_score": calcBrierScore(estimates, outcomes, n_keys=N_CONFIDENCE_KEYS),
                "calibration": calcCalibration(estimates, outcomes, n_keys=N_CONFIDENCE_KEYS),
                "resolution": calcResolution(estimates, outcomes, n_keys=N_CONFIDENCE_KEYS),
                "outcome_variance": calcOutcomeVariance(outcomes),
                "calibration_converted_score": calibration_converted,
                "rewarded_score": rewarded_score,
            }
        )
    return pd.DataFrame(rows)


def build_task_summaries(data: PaperData) -> pd.DataFrame:
    frames = []
    for test, df in [(0, data.initial), (1, data.retest_1), (2, data.retest_2)]:
        summary = summarize_task_session(df)
        summary["test"] = test
        summary["test_label"] = TEST_LABELS[test]
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def above_chance_subjects(task_summaries: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    accuracy = task_summaries.pivot(index="subject_id", columns="test", values="accuracy")
    return accuracy.index[(accuracy > threshold).all(axis=1)].to_numpy()


def less_stringent_performance_subjects(task_summaries: pd.DataFrame) -> np.ndarray:
    accuracy = task_summaries.pivot(index="subject_id", columns="test", values="accuracy")
    return accuracy.index[(accuracy > 0.55).sum(axis=1) >= 2].to_numpy()


def sample_definition_table(task_summaries: pd.DataFrame) -> pd.DataFrame:
    accuracy = task_summaries.pivot(index="subject_id", columns="test", values="accuracy")
    rows = [
        ("All three sessions completed", np.repeat(True, len(accuracy))),
        ("Accuracy > .5 in all three test sessions", (accuracy > 0.5).all(axis=1)),
        ("Accuracy >= .5 in all three test sessions", (accuracy >= 0.5).all(axis=1)),
        ("Accuracy > .5 in at least two test sessions", (accuracy > 0.5).sum(axis=1) >= 2),
        ("Mean test accuracy > .5 across sessions", accuracy.mean(axis=1) > 0.5),
        ("Less stringent sensitivity rule: accuracy > .55 in at least two sessions", (accuracy > 0.55).sum(axis=1) >= 2),
    ]
    return pd.DataFrame({"definition": [r[0] for r in rows], "n": [int(np.sum(r[1])) for r in rows]})


def accuracy_calibration_correlations(task_summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test, test_df in task_summaries.groupby("test"):
        r, p = pearsonr(test_df["accuracy"], test_df["calibration"])
        rows.append({"test": test, "test_label": TEST_LABELS[test], "r": r, "p": p, "n": test_df.shape[0]})
    return pd.DataFrame(rows)


def calibration_change_tests(task_summaries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = (
        task_summaries.groupby(["test", "test_label"])["calibration"]
        .mean()
        .reset_index(name="mean_calibration")
    )
    anova = pg.rm_anova(
        data=task_summaries,
        dv="calibration",
        within="test",
        subject="subject_id",
        correction=True,
        detailed=True,
    )
    wide = task_summaries.pivot(index="subject_id", columns="test", values="calibration")
    t_stat, p_one_sided = ttest_rel(wide[0], wide[2], alternative="greater")
    diff = wide[0] - wide[2]
    paired = pd.DataFrame(
        [
            {
                "comparison": "Initial > Final",
                "t": t_stat,
                "p_one_sided": p_one_sided,
                "cohens_dz": diff.mean() / diff.std(ddof=1),
                "mean_initial": wide[0].mean(),
                "mean_final": wide[2].mean(),
                "n": wide.shape[0],
            }
        ]
    )
    return means, anova, paired


def icc_table(task_summaries: pd.DataFrame, metrics: Iterable[str] = TASK_METRICS) -> pd.DataFrame:
    """ICC(2,k): absolute agreement of the mean across the k administrations.

    This is not the reliability of a single administration. Retain Pingouin's
    existing intervals, and export their bounds separately for tables/figures.
    """
    if task_summaries.duplicated(["subject_id", "test"]).any():
        raise ValueError("ICC requires one observation per participant and administration.")
    k = task_summaries["test"].nunique()
    if not task_summaries.groupby("subject_id")["test"].nunique().eq(k).all():
        raise ValueError("ICC requires the same administrations for every participant.")
    rows = []
    for metric in metrics:
        icc = pg.intraclass_corr(
            data=task_summaries,
            raters="test",
            targets="subject_id",
            ratings=metric,
        )
        selected = icc.loc[(icc["Type"] == "ICC2k") & (icc["Description"] == "Average random raters")].iloc[0]
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "icc_type": selected["Type"],
                "icc_model": "Two-way random effects, absolute agreement",
                "unit": f"Mean of {k} administrations",
                "k": k,
                "n_subjects": task_summaries["subject_id"].nunique(),
                "icc": selected["ICC"],
                "f": selected["F"],
                "p": selected["pval"],
                "ci95": selected["CI95%"],
                "ci95_low": float(selected["CI95%"][0]),
                "ci95_high": float(selected["CI95%"][1]),
            }
        )
    return pd.DataFrame(rows)


def strategy_count_tables(completer_task_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    strategy = completer_task_rows[
        ["subject_id", "strategy_prompt", "strategy_response", "test"]
    ].drop_duplicates()

    tables: dict[str, pd.DataFrame] = {}
    for prompt in ["strategy_classification", "decision_process"]:
        table = (
            strategy.loc[strategy["strategy_prompt"] == prompt]
            .groupby(["test", "strategy_response"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        table.insert(1, "test_label", table["test"].map(TEST_LABELS))
        tables[prompt] = table

    joint_tables = []
    for test in sorted(strategy["test"].unique()):
        pivot = strategy.loc[strategy["test"] == test].pivot(
            index="subject_id", columns="strategy_prompt", values="strategy_response"
        )
        crosstab = pd.crosstab(pivot["decision_process"], pivot["strategy_classification"])
        crosstab.insert(0, "decision_process", crosstab.index)
        crosstab.insert(0, "test_label", TEST_LABELS[test])
        crosstab.insert(0, "test", test)
        joint_tables.append(crosstab.reset_index(drop=True))
    tables["joint"] = pd.concat(joint_tables, ignore_index=True)
    return tables


def get_retest_distances(df: pd.DataFrame, retest: int) -> np.ndarray:
    prototype_a = PROTOTYPE_CODES.loc[
        (PROTOTYPE_CODES["prototype"] == "A") & (PROTOTYPE_CODES["retest"] == retest), "name"
    ].iloc[0].split("_")
    prototype_b = PROTOTYPE_CODES.loc[
        (PROTOTYPE_CODES["prototype"] == "B") & (PROTOTYPE_CODES["retest"] == retest), "name"
    ].iloc[0].split("_")

    distances = np.zeros((df.shape[0], 2))
    for i, name in enumerate(df["name"].to_list()):
        stim_code = name.split("_")
        distances[i, 0] = sum(proto != stim for proto, stim in zip(prototype_a, stim_code))
        distances[i, 1] = sum(proto != stim for proto, stim in zip(prototype_b, stim_code))
    return np.min(distances, axis=1)


def add_feature_distance(df: pd.DataFrame, test: int) -> pd.DataFrame:
    """Feature distances for main-task testing trials, excluding training/tutorials."""
    data_slice = df.loc[(df["use"] == "task") & (df["block"] == "test")].copy()
    if test == 0:
        data_slice.loc[data_slice["name"].str.contains("prototype"), "distance"] = 0
        for distance in [1, 2, 3, 4]:
            data_slice.loc[data_slice["name"].str.contains(f"d{distance}"), "distance"] = distance
    else:
        data_slice["distance"] = get_retest_distances(data_slice, retest=test)
    data_slice["distance"] = data_slice["distance"].astype(int)
    return data_slice


def fit_distance_models(data: PaperData, subject_ids: Iterable[float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each session's participant random-intercept model on testing trials only."""
    subject_ids = None if subject_ids is None else set(subject_ids)
    model_rows = []
    grouped_frames = []
    for test, df in [(0, data.initial), (1, data.retest_1), (2, data.retest_2)]:
        data_slice = add_feature_distance(df, test)
        if subject_ids is not None:
            data_slice = data_slice.loc[data_slice["subject_id"].isin(subject_ids)]

        grouped = data_slice.groupby(["subject_id", "distance"])["correct"].mean().reset_index()
        grouped["test"] = test
        grouped["test_label"] = TEST_LABELS[test]
        grouped["block"] = "test"
        grouped_frames.append(grouped)

        model = smf.mixedlm("correct ~ distance", grouped, groups=grouped["subject_id"])
        if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
            raise ValueError(f"Feature-distance design is rank deficient for session {test}.")
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            result = model.fit()
        if not result.converged or not np.isfinite(result.params).all():
            raise RuntimeError(f"Feature-distance model did not converge for session {test}.")
        covariance = result.cov_params().loc[result.fe_params.index, result.fe_params.index].to_numpy()
        if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
            raise RuntimeError(f"Invalid feature-distance fixed-effect covariance for session {test}.")
        distance_ci = result.conf_int().loc["distance"]
        model_rows.append(
            {
                "test": test,
                "test_label": TEST_LABELS[test],
                "block": "test",
                "n_subjects": grouped["subject_id"].nunique(),
                "n_source_rows": len(data_slice),
                "n_unique_trials": len(data_slice.drop_duplicates(["subject_id", "trial_number"])),
                "distance_slope": result.params["distance"],
                "distance_se": result.bse["distance"],
                "distance_z": result.tvalues["distance"],
                "distance_p": result.pvalues["distance"],
                "distance_ci95_low": distance_ci.iloc[0],
                "distance_ci95_high": distance_ci.iloc[1],
                "scale": result.scale,
                "random_intercept_variance": result.cov_re.iloc[0, 0],
                "random_to_residual_variance": result.cov_re.iloc[0, 0] / result.scale,
                "converged": result.converged,
                "warnings": " | ".join(dict.fromkeys(str(w.message) for w in fit_warnings)),
            }
        )
    return pd.DataFrame(model_rows), pd.concat(grouped_frames, ignore_index=True)


def subject_distance_slopes(grouped_distance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (test, subject_id), group in grouped_distance.groupby(["test", "subject_id"]):
        slope, intercept, r_value, p_value, stderr = linregress(group["distance"], group["correct"])
        rows.append(
            {
                "test": test,
                "test_label": TEST_LABELS[test],
                "subject_id": subject_id,
                "slope": slope,
                "intercept": intercept,
                "r": r_value,
                "p": p_value,
                "stderr": stderr,
            }
        )
    return pd.DataFrame(rows)


def mean_task_summaries(task_summaries: pd.DataFrame) -> pd.DataFrame:
    return (
        task_summaries.groupby("subject_id")[
            TASK_METRICS + ["calibration_converted_score"]
        ]
        .mean()
        .reset_index()
    )


def questionnaire_task_correlations(
    questionnaire_subscales: pd.DataFrame,
    task_summaries: pd.DataFrame,
    subject_ids: Iterable[float] | None = None,
) -> pd.DataFrame:
    task_mean = mean_task_summaries(task_summaries)
    joint = pd.merge(questionnaire_subscales, task_mean, on="subject_id", how="inner")
    if subject_ids is not None:
        joint = joint.loc[joint["subject_id"].isin(set(subject_ids))]

    rows = []
    for subscale in sorted(joint["subscale"].dropna().unique()):
        subscale_df = joint.loc[joint["subscale"] == subscale]
        for metric in TASK_METRICS:
            r, p = pearsonr(subscale_df["score"], subscale_df[metric])
            rows.append(
                {
                    "subscale": subscale,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n": subscale_df.shape[0],
                    "r": r,
                    "p_uncorrected": p,
                }
            )
    results = pd.DataFrame(rows)
    _, bonf_p, _, _ = multipletests(results["p_uncorrected"], method="bonferroni")
    _, fdr_p, _, _ = multipletests(results["p_uncorrected"], method="fdr_bh")
    results["p_bonferroni"] = bonf_p
    results["p_fdr_bh"] = fdr_p
    results["significant_uncorrected"] = results["p_uncorrected"] < 0.05
    results["significant_bonferroni"] = results["p_bonferroni"] < 0.05
    results["significant_fdr_bh"] = results["p_fdr_bh"] < 0.05
    return results.sort_values(["p_uncorrected", "subscale", "metric"]).reset_index(drop=True)


def selected_questionnaire_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    selected = [
        ("bfi10_extraversion", "brier_score"),
        ("bfi10_extraversion", "resolution"),
        ("bfi10_agreeableness", "resolution"),
        ("bapq_aloof", "resolution"),
        ("bfi10_extraversion", "outcome_variance"),
    ]
    mask = pd.Series(False, index=correlations.index)
    for subscale, metric in selected:
        mask |= (correlations["subscale"] == subscale) & (correlations["metric"] == metric)
    return correlations.loc[mask].copy()


def pearson_required_n_fisher_z(
    r: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    z_effect = abs(np.arctanh(r))
    z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    z_power = norm.ppf(power)
    return int(np.ceil(((z_alpha + z_power) / z_effect) ** 2 + 3))


def pearson_power_nct_approx(n: int, r: float, alpha: float = 0.05, two_sided: bool = True) -> float:
    df = n - 2
    noncentrality = r * np.sqrt(df / (1 - r**2))
    if two_sided:
        critical_t = t.ppf(1 - alpha / 2, df)
        return nct.sf(critical_t, df, noncentrality) + nct.cdf(-critical_t, df, noncentrality)
    critical_t = t.ppf(1 - alpha, df)
    return nct.sf(critical_t, df, noncentrality)


def pearson_required_n_nct_approx(
    r: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    max_n: int = 10000,
) -> int:
    for n in range(4, max_n + 1):
        if pearson_power_nct_approx(n, r, alpha=alpha, two_sided=two_sided) >= power:
            return n
    raise ValueError(f"Required N exceeds max_n={max_n}")


def power_analysis_tables(
    target_rs: Iterable[float] = (0.20, 0.25, 0.30),
    powers: Iterable[float] = (0.80, 0.90),
    n_tests: int = 66,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conservative_alpha = 0.05 / n_tests
    summary_rows = []
    for target_r in target_rs:
        for target_power in powers:
            summary_rows.append(
                {
                    "target_r": target_r,
                    "power": target_power,
                    "alpha": 0.05,
                    "correction_family": "single planned test",
                    "required_n_fisher_z": pearson_required_n_fisher_z(target_r, alpha=0.05, power=target_power),
                    "required_n_nct": pearson_required_n_nct_approx(target_r, alpha=0.05, power=target_power),
                }
            )
            summary_rows.append(
                {
                    "target_r": target_r,
                    "power": target_power,
                    "alpha": conservative_alpha,
                    "correction_family": f"conservative corrected threshold, m={n_tests}",
                    "required_n_fisher_z": pearson_required_n_fisher_z(
                        target_r, alpha=conservative_alpha, power=target_power
                    ),
                    "required_n_nct": pearson_required_n_nct_approx(
                        target_r, alpha=conservative_alpha, power=target_power
                    ),
                }
            )

    current_power_rows = []
    for target_r in target_rs:
        current_power_rows.append(
            {
                "target_r": target_r,
                "n": 80,
                "alpha": 0.05,
                "correction_family": "single planned test",
                "power_nct": pearson_power_nct_approx(80, target_r, alpha=0.05),
            }
        )
        current_power_rows.append(
            {
                "target_r": target_r,
                "n": 80,
                "alpha": conservative_alpha,
                "correction_family": f"conservative corrected threshold, m={n_tests}",
                "power_nct": pearson_power_nct_approx(80, target_r, alpha=conservative_alpha),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(current_power_rows)


def questionnaire_subscale_covariance(questionnaire_subscales: pd.DataFrame) -> pd.DataFrame:
    vals = questionnaire_subscales.pivot_table(
        values="score", index="subject_id", columns="subscale", aggfunc="first"
    )
    vals = vals.apply(lambda col: (col - col.mean()) / col.std(ddof=0), axis=0)
    vals = renameDfColumns(vals.copy())
    return vals.corr()


def save_table(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def save_indexed_table(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def plot_accuracy_calibration(correlations: pd.DataFrame, task_summaries: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for test, test_df in task_summaries.groupby("test"):
        corr_row = correlations.loc[correlations["test"] == test].iloc[0]
        sns.regplot(
            data=test_df,
            x="accuracy",
            y="calibration",
            ax=ax,
            label=f"{TEST_LABELS[test]}, r={corr_row['r']:.2f}, p={corr_row['p']:.3g}",
        )
    ax.set_xlabel("Accuracy (higher better)")
    ax.set_ylabel("Calibration (lower better)")
    ax.set_title("Accuracy vs Calibration")
    ax.legend(fontsize=8)
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_icc_reliability(task_summaries: pd.DataFrame, icc: pd.DataFrame, path: str | Path):
    plot_df = task_summaries.rename(columns=METRIC_LABELS).copy()
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()
    for i, metric in enumerate(TASK_METRICS):
        label = METRIC_LABELS[metric]
        sns.violinplot(data=plot_df, x="test_label", y=label, color="lightgray", ax=axes[i])
        sns.stripplot(data=plot_df, x="test_label", y=label, color="black", alpha=0.35, size=2, ax=axes[i])
        estimate = icc.loc[icc["metric"] == metric].iloc[0]
        axes[i].set_title(
            f"{label}\nICC(2,{int(estimate['k'])})={estimate['icc']:.2f} "
            f"(95% CI {estimate['ci95_low']:.2f} to {estimate['ci95_high']:.2f})",
            fontsize=10,
        )
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
    sns.despine()
    fig.suptitle("Reliability of the mean across three administrations", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_strategy_counts(strategy_tables: dict[str, pd.DataFrame], path: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "strategy_classification", "Attention Strategy"),
        (axes[1], "decision_process", "Decision Strategy"),
    ]:
        table = strategy_tables[key].melt(
            id_vars=["test", "test_label"], var_name="strategy_response", value_name="count"
        )
        sns.barplot(data=table, x="test_label", y="count", hue="strategy_response", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_distance_regression(grouped_distance: pd.DataFrame, path: str | Path):
    slopes = subject_distance_slopes(grouped_distance)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for test in [0, 1, 2]:
        line_ax = axes[0, test]
        hist_ax = axes[1, test]
        test_grouped = grouped_distance.loc[grouped_distance["test"] == test]
        for _, subject_group in test_grouped.groupby("subject_id"):
            sns.regplot(
                data=subject_group,
                x="distance",
                y="correct",
                ci=None,
                scatter=False,
                line_kws={"alpha": 0.25, "linewidth": 1},
                ax=line_ax,
            )
        sns.scatterplot(
            data=test_grouped.groupby("distance")["correct"].mean().reset_index(),
            x="distance",
            y="correct",
            color="black",
            ax=line_ax,
        )
        line_ax.set_title(TEST_LABELS[test])
        line_ax.set_xlabel("Feature Distance")
        line_ax.set_ylabel("P(Correct)")

        sns.histplot(data=slopes.loc[slopes["test"] == test], x="slope", bins=18, ax=hist_ax)
        hist_ax.axvline(0, color="black", linestyle="--", linewidth=1)
        hist_ax.set_xlabel("Subject Slope")
        hist_ax.set_ylabel("Count")
    sns.despine()
    fig.suptitle(f"Testing trials only (N={grouped_distance['subject_id'].nunique()})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def plot_questionnaire_covariance(corr: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Questionnaire Subscale Correlations")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig


def strategy_metric_table(data: PaperData, task_summaries: pd.DataFrame) -> pd.DataFrame:
    """One row per subject and session with short-labeled strategy responses and task metrics."""
    strategy = data.completer_task_rows[
        ["subject_id", "strategy_prompt", "strategy_response", "test"]
    ].drop_duplicates()
    wide = strategy.pivot(
        index=["subject_id", "test"], columns="strategy_prompt", values="strategy_response"
    ).reset_index()
    wide["classification_strategy"] = wide["strategy_classification"].map(STRATEGY_RESPONSE_LABELS)
    wide["decision_strategy"] = wide["decision_process"].map(STRATEGY_RESPONSE_LABELS)
    return wide[["subject_id", "test", "classification_strategy", "decision_strategy"]].merge(
        task_summaries, on=["subject_id", "test"], validate="one_to_one"
    )


def _strategy_analysis_rows(
    strategy_metrics: pd.DataFrame,
    exclude_singletons: Iterable[str] = ("Wrote down",),
) -> pd.DataFrame:
    """Keep the established singleton exclusion; check rank separately."""
    excluded = set(exclude_singletons)
    df = strategy_metrics.loc[
        ~strategy_metrics["classification_strategy"].isin(excluded)
        & ~strategy_metrics["decision_strategy"].isin(excluded)
    ].copy()
    if df.empty or df[["subject_id", "test", "classification_strategy", "decision_strategy"]].isna().any().any():
        raise ValueError("Strategy analysis requires nonmissing participant, session, and strategy values.")
    if df.duplicated(["subject_id", "test"]).any():
        raise ValueError("Strategy analysis requires one row per participant and session.")
    return df


def strategy_design_audit(strategy_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report all cells (including zeros) and ranks before model fitting.

    Lower-order models impose restrictions on interactions; their full rank
    does not make all effects in the original saturated model estimable.
    """
    df = _strategy_analysis_rows(strategy_metrics)
    factors = ["classification_strategy", "decision_strategy", "test"]
    index = pd.MultiIndex.from_product([sorted(df[c].unique()) for c in factors], names=factors)
    cells = df.groupby(factors).size().reindex(index, fill_value=0).rename("n").reset_index()
    audits = []
    for name, formula in STRATEGY_FORMULAS.items():
        design = dmatrix(formula, df, NA_action="raise", return_type="dataframe")
        rank = np.linalg.matrix_rank(design)
        audits.append({
            "model": name, "formula": formula, "n_rows": len(df),
            "n_subjects": df["subject_id"].nunique(),
            "n_excluded_rows": len(strategy_metrics) - len(df),
            "n_columns": design.shape[1], "rank": rank,
            "rank_deficiency": design.shape[1] - rank,
            "full_rank": rank == design.shape[1],
            "condition_number": np.linalg.cond(design),
            "possible_cells": len(cells), "observed_cells": int(cells["n"].gt(0).sum()),
        })
    return pd.DataFrame(audits), cells


def strategy_metric_models(
    strategy_metrics: pd.DataFrame,
    metrics: Iterable[str] = TASK_METRICS,
    model_kind: str = "additive",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Participant random-intercept models fitted by REML.

    The primary additive model tests the three factors adjusted for one another.
    The two-way sensitivity model tests only interactions (not reference-level
    main effects). Each reported omnibus test is an asymptotic Wald chi-square
    test. BH-FDR is applied across all metric-by-term tests in each returned
    table, separately for the primary and sensitivity analyses.

    The full factorial model is available for rank auditing only. There are
    empty cells in the paper data, so its intended 27-parameter specification
    cannot be identified. Never silently fit it with a generalized inverse.
    """
    if model_kind not in ("additive", "two_way"):
        raise ValueError("Use additive or two_way; the full factorial is for auditing only.")
    df = _strategy_analysis_rows(strategy_metrics)
    metrics = list(metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Supply a nonempty list of distinct task metrics.")
    if not np.isfinite(df[metrics].to_numpy(dtype=float)).all():
        raise ValueError("Strategy outcomes must be finite; missing outcomes require an explicit sample decision.")
    tests, coefficients, diagnostics = [], [], []
    for metric in metrics:
        formula = f"{metric} ~ {STRATEGY_FORMULAS[model_kind]}"
        model = smf.mixedlm(formula, df, groups=df["subject_id"], re_formula="1", missing="raise")
        rank = np.linalg.matrix_rank(model.exog)
        if rank != model.exog.shape[1]:
            raise ValueError(f"{model_kind} design is rank deficient: rank {rank} / {model.exog.shape[1]}.")
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            result = model.fit(reml=True, method="bfgs", maxiter=1000)
        if not result.converged or not np.isfinite(result.params).all():
            raise RuntimeError(f"Strategy model did not converge for {metric} ({model_kind}).")
        covariance = result.cov_params().loc[result.fe_params.index, result.fe_params.index].to_numpy()
        if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
            raise RuntimeError(f"Invalid fixed-effect covariance for {metric} ({model_kind}).")
        metadata = {"model": model_kind, "metric": metric, "metric_label": METRIC_LABELS[metric]}
        for term, term_slice in model.data.design_info.term_name_slices.items():
            if term == "Intercept" or (model_kind == "two_way" and ":" not in term):
                continue
            # Include zeros for the variance parameter; test only fixed effects.
            contrast = np.zeros((term_slice.stop - term_slice.start, len(result.params)))
            contrast[:, term_slice] = np.eye(contrast.shape[0])
            tested_covariance = contrast[:, :model.k_fe] @ covariance @ contrast[:, :model.k_fe].T
            if np.linalg.matrix_rank(tested_covariance) != contrast.shape[0]:
                raise RuntimeError(f"Nonestimable omnibus test for {metric}: {term}.")
            test = result.wald_test(contrast, use_f=False, scalar=True)
            tests.append({**metadata, "term": STRATEGY_TERM_LABELS[term],
                          "df": contrast.shape[0], "chi2": float(test.statistic),
                          "p": float(test.pvalue)})
        ci = result.conf_int()
        for term in result.fe_params.index:
            coefficients.append({**metadata, "coefficient": term,
                                 "estimate": result.fe_params[term], "se": result.bse_fe[term],
                                 "ci95_low": ci.loc[term, 0], "ci95_high": ci.loc[term, 1]})
        diagnostics.append({**metadata, "formula": formula, "fit_method": "REML", "optimizer": "bfgs",
                            "n_rows": int(result.nobs), "n_subjects": df["subject_id"].nunique(),
                            "rank": rank, "n_columns": model.exog.shape[1],
                            "condition_number": np.linalg.cond(model.exog), "converged": result.converged,
                            "random_intercept_variance": result.cov_re.iloc[0, 0],
                            "residual_variance": result.scale,
                            "random_to_residual_variance": result.cov_re.iloc[0, 0] / result.scale,
                            "warnings": " | ".join(dict.fromkeys(str(w.message) for w in fit_warnings))})
    tests = pd.DataFrame(tests)
    tests["p_fdr_bh"] = multipletests(tests["p"], method="fdr_bh")[1]
    tests["fdr_family_size"] = len(tests)
    tests["significant_uncorrected"] = tests["p"] < 0.05
    tests["significant_fdr_bh"] = tests["p_fdr_bh"] < 0.05
    return tests, pd.DataFrame(coefficients), pd.DataFrame(diagnostics)


def save_supplemental_table_1_pdf(
    model_table: pd.DataFrame, path: str | Path, n_rows: int, n_subjects: int,
):
    """Render the corrected primary strategy model tests on a letter page."""
    fig = plt.figure(figsize=LETTER_PORTRAIT)
    ax = fig.add_axes([0.07, 0.25, 0.86, 0.58])
    ax.axis("off")

    cell_rows = []
    for metric_label, metric_df in model_table.groupby("metric_label", sort=False):
        for i, (_, row) in enumerate(metric_df.iterrows()):
            # Four decimals avoid turning values such as .0495 into .050.
            p_txt = f"{row['p']:.4f}" if row["p"] >= 0.0001 else f"{row['p']:.1e}"
            q_txt = f"{row['p_fdr_bh']:.4f}" if row["p_fdr_bh"] >= 0.0001 else f"{row['p_fdr_bh']:.1e}"
            cell_rows.append(
                [metric_label if i == 0 else "", row["term"], f"{row['df']:.0f}", f"{row['chi2']:.2f}", p_txt, q_txt]
            )

    table = ax.table(
        cellText=cell_rows,
        colLabels=["Task metric", "Term", "df", "Wald chi-square", "P", "BH-FDR P"],
        colWidths=[0.19, 0.29, 0.05, 0.19, 0.14, 0.14],
        cellLoc="left",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    for (row_i, _), cell in table.get_celld().items():
        cell.set_edgecolor("0.8")
        if row_i == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("0.93")

    fig.text(0.07, 0.945, "Supplemental Table 1", fontsize=13, fontweight="bold")
    fig.text(
        0.07,
        0.925,
        "Strategy associations adjusted for session. Separate linear mixed models for each task metric\n"
        "include classification strategy, decision strategy, and session as additive categorical fixed effects\n"
        "and a participant random intercept. Models were fitted using restricted maximum likelihood.",
        fontsize=9,
        va="top",
    )
    fig.text(
        0.07,
        0.18,
        f"N={n_rows} participant-sessions from {n_subjects} completers; the singleton 'Wrote down' report was excluded.\n"
        "All primary designs have rank 7 of 7. Tests are asymptotic Wald chi-square tests. P values are\n"
        f"reported both unadjusted and with Benjamini-Hochberg correction across {len(model_table)} tests.\n\n"
        "A separate sensitivity analysis includes all two-way interactions (rank 19 of 19). The original\n"
        "three-way specification has rank 24 of 27 because three cells are empty; it is not used for inference.\n"
        "Strategy associations are exploratory and do not establish causal effects.",
        fontsize=8,
        va="top",
        color="0.35",
    )
    fig.savefig(path)
    return fig


def plot_supplemental_figure_1(strategy_metrics: pd.DataFrame, path: str | Path):
    """Accuracy and outcome variance by reported strategy and session (Supplemental Figure 1)."""
    fig, axes = plt.subplots(2, 2, figsize=LETTER_PORTRAIT, gridspec_kw={
        "left": 0.09, "right": 0.97, "top": 0.85, "bottom": 0.17, "hspace": 0.5, "wspace": 0.25,
    })
    plot_df = strategy_metrics.copy()
    plot_df["test_label"] = plot_df["test"].map(TEST_LABELS)
    panels = [
        (axes[0, 0], "classification_strategy", "accuracy", "Accuracy by classification strategy", False),
        (axes[0, 1], "decision_strategy", "accuracy", "Accuracy by decision strategy", False),
        (axes[1, 0], "classification_strategy", "outcome_variance", "Outcome variance by classification strategy", True),
        (axes[1, 1], "decision_strategy", "outcome_variance", "Outcome variance by decision strategy", True),
    ]
    for ax, strategy_col, metric, title, show_legend in panels:
        sns.barplot(data=plot_df, x="test_label", y=metric, hue=strategy_col, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        if show_legend:
            ax.legend(fontsize=7, title=None, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
        else:
            ax.legend_.remove()
    sns.despine()
    fig.suptitle("Supplemental Figure 1", x=0.09, y=0.94, ha="left", fontsize=13, fontweight="bold")
    fig.text(
        0.09,
        0.915,
        "Task performance by self-reported strategy. Bars show the mean across participants reporting each\n"
        "strategy in each session; error bars show 95% confidence intervals.",
        fontsize=9,
        va="top",
    )
    fig.savefig(path)
    return fig


def plot_subscale_correlation_figures(
    questionnaire_subscales: pd.DataFrame,
    task_summaries: pd.DataFrame,
    out_dir: str | Path,
    figure_numbers_start: int = 2,
) -> dict[str, Path]:
    """Scatter panels of every questionnaire subscale against every mean task metric,
    split by instrument into letter-page PDFs (Supplemental Figures 2-6)."""
    out_dir = Path(out_dir)
    task_mean = mean_task_summaries(task_summaries)
    joint = pd.merge(questionnaire_subscales, task_mean, on="subject_id", how="inner")

    paths: dict[str, Path] = {}
    for fig_offset, (fig_name, subscales) in enumerate(QUESTIONNAIRE_INSTRUMENT_GROUPS.items()):
        fig, axes = plt.subplots(
            len(TASK_METRICS), 3, figsize=LETTER_PORTRAIT, gridspec_kw={
                "left": 0.09, "right": 0.97, "top": 0.88, "bottom": 0.05, "hspace": 0.55, "wspace": 0.35,
            },
        )
        for col in range(3):
            for row in range(len(TASK_METRICS)):
                ax = axes[row, col]
                if col >= len(subscales):
                    ax.axis("off")
                    continue
                subscale = subscales[col]
                metric = TASK_METRICS[row]
                subscale_df = joint.loc[joint["subscale"] == subscale]
                r, p = pearsonr(subscale_df["score"], subscale_df[metric])
                sns.regplot(
                    data=subscale_df, x="score", y=metric, ax=ax,
                    scatter_kws={"s": 8, "alpha": 0.55}, line_kws={"linewidth": 1.2},
                )
                p_txt = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
                ax.set_title(f"r={r:.2f}, p={p_txt}", fontsize=8)
                ax.set_xlabel(SUBSCALE_LABELS[subscale] if row == len(TASK_METRICS) - 1 else "", fontsize=8)
                ax.set_ylabel(METRIC_LABELS[metric] if col == 0 else "", fontsize=8)
                ax.tick_params(labelsize=7)
        sns.despine()
        figure_number = figure_numbers_start + fig_offset
        subscale_names = ", ".join(SUBSCALE_LABELS[s] for s in subscales)
        fig.suptitle(f"Supplemental Figure {figure_number}", x=0.09, y=0.955, ha="left", fontsize=13, fontweight="bold")
        fig.text(
            0.09,
            0.935,
            f"{subscale_names} versus task metrics.\n"
            f"Each point is one of the {task_mean.shape[0]} completers; task metrics are averaged across the three sessions.\n"
            "r and p are uncorrected Pearson correlations.",
            fontsize=9,
            va="top",
        )
        path = out_dir / f"{fig_name}.pdf"
        fig.savefig(path)
        paths[fig_name] = path
        plt.close(fig)
    return paths


def plot_questionnaire_task_heatmap(correlations: pd.DataFrame, path: str | Path):
    plot_df = correlations.pivot(index="subscale", columns="metric_label", values="r")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(plot_df, cmap="coolwarm", center=0, vmin=-0.5, vmax=0.5, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Questionnaire Subscale vs Mean Task Metric Correlations")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
