"""Scoring functions for the metacognition drone-recon task.

Confidence is reported on a fixed 4-key scale. Keys 1-4 are mapped to the
midpoints of four equal bins on the .5-1 probability range (.5625, .6875,
.8125, .9375), and the mapping is identical for all subjects and sessions.
The Brier score is the trial-weighted mean squared error between the mapped
confidence probability and the outcome, and satisfies the Murphy
decomposition PS = C - R + O with the calibration (C), resolution (R), and
outcome-variance (O) functions below.
"""

import numpy as np


def renameDfColumns(df, key_subscales=None, key_plot_names=None):
    """Renames the columns of a dataframe to nicer-formatted names for plotting

    Args:
        df (pandas.DataFrame): dataframe whose columns should be renamed
        key_subscales (list, optional): Columns to rename. If None, it renames several subscales. Defaults to None.
        key_plot_names (list, optional): The new names. If None, it provides the corresponding subscales. Defaults to None.

    Returns:
        pandas.DataFrame: Same dataframe with renamed columns
    """
    if key_subscales is None:
        key_subscales = ['bfi10_extraversion', 'bfi10_agreeableness',
         'bfi10_conscientiousness', 'bfi10_neuroticism', 'bfi10_openness',
         'bapq_aloof', 'bapq_pragmatic_language', 'bapq_rigid',
         'asrs_inattention', 'asrs_hyperactivity_impulsivity', 'phq9_na']
    if key_plot_names is None:
        key_plot_names = ['BFI-10: Extraversion','BFI-10: Agreeableness','BFI-10: Conscientiousness',
                  'BFI-10: Neuroticism','BFI-10: Openness','BAPQ: Aloof','BAPQ: PragLang',
                  'BAPQ: Rigidity','ASRS: Inattention','ASRS: Hyperactivity/Impulsivity','PHQ9']
    df.rename(columns=dict(zip(key_subscales, key_plot_names)), inplace=True)
    return df


def binEstimatesOutcomes(estimates, outcomes, n_keys=4):
    """Bin the estimates and outcomes by confidence, and supply counts.

    Bins are defined by the fixed confidence scale (keys 1 through n_keys mapped
    onto the .5-1 probability range), not by the maximum key a given subject
    happened to use, so the key-to-probability mapping is identical for all
    subjects and sessions.

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates
        outcomes (list or numpy.array): vector of trial by trial outcomes
        n_keys (int, optional): number of keys on the confidence scale. Defaults to 4.

    Returns:
        numpy.arrays: estimates_binned, outcomes_binned, counts
    """
    estimates = np.asarray(estimates).astype(int)
    if np.min(estimates) < 1 or np.max(estimates) > n_keys:
        raise ValueError(f'Confidence estimates must be keys between 1 and n_keys={n_keys}')
    counts = [0] * n_keys
    correct = [0] * n_keys

    for i in range(len(estimates)):
        counts[estimates[i] - 1] += 1
        correct[estimates[i] - 1] += outcomes[i]

    estimates_binned_ = np.linspace(0.5, 1, n_keys + 1)
    estimates_binned = [(estimates_binned_[i] + estimates_binned_[i+1]) / 2 for i in range(len(estimates_binned_) - 1)]

    outcomes_binned = [n / counts[i] if counts[i] != 0 else 0 for i, n in enumerate(correct)]

    return np.array(estimates_binned), np.array(outcomes_binned), np.array(counts)


def convertConfidenceKeysToProbs(estimates, n_keys=4):
    """Creates confidence probabilities from confidence keys

    Uses the fixed n_keys confidence scale so the key-to-probability mapping
    does not depend on which keys a subject happened to use.

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates
        n_keys (int, optional): number of keys on the confidence scale. Defaults to 4.

    Returns:
        numpy.array: estimates_binned
    """
    estimates_binned_ = np.linspace(0.5, 1, n_keys + 1)
    estimates_binned = [(estimates_binned_[i] + estimates_binned_[i+1]) / 2 for i in range(len(estimates_binned_) - 1)]
    return np.array(estimates_binned)


def calcBrierScore(estimates, outcomes, by_trial=False, n_keys=4):
    """Calculates Brier score (averaged across trials) or probability score (not averaged across trials)

    The aggregate score is the mean across trials of the squared difference
    between each trial's confidence probability (key mapped onto the .5-1
    range via the fixed n_keys scale) and its outcome. This satisfies the
    Murphy decomposition PS = C - R + O with the calibration (C), resolution
    (R), and outcome-variance (O) functions in this module.

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates (keys)
        outcomes (list or numpy.array): vector of trial by trial outcomes
        by_trial (bool, optional): whether to calculate by trial or averaged across. Defaults to False.
            When by_trial is True, estimates must already be probabilities.
        n_keys (int, optional): number of keys on the confidence scale. Defaults to 4.

    Raises:
        ValueError: variable check for by_trial
        ValueError: equal length of estimate and outcome vectors

    Returns:
        float or vector of floats
    """
    try:
        assert(type(by_trial)==bool)
    except:
        raise ValueError('by_trial must be a boolean variable')
    if len(estimates) != len(outcomes):
        raise ValueError("The number of estimates and outcomes must be equal")
    if not by_trial:
        key_probs = convertConfidenceKeysToProbs(estimates, n_keys=n_keys)
        trial_probs = key_probs[np.asarray(estimates).astype(int) - 1]
        brier_score = np.mean((trial_probs - np.asarray(outcomes).astype(float))**2)
    elif by_trial:
        print('Calculating probability score (by trial)')
        brier_score = (estimates-outcomes)**2
    return brier_score


def calcCalibration(estimates, outcomes, n_keys=4):
    """Computes the calibration score (C) for a set of estimates and outcomes

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates
        outcomes (list or numpy.array): vector of trial by trial outcomes
        n_keys (int, optional): number of keys on the confidence scale. Defaults to 4.

    Returns:
        float: calibration score
    """
    estimates_binned, outcomes_binned, N_j = binEstimatesOutcomes(estimates, outcomes, n_keys=n_keys)
    C = np.sum(N_j*(estimates_binned-outcomes_binned)**2) / sum(N_j)
    return C


def convertCalibration(estimates, outcomes, n_keys=4):
    """Calculates calibration and converts it to a 0-1 scale, where 0 is the worst possible calibration and
       1 is the best possible calibration.

    The maximum error holds the participant's confidence keys and bin counts
    fixed. All mapped probabilities exceed .5, so every bin's squared error
    is maximized when all outcomes in that bin are incorrect. This is the
    retrospective analysis normalization, not a reconstruction of the score
    displayed by the historical webapp or of bonuses actually paid.

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates.
        outcomes (list or numpy.array): vector of trial by trial outcomes.
        n_keys (int, optional): number of keys on the confidence scale. Defaults to 4.

    Returns:
        float: converted calibration score
    """
    outcomes_worst = np.zeros_like(estimates, dtype=float)
    calibration_worst = calcCalibration(estimates, outcomes_worst, n_keys=n_keys)
    calibration = calcCalibration(estimates, outcomes, n_keys=n_keys)

    calibration_converted_score = 1 - (calibration / calibration_worst)
    calibration_converted_score = max(min(calibration_converted_score, 1),0)

    return calibration_converted_score


def calcOutcomeVariance(outcomes):
    """Calculates the variance of the outcomes (O) in the Brier score decomposition.

    Args:
        outcomes (list or numpy.array): vector of trial by trial outcomes

    Returns:
        float: outcome variance
    """
    O = np.mean(outcomes) * (1 - np.mean(outcomes))
    return O


def calcResolution(estimates, outcomes, n_keys=4):
    """Calculates the resolution score (R) for a set of estimates and outcomes in the Brier score decomposition.

    Args:
        estimates (list or numpy.array): vector of trial-by-trial confidence estimates.
        outcomes (list or numpy.array): vector of trial by trial outcomes.
        n_keys (int, optional): number of keys on the confidence scale. Defaults to 4.

    Returns:
        float: resolution score
    """
    _, outcomes_binned, N_j = binEstimatesOutcomes(estimates, outcomes, n_keys=n_keys)
    R = np.sum(N_j * (outcomes_binned - np.mean(outcomes))**2) / sum(N_j)
    return R
