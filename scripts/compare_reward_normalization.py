import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description='Compare full before/after notebook runs and independently verify scores.')
parser.add_argument('--baseline',required=True,type=Path)
parser.add_argument('--corrected',required=True,type=Path)
parser.add_argument('--supplied',required=True,type=Path)
parser.add_argument('--data',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
old=args.baseline.resolve()/'tables'
new=args.corrected.resolve()/'tables'
supplied=args.supplied.resolve()/'tables'
audit=args.output.resolve(); audit.mkdir(parents=True,exist_ok=True)
validation=[]
changed=[]
for source in sorted(supplied.glob('*.csv')):
    a=pd.read_csv(source); b=pd.read_csv(old/source.name); c=pd.read_csv(new/source.name)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-6,atol=1e-10)
    validation.append(source.name)
    try: pd.testing.assert_frame_equal(b,c,check_dtype=False,rtol=1e-10,atol=1e-12)
    except AssertionError: changed.append(source.name)

def comparison(name, keys):
    a=pd.read_csv(old/name);b=pd.read_csv(new/name)
    m=a.merge(b,on=keys,suffixes=('_before','_after'),validate='one_to_one')
    assert len(m)==len(a)==len(b)
    m.to_csv(audit/name,index=False)
    return m

scores=comparison('task_summaries.csv',['subject_id','test','test_label'])
for metric in ['accuracy','brier_score','calibration','resolution','outcome_variance']:
    np.testing.assert_allclose(scores[metric+'_before'],scores[metric+'_after'],rtol=0,atol=1e-12)
scores['reward_change']=scores.rewarded_score_after-scores.rewarded_score_before
scores['converted_calibration_change']=scores.calibration_converted_score_after-scores.calibration_converted_score_before
scores.to_csv(audit/'participant_session_score_changes.csv',index=False)
np.testing.assert_array_less(-1e-12, scores.reward_change+1e-12)
means=comparison('task_metric_means_by_session.csv',['test','test_label'])
icc=comparison('icc2k_all_completers.csv',['metric'])
primary=comparison('supplemental_table_1_strategy_mixed_models.csv',['model','metric','term'])
interactions=comparison('strategy_two_way_interactions_sensitivity.csv',['model','metric','term'])
coefficients=comparison('strategy_model_coefficients.csv',['model','metric','coefficient'])
diagnostics=comparison('strategy_model_diagnostics.csv',['model','metric'])
questionnaires={}
for sample in ['all80','strict51']:
    questionnaires[sample]=comparison(f'questionnaire_task_correlations_{sample}_corrected.csv',['subscale','metric'])
for df in [primary,interactions]:
    nonreward=df.loc[df.metric!='rewarded_score']
    np.testing.assert_allclose(nonreward.p_before,nonreward.p_after,rtol=0,atol=1e-12)
for df in questionnaires.values():
    nonreward=df.loc[df.metric!='rewarded_score']
    np.testing.assert_allclose(nonreward.r_before,nonreward.r_after,rtol=0,atol=1e-12)
    np.testing.assert_allclose(nonreward.p_uncorrected_before,nonreward.p_uncorrected_after,rtol=0,atol=1e-12)

# Independent score check from one record per distinct task trial.
raw=args.data.resolve()
independent=[]
for session,filename in enumerate(['initial_test','retest_1','retest_2']):
    raw_df=pd.read_csv(raw/f'task_data_{filename}.csv')
    trial_df=raw_df.loc[(raw_df.use=='task') & (raw_df.block=='test') & raw_df.subject_id.isin(scores.subject_id)]
    keys=['subject_id','trial_number']
    assert trial_df.groupby(keys).size().eq(4).all()
    assert trial_df.groupby(keys)[['confidence','correct','name']].nunique().eq(1).all().all()
    trial_df=trial_df.drop_duplicates(keys)
    for sid,g in trial_df.groupby('subject_id'):
        assert len(g)==42
        key=g.confidence.to_numpy().astype(int)
        outcome=g.correct.to_numpy().astype(float)
        probabilities=np.array([.5625,.6875,.8125,.9375])
        c=0.
        for k,p in enumerate(probabilities,1):
            selected=(key==k)
            if selected.any(): c+=selected.mean()*(p-outcome[selected].mean())**2
        maximum=np.mean(probabilities[key-1]**2)
        converted=1-c/maximum
        reward=(outcome.mean()+converted)/2
        saved=scores.loc[(scores.subject_id==sid)&(scores.test==session)].iloc[0]
        np.testing.assert_allclose([c,converted,reward],
                                  [saved.calibration_after,saved.calibration_converted_score_after,saved.rewarded_score_after],
                                  rtol=0,atol=1e-12)
        independent.append({'test':session,'subject_id':sid,'n_unique_trials':len(g),'maximum_calibration_error':maximum,
                            'independent_calibration':c,'independent_converted_calibration':converted,'independent_reward':reward})
pd.DataFrame(independent).to_csv(audit/'independent_scoring_check.csv',index=False)

def flips(df,col):
    return df.loc[df[col+'_before']!=df[col+'_after']].to_dict('records')
summary={
    'baseline_csvs_matched':len(validation), 'changed_csvs':changed,
    'participant_sessions':len(scores),'subjects':int(scores.subject_id.nunique()),
    'reward_changed_sessions':int((scores.reward_change.abs()>1e-10).sum()),
    'reward_changed_subjects':int(scores.loc[scores.reward_change.abs()>1e-10,'subject_id'].nunique()),
    'reward_mean_absolute_change':float(scores.reward_change.abs().mean()),
    'reward_max_absolute_change':float(scores.reward_change.abs().max()),
    'reward_min_change':float(scores.reward_change.min()),
    'reward_before_mean':float(scores.rewarded_score_before.mean()),'reward_after_mean':float(scores.rewarded_score_after.mean()),
    'unchanged_other_five_metrics':True,'independent_scoring_checks_passed':len(independent),
    'reward_icc':icc.loc[icc.metric=='rewarded_score'].to_dict('records')[0],
    'reward_primary_tests':primary.loc[primary.metric=='rewarded_score'].to_dict('records'),
    'primary_fdr_flips':flips(primary,'significant_fdr_bh'),
    'primary_fdr_changed_tests':int((abs(primary.p_fdr_bh_before-primary.p_fdr_bh_after)>1e-12).sum()),
    'primary_nonreward_fdr_changed_tests':int(((abs(primary.p_fdr_bh_before-primary.p_fdr_bh_after)>1e-12)&(primary.metric!='rewarded_score')).sum()),
    'interaction_nominal_flips':flips(interactions,'significant_uncorrected'),
    'interaction_fdr_flips':flips(interactions,'significant_fdr_bh'),
    'interaction_min_p_before':float(interactions.p_before.min()),'interaction_min_p_after':float(interactions.p_after.min()),
    'all_strategy_models_converged':bool(diagnostics.converged_after.all()),
    'questionnaires':{},
}
for name,df in questionnaires.items():
    summary['questionnaires'][name]={
        'nominal_before':int(df.significant_uncorrected_before.sum()),'nominal_after':int(df.significant_uncorrected_after.sum()),
        'fdr_before':int(df.significant_fdr_bh_before.sum()),'fdr_after':int(df.significant_fdr_bh_after.sum()),
        'bonferroni_before':int(df.significant_bonferroni_before.sum()),'bonferroni_after':int(df.significant_bonferroni_after.sum()),
        'nominal_flips':flips(df,'significant_uncorrected'),'fdr_flips':flips(df,'significant_fdr_bh'),
        'fdr_changed_tests':int((abs(df.p_fdr_bh_before-df.p_fdr_bh_after)>1e-12).sum()),
        'reward_correlations':df.loc[df.metric=='rewarded_score'].to_dict('records'),
    }
(audit/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['questionnaires','reward_primary_tests']},indent=2))
for name,s in summary['questionnaires'].items():
    print(name, {k:v for k,v in s.items() if k!='reward_correlations'})
