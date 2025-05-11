# Importing libraries
import os
import pickle
import arviz           as az 
import numpy           as np
import pandas          as pd
import bambi           as bmb
from statsmodels.formula.api import mixedlm
from   copy                  import deepcopy
from   util                  import *

template = {          
    'subject':              [],
    'condition':            [],
    'bias':                 [],
    'correct':              [],
    'outcome':              [],
    'transition':           [],
    'outcome_x_transition': [],
    'rewarded_common':      [],
    'rewarded_rare':        [],
    'non_rewarded_common':  [],
    'non_rewarded_rare':    [],
    'stay':                 [],
}

def make_log_reg_data(subject):
    data = get_subject_data(subject)
    d    = deepcopy(template)
    for ii in range(1, len(data)):
        prev = data.iloc[ii-1]
        curr = data.iloc[ii]
        # if the previous or current choices are NaN, or if the previous reward is NaN -- skip this iteration
        if pd.isna(prev['choice1']) or pd.isna(curr['choice1']) or pd.isna(prev['reward']):
            continue
        # Intercept (stay bias)
        d['bias'].append(1.)
        # 0.5 for rewarded trials, -0.5 for non-rewarded trials
        d['outcome'].append(prev['reward'] - 0.5)
        # 0.5 for common trials, -0.5 for rare trials
        d['transition'].append(prev['common'] - 0.5)
        # 0.5 for rewarded common trials, -0.5 for non-rewarded common trials
        # 0.5 for non-rewarded rare trials, -0.5 for rewarded rare trials
        d['outcome_x_transition'].append((prev['reward'] == prev['common']).astype(float) - 0.5)
        # 0.5 for correct choice, -0.5 for incorrect choice [Binarised]
        p_choice1 = [prev['reward_1_1'], prev['reward_1_2']]
        p_choice2 = [prev['reward_2_1'], prev['reward_2_2']]
        if ((p_choice1 > p_choice2) and prev['choice1'] == 1) or \
              ((p_choice2 > p_choice1) and prev['choice1'] == 2):
            d['correct'].append(0.5)
        else:
            d['correct'].append(-0.5)
        #! I am very unsure what is happening here
        # Reward vs. Trial Type
        d['rewarded_common'].append(0.)
        d['rewarded_rare'].append(0.)
        d['non_rewarded_common'].append(0.)
        d['non_rewarded_rare'].append(0.)
        if prev['reward'] == 1 and prev['common'] == 1:
            d['rewarded_common'][-1] = 0.5
        elif prev['reward'] == 1 and prev['common'] == 0:
            d['rewarded_rare'][-1] = 0.5
        elif prev['reward'] == 0 and prev['common'] == 1:
            d['non_rewarded_common'][-1] = 0.5
        elif prev['reward'] == 0 and prev['common'] == 0:
            d['non_rewarded_rare'][-1] = 0.5
        # Dependent variable
        d['stay'].append((curr['choice1'] == prev['choice1']).astype(float))
    d['subject']   = [subject] * len(d['bias'])
    d['condition'] = ([data.iloc[0]['Condition']] == 'Story') * len(d['bias'])
    return pd.DataFrame(d)

def make_data():
    arr      = []
    subjects = get_all_subjects()
    for subject in subjects:
        arr.append(make_log_reg_data(subject))
    df       = pd.concat(arr)
    df.to_csv(os.path.join(DATA_DIRECTORY, 'log_reg_data.csv'), index=False)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'log_reg_data.csv'))

    ## Hierarchical Bayesian Model
    # model = bmb.Model(
    #     'stay ~ outcome + transition + outcome_x_transition + (1|condition/subject)', 
    #     df, 
    #     family='bernoulli'
    # )
    ## Fit the model
    # results = model.fit(cores=1, target_accept=0.95, chains=4, tune=2000)

    ## Hierarchical MLE Model
    df['condition'] = df['condition'].astype('category')
    model = mixedlm(
        "stay ~ 1 + C(condition) + correct + outcome + transition + outcome_x_transition",
        df,
        groups=df["subject"],
        re_formula="~ correct + outcome + transition + outcome_x_transition"
    )
    result = model.fit(reml=False)
    print(result.summary())
    print(f"AIC: {result.aic}")
    # Saving the results
    subject_effects = result.random_effects
    d = {
        'subject': [],
        'outcome': [],
        'outcome_x_transition': [],
    }

    for subject_id, values in result.random_effects.items():
        d['subject'].append(subject_id)
        d['outcome'].append(values[2])
        d['outcome_x_transition'].append(values[4])
    df = pd.DataFrame(d)
    df.to_csv(os.path.join(RESULTS_DIRECTORY, 'log_reg_subject_effects.csv'), index=False)