# Importing libraries
import os
import pickle
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from   util   import *

if __name__ == "__main__":
    questionnaire     = pd.read_csv(os.path.join(DATA_DIRECTORY, "questionnaire_data.csv"))
    result            = pd.read_csv(os.path.join(RESULTS_DIRECTORY, 'log_reg_subject_effects.csv'))
    story_subjects    = get_story_subjects()
    abstract_subjects = get_abstract_subjects()
    d = {
        'Subject':              [],
        'Understanding':        [],
        'Effort':               [],
        'Complexity':           [],
        'Reward':               [],
        'outcome':              [],
        'outcome_x_transition': [],
        'condition':            [],
    }
    for subject in result['Subject'].unique():
        if subject == 48: continue
        d['Subject'].append(subject)
        d['Understanding'].append(questionnaire[questionnaire['Subject'] == subject]['Understanding'].values[0])
        d['Effort'].append(questionnaire[questionnaire['Subject'] == subject]['Effort'].values[0])
        d['Complexity'].append(questionnaire[questionnaire['Subject'] == subject]['Complexity'].values[0])
        d['Reward'].append(questionnaire[questionnaire['Subject'] == subject]['Reward'].values[0])
        d['outcome'].append(result[result['Subject'] == subject]['outcome'].values[0])
        d['outcome_x_transition'].append(result[result['Subject'] == subject]['outcome_x_transition'].values[0])
        if subject in story_subjects:
            d['condition'].append(0.)
        elif subject in abstract_subjects:
            d['condition'].append(1.)
        else:
            raise ValueError(f"Unknown subject {subject}")
    df = pd.DataFrame(d)
    print(df)

    variables = ['Understanding', 'Effort', 'Complexity', 'Reward', 'outcome', 'outcome_x_transition']
    for variable in variables:
        print(f"Ttest for {variable}")
        print(f"Story: {round(df[df['condition'] == 0][variable].mean(), 3)}")
        print(f"Abstract: {round(df[df['condition'] == 1][variable].mean(), 3)}")
        test = ttest_ind(df[df['condition'] == 0][variable], df[df['condition'] == 1][variable])
        print(f"t-statistic: {round(test.statistic, 3)}")
        print(f"p-value: {round(test.pvalue, 3)}")
        print('\n')