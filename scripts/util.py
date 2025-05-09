# Loading the libraries
import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# Bookkeeping variables
DATA_DIRECTORY          = "../data"
PLOTS_DIRECTORY         = "../plots"
RESULTS_DIRECTORY       = "../results"
DATA_CSV                = os.path.join(DATA_DIRECTORY, "all_subjects.csv")
QUESTIONNAIRE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'questionnaire')

# Data columns
COLUMNS = ["trial", "common",
            "reward_1_1", "reward_1_2", "reward_2_1", "reward_2_2", 
            "isymbol_lft", "isymbol_rgt", 
            "fsymbol_lft", "fsymbol_rgt",
            "rt1", "rt2",
            "choice1", "choice2",
            "final_state", "reward", "slow", 
            "Subject", "Condition"]

### Functions to save the data
def save_plot(fig, name):
    fig.savefig(os.path.join(PLOTS_DIRECTORY, name), dpi=300)
    print(f"Saved figure {name} to {PLOTS_DIRECTORY}")

### Functions to load the data
# loading the data
def get_data(usecols=None):
    return pd.read_csv(DATA_CSV, usecols=usecols)

# loading data for story condition
def get_story_data(usecols=None):
    data = get_data(usecols=usecols)
    data = data[data.Condition == "story"]
    return data

# loading data for abstract condition
def get_abstract_data(usecols=None):
    data = get_data(usecols=usecols)
    data = data[data.Condition == "abstract"]
    return data

### Functions to get subjects
def get_all_subjects():
    return sorted(get_data(usecols=['Subject'])['Subject'].unique().astype(int).tolist())

def get_story_subjects():
    return sorted(get_story_data(usecols=['Subject', 'Condition'])['Subject'].unique().astype(int).tolist())

def get_abstract_subjects():
    return sorted(get_abstract_data(usecols=['Subject', 'Condition'])['Subject'].unique().astype(int).tolist())

def get_subject_data(subject, usecols=None):
    data = get_data(usecols=usecols)
    data = data[data.Subject == subject].reset_index(drop=True)
    return data

def get_subject_questionnaire(subject):
    filepath = os.path.join(QUESTIONNAIRE_DIRECTORY, str(subject).zfill(3) + "_story.txt")
    if os.path.exists(filepath): return filepath
    else: return filepath.replace("story", "abstract")

if __name__ == "__main__":
    get_subject_questionnaire(1)