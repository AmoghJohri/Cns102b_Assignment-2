# Importing libraries
from util import *
from scipy.stats import pearsonr

expr1 = 'Please rate on a 0 to 4 scale how well you understood the game.'
expr2 = 'Please rate on a 0 to 4 scale how effortful playing the game was.'
expr3 = 'Please rate on a 0 to 4 scale how complex you thought the game was.'

def get_subject_ratings(subject):
    # No questionnaire data for subject 48
    if subject == 48:
        return None
    # Load questionnaire data
    filepath = get_subject_questionnaire(subject)
    with open(filepath, 'r') as file:
        lines = file.readlines()
    # Remove all trailing '\n'
    lines = [line.rstrip('\n') for line in lines]
    # Remove all blanks
    lines = [line for line in lines if line]
    d = {
        'understanding': None,
        'effort': None,
        'complexity': None
    }
    for ii in range(len(lines)):
        if expr1 == lines[ii]:
            d['understanding'] = int(lines[ii+1][0])
        elif expr2 == lines[ii]:
            d['effort'] = int(lines[ii+1][0])
        elif expr3 == lines[ii]:
            d['complexity'] = int(lines[ii+1][0])
    return d

if __name__ == "__main__":
    subjects = get_all_subjects()
    d = {
        'Subject':       [],
        'Understanding': [],
        'Effort':        [],
        'Complexity':    [],
        'Reward':        []
    }
    for subject in subjects:
        # Skip subject 48
        if subject == 48:
            continue
        output = get_subject_ratings(subject)
        reward = round(get_subject_data(subject)['reward'].mean(), 4)
        # Storing the results
        d['Subject'].append(subject)
        d['Understanding'].append(output['understanding'])
        d['Effort'].append(output['effort'])
        d['Complexity'].append(output['complexity'])
        d['Reward'].append(reward)
    # Convert to dataframe
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(DATA_DIRECTORY, 'questionnaire_data.csv'), index=False)
    for each in ['Understanding', 'Effort', 'Complexity']:
        r, p = pearsonr(df[each], df['Reward'])
        print(f"{each} vs Reward: r = {r:.4f}, p = {p:.4f}")
