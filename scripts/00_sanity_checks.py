# Importing libraries
from util import *
from scipy.stats import ttest_ind
import time

# simulating a single trial
def simulate_trial(trial, choice1, choice2):
    p_reward = trial['reward_{}_{}'.format(choice1, choice2)]
    return np.random.choice([0, 1], p=[1-p_reward, p_reward])

def run_random_agent(data):
    # Seed
    np.random.seed(42)
    # Bookkeeping variables
    choice1_set = [1, 2]        # possible choice - 1
    choice2_set = [1, 2]        # possible choice - 2
    n_trials    = data.shape[0] # number of trials
    rewards     = []            # store the trial-wise rewards
    # randomly select choices
    choice1s    = np.random.choice(choice1_set, n_trials)
    choice2s    = np.random.choice(choice2_set, n_trials)
    # simulate trials
    #! not considering 'slow' trials
    for index, row in data.iterrows():
        reward = simulate_trial(row, choice1s[index], choice2s[index])
        rewards.append(reward)
    return np.asarray(rewards)

if __name__ == "__main__":
    subjects       = get_all_subjects()
    d = {
        'Subject':             [],
        'Reward':              [],
        'Random-Agent-Reward': [],
        'T-test':              [],
        'P-value':             []
    }
    for subject in subjects:
        start = time.time()
        print(f"Starting subject {subject}")
        # Load data
        data    = get_subject_data(subject)
        rewards = data['reward'].values
        # Run random agent
        random_rewards = run_random_agent(data)
        # Calculate t-test
        t_stat, p_value = ttest_ind(rewards, random_rewards)
        # Store results
        d['Subject'].append(subject)
        d['Reward'].append(round(rewards.mean(), 4))
        d['Random-Agent-Reward'].append(round(random_rewards.mean(), 4))
        d['T-test'].append(round(t_stat, 4))
        d['P-value'].append(round(p_value, 4))
        print(f"Subject {subject} took {time.time() - start:.2f} seconds")
    # Convert to DataFrame
    df = pd.DataFrame(d)
    # Save to CSV
    df.to_csv(os.path.join(RESULTS_DIRECTORY, "subjects_vs_random-agents.csv"), index=False)

