functions {
    real reward_as_cue_hybrid(int num_trials, 
                              array[] int subject, 
                              array[] int action1, 
                              array[] int s2, 
                              array[] int reward,
                              real alpha1, real alpha2, real beta1, real p, real w) {
        real log_lik = 0;

        // Initialize first-stage cue-based Q-values (4 contexts x 2 actions)
        array[4, 2] real Q1;
        for (i in 1:4)
            for (j in 1:2)
                Q1[i, j] = 0;

        // Initialize second-stage values
        array[2] real V2;
        for (j in 1:2)
            V2[j] = 0;

        // Known transition probabilities
        array[2, 2] real P_trans;
        P_trans[1, 1] = 0.7;  P_trans[1, 2] = 0.3;  // action 1
        P_trans[2, 1] = 0.3;  P_trans[2, 2] = 0.7;  // action 2

        int context;

        for (t in 1:num_trials) {
            // reset between subjects
            if (t == 1 || subject[t] != subject[t - 1]) {
                for (i in 1:4)
                    for (j in 1:2)
                        Q1[i, j] = 0;
                for (j in 1:2)
                    V2[j] = 0;
                context = 1;
            }

            // ------------------------
            // Model-free value
            // ------------------------
            array[2] real Qcue;
            Qcue[1] = Q1[context, 1];
            Qcue[2] = Q1[context, 2];

            // ------------------------
            // Model-based value
            // ------------------------
            array[2] real Qmb;
            for (a in 1:2) {
                Qmb[a] = P_trans[a, 1] * V2[1] + P_trans[a, 2] * V2[2];
            }

            // ------------------------
            // Hybrid value
            // ------------------------
            array[2] real Q;
            for (a in 1:2)
                Q[a] = (1 - w) * Qcue[a] + w * Qmb[a];

            // ------------------------
            // Perseveration
            // ------------------------
            real x1 = beta1 * (Q[2] - Q[1]);

            if (t > 1 && subject[t] == subject[t - 1]) {
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }

            // ------------------------
            // Choice likelihood
            // ------------------------
            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // ------------------------
            // First-stage cue-based update
            // ------------------------
            real rpe1 = reward[t] - Q1[context, action1[t]];
            Q1[context, action1[t]] += alpha1 * rpe1;

            // ------------------------
            // Second-stage value update
            // ------------------------
            real rpe2 = reward[t] - V2[s2[t]];
            V2[s2[t]] += alpha2 * rpe2;

            // ------------------------
            // Update context for next trial
            // ------------------------
            if (reward[t] == 1 && s2[t] == 1)
                context = 1;
            else if (reward[t] == 1 && s2[t] == 2)
                context = 2;
            else if (reward[t] == 0 && s2[t] == 1)
                context = 3;
            else
                context = 4;
        }

        return log_lik;
    }
}

data {
    int<lower=1> num_trials;
    array[num_trials] int<lower=1> subject;
    array[num_trials] int<lower=1, upper=2> action1;
    array[num_trials] int<lower=1, upper=2> s2;
    array[num_trials] int<lower=0, upper=1> reward;
}

parameters {
    real<lower=0, upper=1> alpha1;        // learning rate for Q1
    real<lower=0, upper=1> alpha2;        // learning rate for V2
    real<lower=0, upper=20> beta1;        // inverse temperature
    real<lower=-20, upper=20> p;          // perseveration bias
    real<lower=0, upper=1> w;             // model-based weight
}

model {
    target += reward_as_cue_hybrid(num_trials, subject, action1, s2, reward,
                                   alpha1, alpha2, beta1, p, w);
}
