functions {
    real triple_hybrid(int num_trials, 
                       array[] int subject, 
                       array[] int action1, 
                       array[] int s2, 
                       array[] int reward,
                       real alpha2, real beta1, real p, real w, real w_rac) {
        real log_lik = 0;
        real alpha1  = 0.05;

        // Initialize model-free Q-values (2 actions)
        array[2] real Qmf;
        for (a in 1:2)
            Qmf[a] = 0;

        // Initialize reward-as-cue Q-values (4 contexts x 2 actions)
        array[4, 2] real Qcue;
        for (c in 1:4)
            for (a in 1:2)
                Qcue[c, a] = 0;

        // Initialize second-stage state values
        array[2] real V2;
        for (s in 1:2)
            V2[s] = 0;

        // Transition matrix
        array[2, 2] real P_trans;
        P_trans[1, 1] = 0.7;  P_trans[1, 2] = 0.3;  // action 1
        P_trans[2, 1] = 0.3;  P_trans[2, 2] = 0.7;  // action 2

        int context;

        for (t in 1:num_trials) {
            // Reset between subjects
            if (t == 1 || subject[t] != subject[t - 1]) {
                for (a in 1:2)
                    Qmf[a] = 0;
                for (c in 1:4)
                    for (a in 1:2)
                        Qcue[c, a] = 0;
                for (s in 1:2)
                    V2[s] = 0;
                context = 1;
            }

            // --------------------
            // Model-free Q values
            // --------------------
            array[2] real Q_MF;
            Q_MF[1] = Qmf[1];
            Q_MF[2] = Qmf[2];

            // --------------------
            // Model-based Q values
            // --------------------
            array[2] real Q_MB;
            for (a in 1:2)
                Q_MB[a] = P_trans[a, 1] * V2[1] + P_trans[a, 2] * V2[2];

            // --------------------
            // Reward-as-cue Q values
            // --------------------
            array[2] real Q_CUE;
            Q_CUE[1] = Qcue[context, 1];
            Q_CUE[2] = Qcue[context, 2];

            // --------------------
            // Combined Q values
            // --------------------
            array[2] real Q;
            for (a in 1:2)
                Q[a] = (1 - w_rac) * (w * Q_MB[a] + 
                       (1 - w) * (Q_MF[a])) + w_rac * Q_CUE[a];

            // --------------------
            // Perseveration
            // --------------------
            real x1 = beta1 * (Q[2] - Q[1]);
            if (t > 1 && subject[t] == subject[t - 1]) {
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }

            // --------------------
            // Choice likelihood
            // --------------------
            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // --------------------
            // Updates
            // --------------------
            real rpe_mf = reward[t] - Qmf[action1[t]];
            Qmf[action1[t]] += alpha1 * rpe_mf;

            real rpe_cue = reward[t] - Qcue[context, action1[t]];
            Qcue[context, action1[t]] += alpha1 * rpe_cue;

            real rpe_v2 = reward[t] - V2[s2[t]];
            V2[s2[t]] += alpha2 * rpe_v2;

            // --------------------
            // Update context
            // --------------------
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
    real<lower=0, upper=1> alpha2;        // learning rate for second stage
    real<lower=0, upper=20> beta1;        // inverse temperature
    real<lower=-20, upper=20> p;          // perseveration bias
    real<lower=0, upper=1> w;             // model-based weight
    real<lower=0, upper=1> w_rac;
}

model {
    target += triple_hybrid(num_trials, subject, action1, s2, reward,
                            alpha2, beta1, p, w, w_rac);
}
