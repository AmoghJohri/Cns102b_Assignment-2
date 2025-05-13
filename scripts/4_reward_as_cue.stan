functions {
    real reward_as_cue_firststage(int num_trials, 
                                  array[] int subject, 
                                  array[] int action1, 
                                  array[] int s2, 
                                  array[] int reward,
                                  real alpha1, real beta1, real p) {
        real log_lik = 0;

        // Initialize 1st-stage Q-values (4 contexts x 2 actions)
        array[4, 2] real Q1;
        for (i in 1:4)
            for (j in 1:2)
                Q1[i, j] = 0;

        // Context at trial t depends on previous trial
        int context;

        for (t in 1:num_trials) {
            // reset between subjects
            if (t == 1 || subject[t] != subject[t - 1]) {
                for (i in 1:4)
                    for (j in 1:2)
                        Q1[i, j] = 0;

                // On first trial, arbitrary context
                context = 1;
            }

            // Softmax choice at 1st stage
            real x1 = (Q1[context, 2] - Q1[context, 1]);

            // 🟢 perseveration
            if (t > 1 && subject[t] == subject[t - 1]) {
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }
            x1 *= beta1;

            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // 1st stage Q-learning update
            real rpe = reward[t] - Q1[context, action1[t]];
            Q1[context, action1[t]] += alpha1 * rpe;

            // update context for next trial
            if (reward[t] == 1 && s2[t] == 1)
                context = 1;
            else if (reward[t] == 1 && s2[t] == 2)
                context = 2;
            else if (reward[t] == 0 && s2[t] == 1)
                context = 3;
            else  // reward == 0 && s2 == 2
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
    real<lower=0, upper=1> alpha1;
    real<lower=0, upper=20> beta1;
    real<lower=-20, upper=20> p;
}

model {
    target += reward_as_cue_firststage(num_trials, subject, action1, s2, reward,
                                       alpha1, beta1, p);
}
