functions {
    real hybrid(int num_trials, array[] int subject, array[] int action1, array[] int s2,
                array[] int action2, array[] int reward, array[] int doRareTrans,
                real alpha2, real eta_T, real beta1,real p) {

        real log_lik = 0;
        array[2, 2] real v;
        array[2, 2] real T;

        // Initialize
        for (i in 1:2) {
            for (j in 1:2) {
                v[i, j] = 0;
                T[i, j] = 0.5;     // unbiased start (uniform: equal chance)
            }
        }

        for (t in 1:num_trials) {
            // 🟢 reset at subject boundary
            if (t == 1 || subject[t] != subject[t - 1]) {
                for (i in 1:2) {
                    for (j in 1:2) {
                        v[i, j] = 0;
                        T[i, j] = 0.5;
                    }
                }
            }

            // 🟢 model-based action values using current T estimates
            real v1 = fmax(v[1,1], v[1,2]);
            real v2 = fmax(v[2,1], v[2,2]);

            real Q1 = T[1,1] * v1 + T[1,2] * v2;
            real Q2 = T[2,1] * v1 + T[2,2] * v2;

            real x1 = Q2 - Q1;

            // 🟢 perseveration
            if (t > 1 && subject[t] == subject[t - 1]) {
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }

            x1 *= beta1;

            // 🟢 first-stage choice likelihood
            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // 🟢 second-stage learning
            v[s2[t], action2[t]] += alpha2 * (reward[t] - v[s2[t], action2[t]]);

            // 🟢 update transition estimates (simple RW-style learning)
            for (j in 1:2) {
                if (j == s2[t])
                    T[action1[t], j] += eta_T * (1 - T[action1[t], j]);
                else
                    T[action1[t], j] += eta_T * (0 - T[action1[t], j]);
            }
        }
        return log_lik;
    }
}
data {
    int<lower=1> num_trials;
    array[num_trials] int<lower=1> subject;
    array[num_trials] int<lower=1, upper=2> action1;
    array[num_trials] int<lower=1, upper=2> action2;
    array[num_trials] int<lower=1, upper=2> s2;
    array[num_trials] int<lower=0, upper=1> reward;
    array[num_trials] int<lower=0, upper=1> doRareTrans;
}
parameters {
    real<lower=0, upper=1> alpha2;
    real<lower=0, upper=1> eta_T;          // 🔥 transition learning rate
    real<lower=0, upper=20> beta1;
    real<lower=-20, upper=20> p;
}
model {
    target += hybrid(num_trials, subject, action1, s2, action2, reward, doRareTrans,
                     alpha2, eta_T, beta1, p);
}
