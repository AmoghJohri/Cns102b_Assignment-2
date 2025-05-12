functions {
    real hybrid(int num_trials, array[] int subject, array[] int action1, array[] int s2,
                array[] int action2, array[] int reward, real alpha1, real alpha2,
                real lmbd, real beta1, real beta2, real w0, real k, real p) {

        real log_lik = 0;
        array[2] real q;
        array[2, 2] real v;
        int trial_counter = 0;


        // Initialize q, v to 0
        for (i in 1:2)
            q[i] = 0;
        for (i in 1:2)
            for (j in 1:2)
                v[i, j] = 0;

        for (t in 1:num_trials) {
            
            // 🟢 reset state variables when subject changes
            if (t == 1 || subject[t] != subject[t - 1]) {
                for (i in 1:2)
                    q[i] = 0;
                for (i in 1:2)
                    for (j in 1:2)
                        v[i, j] = 0;
                trial_counter = 0;
            }

            real w_t = w0 + k * (trial_counter - 1);
            w_t      = fmin(fmax(w_t, 0), 1);  // constrain w_t to [0, 1]

            // 🟢 model-based + model-free value combination
            real x1 = w_t * 0.4 * (max(v[2]) - max(v[1])) +
                      (1 - w_t) * (q[2] - q[1]);

            // 🟢 perseveration
            if (t > 1 && subject[t] == subject[t - 1]) {
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }

            x1 *= beta1;

            // 🟢 first stage choice likelihood
            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // 🟢 second stage choice likelihood
            real x2 = beta2 * (v[s2[t], 2] - v[s2[t], 1]);
            if (action2[t] == 2)
                log_lik += log_inv_logit(x2);
            else
                log_lik += log1m_inv_logit(x2);

            // 🟢 learning updates
            q[action1[t]] += alpha1 * (v[s2[t], action2[t]] - q[action1[t]]) +
                             alpha1 * lmbd * (reward[t] - v[s2[t], action2[t]]);
            v[s2[t], action2[t]] += alpha2 * (reward[t] - v[s2[t], action2[t]]);
        }
        return log_lik;
    }
}
data {
    int<lower=1> num_trials;
    array[num_trials] int<lower=1> subject;            // 🟢 subject ID for each trial
    array[num_trials] int<lower=1, upper=2> action1;
    array[num_trials] int<lower=1, upper=2> action2;
    array[num_trials] int<lower=1, upper=2> s2;
    array[num_trials] int<lower=0, upper=1> reward;
}
parameters {
    real<lower=0, upper=1> alpha1;
    real<lower=0, upper=1> alpha2;
    real<lower=0, upper=1> lmbd;
    real<lower=0, upper=20> beta1;
    real<lower=0, upper=20> beta2;
    real<lower=0,   upper=1>  w0;
    real<lower=-0.1,  upper=0.1>  k;
    real<lower=-20, upper=20> p;
}
model {
    target += hybrid(num_trials, subject, action1, s2, action2, reward,
                     alpha1, alpha2, lmbd, beta1, beta2, w0, k, p);
}
