functions {
    real hybrid_wsls(int num_trials, array[] int subject, array[] int action1, array[] int s2,
                     array[] int action2, array[] int reward, array[] int doRareTrans,
                     real alpha1, real alpha2, real lmbd, real beta_heur,
                     real beta_MB_WSLS, real wWS_MB,
                     real w, real p) {

        real log_lik = 0;
        array[2] real q;
        array[2, 2] real v;

        // initialize values
        for (i in 1:2)
            q[i] = 0;
        for (i in 1:2)
            for (j in 1:2)
                v[i, j] = 0;

        for (t in 1:num_trials) {
            // reset between subjects
            if (t == 1 || subject[t] != subject[t - 1]) {
                for (i in 1:2)
                    q[i] = 0;
                for (i in 1:2)
                    for (j in 1:2)
                        v[i, j] = 0;
            }

            // --- RL value combination ---
            real mb_value = 0.4 * (max(v[2]) - max(v[1]));
            real mf_value = q[2] - q[1];
            real x1       = (w * mb_value + (1 - w) * mf_value);

            // --- Dual WSLS biases ---
            real bias_MB = 0;

            if (t > 1 && subject[t] == subject[t - 1]) {
                // MB WSLS
                if ((doRareTrans[t - 1] == 0 && reward[t - 1] == 1) ||
                    (doRareTrans[t - 1] == 1 && reward[t - 1] == 0)) {
                    if (action1[t - 1] == 2)
                        bias_MB = wWS_MB;
                    else
                        bias_MB = -wWS_MB;
                }

                bias_MB *= beta_MB_WSLS;

                x1       = (1 - beta_heur) * x1 + beta_heur * bias_MB;

                // perseveration
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }

            // --- choice likelihood ---
            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // --- learning updates ---
            q[action1[t]] += alpha1 * (v[s2[t], action2[t]] - q[action1[t]]) +
                             alpha1 * lmbd * (reward[t] - v[s2[t], action2[t]]);
            v[s2[t], action2[t]] += alpha2 * (reward[t] - v[s2[t], action2[t]]);
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
    real<lower=0, upper=1>    alpha1;
    real<lower=0, upper=2>    alpha2;
    real<lower=0, upper=1>    lmbd;
    real<lower=0, upper=20>   beta_MB_WSLS;
    real<lower=0, upper=10>   wWS_MB;
    real<lower=0, upper=1>    w;
    real<lower=0, upper=1>    beta_heur;
    real<lower=-20, upper=20> p;
}
model {
    target += hybrid_wsls(num_trials, subject, action1, s2, action2, reward, doRareTrans,
                          alpha1, alpha2, lmbd, beta_heur,
                          beta_MB_WSLS, wWS_MB,
                          w, p);
}
