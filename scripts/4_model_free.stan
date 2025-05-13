functions {
    real hybrid(int num_trials, array[] int subject, array[] int action1, array[] int s2,
                array[] int action2, array[] int reward, array[] int doRareTrans,
                real alpha1, real alpha2, real lmbd, real beta1, real p) {

        real log_lik = 0;
        array[2] real q;
        array[2, 2] real v;

        // Initialize
        for (i in 1:2)
            q[i] = 0;
        for (i in 1:2)
            for (j in 1:2)
                v[i, j] = 0;

        for (t in 1:num_trials) {
            // 🟢 reset at subject change
            if (t == 1 || subject[t] != subject[t - 1]) {
                for (i in 1:2)
                    q[i] = 0;
                for (i in 1:2)
                    for (j in 1:2)
                        v[i, j] = 0;
            }

            // 🟢 model-free value only
            real x1 = q[2] - q[1];

            // 🟢 perseveration
            if (t > 1 && subject[t] == subject[t - 1]) {
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }

            x1 *= beta1;

            // 🟢 first-stage choice
            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

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
    array[num_trials] int<lower=1> subject;
    array[num_trials] int<lower=1, upper=2> action1;
    array[num_trials] int<lower=1, upper=2> action2;
    array[num_trials] int<lower=1, upper=2> s2;
    array[num_trials] int<lower=0, upper=1> reward;
    array[num_trials] int<lower=0, upper=1> doRareTrans;
}
parameters {
    real<lower=0, upper=1> alpha1;
    real<lower=0, upper=1> alpha2;
    real<lower=0, upper=1> lmbd;
    real<lower=0, upper=20> beta1;
    real<lower=-20, upper=20> p;
}
model {
    target += hybrid(num_trials, subject, action1, s2, action2, reward, doRareTrans,
                     alpha1, alpha2, lmbd, beta1, p);
}
