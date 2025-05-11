functions {
    real hybrid(int num_trials, array[] int action1, array[] int s2, array[] int action2,
        array[] int reward, real alpha1, real alpha2, real lmbd, real beta1, 
        real beta2, real w0, real k, real p) {

        real log_lik;
        array[2] real q;
        array[2, 2] real v;

        // Initializing values
        for (i in 1:2)
            q[i] = 0;
        for (i in 1:2)
            for (j in 1:2)
                v[i, j] = 0;

        log_lik = 0;

        for (t in 1:num_trials) {
            real x1;
            real x2;
            real w_t = w0 + k * (t - 1);
            w_t      = fmin(fmax(w_t, 0), 1);  // constrain w_t to [0, 1]
            x1 = // Model-based value
                w_t*0.4*(max(v[2]) - max(v[1])) +
                // Model-free value
                (1 - w_t)*(q[2] - q[1]);
            // Perseveration
            if (t > 1) {
                if (action1[t - 1] == 2)
                    x1 += p;
                else
                    x1 -= p;
            }
            // Exploration
            x1 *= beta1;
            // First stage choice
            if (action1[t] == 2)
                log_lik += log_inv_logit(x1);
            else
                log_lik += log1m_inv_logit(x1);

            // Second stage choice
            x2 = beta2*(v[s2[t], 2] - v[s2[t], 1]);
            if (action2[t] == 2)
                log_lik += log_inv_logit(x2);
            else
                log_lik += log1m_inv_logit(x2);

            // Learning
            q[action1[t]] += alpha1*(v[s2[t], action2[t]] - q[action1[t]]) +
                alpha1*lmbd*(reward[t] - v[s2[t], action2[t]]);
            v[s2[t], action2[t]] += alpha2*(reward[t] - v[s2[t], action2[t]]);
        }
        return log_lik;
    }
}
data {
    // Number of trials
    int<lower=1> num_trials;
    array[num_trials] int<lower=1, upper=2> action1; // First stage actions
    array[num_trials] int<lower=1, upper=2> action2; // Second stage actions
    array[num_trials] int<lower=1, upper=2> s2;      // Second stage states
    array[num_trials] int<lower=0, upper=1> reward;  // Rewards
}
parameters {
    real<lower=0,   upper=1>   alpha1;
    real<lower=0,   upper=1>   alpha2;
    real<lower=0,   upper=1>   lmbd;
    real<lower=0,   upper=20>  beta1;
    real<lower=0,   upper=20>  beta2;
    real<lower=0,   upper=1>   w0;
    real<lower=-1,  upper=1>   k;
    real<lower=-20, upper=20>  p;
}
model {
    target += hybrid(num_trials, action1, s2, action2, reward, alpha1, alpha2, lmbd, beta1,
        beta2, w0, k, p);
}

