functions {
    real dual_wsls_model(int num_trials, array[] int subject, array[] int action1,
                         array[] int reward, array[] int doRareTrans,
                         real beta_MB, real beta_MF,
                         real wWS_MB, real wLS_MB,
                         real wWS_MF, real wLS_MF,
                         real w) {

        real log_lik = 0;

        for (t in 1:num_trials) {
            // skip first trial
            if (t == 1 || subject[t] != subject[t - 1])
                continue;

            real bias_MB = 0;
            real bias_MF = 0;

            // 🔵 Model-Based WSLS
            if ((doRareTrans[t - 1] == 0 && reward[t - 1] == 1) ||
                (doRareTrans[t - 1] == 1 && reward[t - 1] == 0)) {
                // stay condition
                if (action1[t - 1] == 2)
                    bias_MB = wWS_MB;
                else
                    bias_MB = -wWS_MB;
            } else {
                // switch condition
                if (action1[t - 1] == 2)
                    bias_MB = -wLS_MB;
                else
                    bias_MB = wLS_MB;
            }

            // 🔴 Model-Free WSLS
            if (reward[t - 1] == 1) {
                if (action1[t - 1] == 2)
                    bias_MF = wWS_MF;
                else
                    bias_MF = -wWS_MF;
            } else {
                if (action1[t - 1] == 2)
                    bias_MF = -wLS_MF;
                else
                    bias_MF = wLS_MF;
            }

            // scale both by separate betas
            bias_MB *= beta_MB;
            bias_MF *= beta_MF;

            // ✅ Combine in value space
            real combined_bias = w * bias_MB + (1 - w) * bias_MF;

            // ✅ Compute choice likelihood
            if (action1[t] == 2)
                log_lik += log_inv_logit(combined_bias);
            else
                log_lik += log1m_inv_logit(combined_bias);
        }

        return log_lik;
    }
}
data {
    int<lower=1> num_trials;
    array[num_trials] int<lower=1> subject;
    array[num_trials] int<lower=1, upper=2> action1;
    array[num_trials] int<lower=0, upper=1> reward;
    array[num_trials] int<lower=0, upper=1> doRareTrans;
}
parameters {
    real<lower=0, upper=20> beta_MB;
    real<lower=0, upper=20> beta_MF;
    real<lower=0, upper=5>  wWS_MB;
    real<lower=0, upper=5>  wLS_MB;
    real<lower=0, upper=5>  wWS_MF;
    real<lower=0, upper=5>  wLS_MF;
    real<lower=0, upper=1>  w;
}
model {
    target += dual_wsls_model(num_trials, subject, action1, reward, doRareTrans,
                              beta_MB, beta_MF,
                              wWS_MB, wLS_MB,
                              wWS_MF, wLS_MF,
                              w);
}
