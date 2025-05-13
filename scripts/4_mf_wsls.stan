functions {
    real wsls_model(int num_trials, array[] int subject, array[] int action1,
                    array[] int action2, array[] int reward, array[] int doRareTrans,
                    real beta1, real wWS, real wLS) {

        real log_lik = 0;

        for (t in 1:num_trials) {
            // Skip first trial (no previous trial)
            if (t == 1 || subject[t] != subject[t - 1])
                continue;

            real bias = 0;

            // Determine WSLS bias
            if (action1[t - 1] == 2) {
                if (reward[t - 1] == 1)
                    bias = wWS;
                else
                    bias = -wLS;
            }
            if (action1[t - 1] == 1) {
                if (reward[t - 1] == 1)
                    bias = -wWS;
                else
                    bias = wLS;
            }

            // Apply softmax
            bias *= beta1;

            if (action1[t] == 2)
                log_lik += log_inv_logit(bias);
            else
                log_lik += log1m_inv_logit(bias);
        }
        return log_lik;
    }
}
data {
    int<lower=1> num_trials;
    array[num_trials] int<lower=1> subject;
    array[num_trials] int<lower=1, upper=2> action1;
    array[num_trials] int<lower=1, upper=2> action2;
    array[num_trials] int<lower=0, upper=1> reward;
    array[num_trials] int<lower=0, upper=1> doRareTrans;
}
parameters {
    real<lower=0, upper=20> beta1;
    real<lower=0, upper=5>  wWS;
    real<lower=0, upper=5>  wLS;
}
model {
    target += wsls_model(num_trials, subject, action1, action2, reward, doRareTrans,
                            beta1, wWS, wLS);
}
