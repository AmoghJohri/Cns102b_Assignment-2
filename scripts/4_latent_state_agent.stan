functions {
    real latent_state_agent(int num_trials, 
                            array[] int subject, 
                            array[] int action1, 
                            array[] int s2, 
                            array[] int reward,
                            real omega, real epsilon, real p) {

        real log_lik = 0;
        real Pgood = 0.625;
        real Pbad  = 0.375;

        for (n in 1:num_trials) {
            if (n == 1 || subject[n] != subject[n - 1]) {
                continue; // no info on first trial of subject
            }

            // ---- Bayesian belief update ----
            int s2_prev = s2[n - 1];
            int r_prev  = reward[n - 1];

            real like1 = (s2_prev == 1) ? 
                (r_prev == 1 ? Pgood : (1 - Pgood)) : 
                (r_prev == 1 ? Pbad  : (1 - Pbad));

            real like2 = (s2_prev == 1) ? 
                (r_prev == 1 ? Pbad  : (1 - Pbad)) : 
                (r_prev == 1 ? Pgood : (1 - Pgood));

            real p_state1 = like1 / (like1 + like2);
            real p_state2 = 1 - p_state1;

            // ---- reversal belief update ----
            real new_p1 = (1 - omega) * p_state1 + omega * p_state2;
            real new_p2 = (1 - omega) * p_state2 + omega * p_state1;

            // ---- expected values for actions ----
            real val_A1 = new_p1 * Pgood + new_p2 * Pbad; // action 1 → state a
            real val_A2 = new_p1 * Pbad  + new_p2 * Pgood; // action 2 → state b\

            // ---- perseveration bias ----
            real persev = (action1[n - 1] == 2) ? p : -p;
            val_A2 += persev;
            val_A1 -= persev;

            // ---- decision rule ----
            int preferred_action = (val_A2 > val_A1) ? 2 : 1;
            int alt_action = 3 - preferred_action;

            real p_choice2 = 0;
            if (preferred_action == 2)
                p_choice2 = (1 - epsilon);
            else if (alt_action == 2)
                p_choice2 = epsilon;

            // ---- log likelihood ----
            if (action1[n] == 2)
                log_lik += log(p_choice2 + 1e-9);
            else
                log_lik += log(1 - p_choice2 + 1e-9);
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
    real<lower=0, upper=1> omega;
    real<lower=0, upper=1> epsilon;
    real<lower=-20, upper=20> p;
}

model {
    target += latent_state_agent(num_trials, subject, action1, s2, reward,
                                 omega, epsilon, p);
}
