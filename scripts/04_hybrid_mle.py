# Copyright (C) 2016, 2017, 2018, 2023 Carolina Feher da Silva

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fit the hybrid reinforcement learning model using maximum-likelihood estimation"""
import sys
import numpy  as np
import pandas as pd
from  os        import mkdir
from  os.path   import join, exists
from  cmdstanpy import CmdStanModel
from  util      import *

PARAM_NAMES            = ('alpha1', 'alpha2', 'lmbd', 'beta1', 'beta2', 'p', 'w')
NOPTIM                 = 1000
HYBRID_SINGLE_PARAMS   = join(RESULTS_DIRECTORY, "hybrid_single.csv")

def optimize_model(stan_model, model_dat):
    log_lik = -np.inf
    for _ in range(NOPTIM):
        while True:
            try:
                op = stan_model.optimize(data=model_dat)
            except RuntimeError as rterror:
                sys.stderr.write(f"Error: {str(rterror)}\n")
            else:
                if op.converged:
                    break
        if op.optimized_params_dict["lp__"] > log_lik:
            log_lik = op.optimized_params_dict["lp__"]
            params = op.optimized_params_dict
    return params

def main():
    game_results = pd.read_csv(os.path.join(DATA_DIRECTORY, "beh_noslow.csv"))
    NTRIALS      = game_results.trial.max() + 1
    # Perform individual fits too to get the mean parameters
    stan_model = CmdStanModel(stan_file=f"hybrid_single.stan")
    results    = []
    for condition, condition_data in game_results.groupby("condition"):
        model_dat = {
            "num_trials": len(condition_data),
            "action1": list(condition_data.choice1),
            "action2": list(condition_data.choice2),
            "s2": list(condition_data.final_state),
            "reward": list(condition_data.reward),
        }
        params = optimize_model(stan_model, model_dat)
        results.append((condition, params))
    with open(HYBRID_SINGLE_PARAMS, "w") as outf:
        outf.write(f'condition,{",".join(PARAM_NAMES)}\n')
        for condition, params in results:
            line = "{},{}\n".format(
                condition,
                ",".join([str(params[k]) for k in PARAM_NAMES]),
            )
            outf.write(line)
    # ---- (2) additional full results file ----
    full_results = []
    for condition, params in results:
        log_lik = params["lp__"]
        k = len(PARAM_NAMES)
        n = game_results[game_results["condition"] == condition].shape[0]
        AIC = 2 * k - 2 * log_lik
        BIC = np.log(n) * k - 2 * log_lik
        param_values = [params[name] for name in PARAM_NAMES]
        full_results.append([condition] + param_values + [log_lik, AIC, BIC])

    output_file = HYBRID_SINGLE_PARAMS.replace(".csv", "_ex.csv")
    columns     = ["condition"] + list(PARAM_NAMES) + ["log_lik", "AIC", "BIC"]
    df_results  = pd.DataFrame(full_results, columns=columns)
    df_results.to_csv(output_file, index=False)
    print(f"Saved full results with fit metrics to {output_file}")

if __name__ == "__main__":
    main()