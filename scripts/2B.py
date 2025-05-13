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
from  os        import mkdir, remove
from  os.path   import join, exists
from  cmdstanpy import CmdStanModel
from  util      import *

MODEL_PARAMS = {
    'standard': ('alpha1', 'alpha2', 'lmbd',  'beta1', 'beta2', 'p', 'w'),
    'lmbda1':   ('alpha1', 'alpha2', 'beta1', 'beta2', 'p',     'w'),
    'p0':       ('alpha1', 'alpha2', 'lmbd',  'beta1', 'beta2', 'w'),
    'alpha':    ('alpha',  'lmbd',   'beta1', 'beta2', 'p',     'w'),
    'beta':     ('alpha1', 'alpha2', 'lmbd',  'beta',  'p',     'w'),
}
MODEL                  = 'alpha'
PARAM_NAMES            = MODEL_PARAMS[MODEL]
NOPTIM                 = 1000

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
    df         = pd.read_csv(os.path.join(DATA_DIRECTORY, "beh_noslow.csv"))
    # Selecting the condition
    conditions = df["condition"].unique()
    # Iterating over the conditions
    for condition in conditions:
        game_results            = df[df["condition"] == condition].copy()
        # Assign integer subject IDs (1, 2, 3, ...) for Stan
        unique_subjects         = game_results["participant"].unique()
        subject_id_map          = {subj: i+1 for i, subj in enumerate(unique_subjects)}  # 1-indexed for Stan
        game_results["subject"] = game_results["participant"].map(subject_id_map)

        # Prepare data for pooled fitting
        model_dat = {
            "num_trials":  len(game_results),
            "subject":     game_results["subject"].astype(int).to_list(),
            "action1":     game_results["choice1"].astype(int).to_list(),
            "action2":     game_results["choice2"].astype(int).to_list(),
            "s2":          game_results["final_state"].astype(int).to_list(),
            "reward":      game_results["reward"].astype(int).to_list(),
            "doRareTrans": (1 - game_results["common"]).astype(int).to_list(),
        }

        # Fit model
        stan_file  = f"2B_{MODEL}"
        stan_model = CmdStanModel(stan_file=f"{stan_file}.stan")  # <-- your modified Stan file name
        params     = optimize_model(stan_model, model_dat)

        # Save results
        log_lik      = params["lp__"]
        k            = len(PARAM_NAMES)
        n            = model_dat["num_trials"]  # total number of trials
        AIC          = 2 * k - 2 * log_lik
        BIC          = np.log(n) * k - 2 * log_lik
        param_values = [params[name] for name in PARAM_NAMES]

        # Save metrics + params to CSV
        full_results_df = pd.DataFrame(
            [[log_lik, AIC, BIC] + param_values],
            columns=["log_lik", "AIC", "BIC"] + list(PARAM_NAMES)
        )
        full_results_df.to_csv(os.path.join(RESULTS_DIRECTORY, f"{stan_file}-{condition}-metrics.csv"), index=False)

        print(f"✅ Metrics + parameters saved")
        # Cleaning up the stan files
        cleanup = [each for each in os.listdir(".") if "." not in each and not os.path.isdir(each)]
        for each in cleanup:
            remove(each)

if __name__ == "__main__":
    main()