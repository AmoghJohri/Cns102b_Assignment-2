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
HYBRID_SINGLE_PARAMS   = join(RESULTS_DIRECTORY, "hybrid_single_dummy.csv")

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
    game_results = pd.read_csv(os.path.join(DATA_DIRECTORY, "beh_noslow_dummy.csv"))
    NTRIALS      = game_results.trial.max() + 1
    # Perform individual fits too to get the mean parameters
    stan_model = CmdStanModel(stan_file=f"hybrid_single.stan")
    results    = []
    for part, part_data in game_results.groupby("participant"):
        model_dat = {
            "num_trials": len(part_data),
            "action1": list(part_data.choice1),
            "action2": list(part_data.choice2),
            "s2": list(part_data.final_state),
            "reward": list(part_data.reward),
        }
        params    = optimize_model(stan_model, model_dat)
        condition = part_data.iloc[0].condition
        results.append((part, condition, params))
    with open(HYBRID_SINGLE_PARAMS, "w") as outf:
        outf.write(f'participant,condition,{",".join(PARAM_NAMES)}\n')
        for part, condition, params in results:
            line = "{},{},{}\n".format(
                part,
                condition,
                ",".join([str(params[k]) for k in PARAM_NAMES]),
            )
            outf.write(line)

if __name__ == "__main__":
    main()