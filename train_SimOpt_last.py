#CONTROLLARE TUTTE LE IMPORTAZIONI
import gym
import argparse
import numpy as np
import random
import torch
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys

from pathlib import Path

from skopt import gp_minimize
from skopt.space import Real 
from skopt.plots import plot_convergence

import seaborn as sns
import random

from utils_simopt_last import train_and_save, BO_obj, update_distribution

# Fix seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

tol_std = 1e-4  # Tolerance for standard deviation convergence 
BO_calls = 30  # Number of Bayesian Optimization calls

# Main SimOpt loop
def main(): 
    
    # 1) distribution p_phi_0
    # Mean (μ) and standard deviation (σ) for initial distributions for thigh, leg and foot masses (torso mass is fixed)
    phi_masses = {
        "thigh":   [3.92699082, 0.5],
        "leg": [2.71433605, 0.5],
        "foot":  [5.08938010, 0.5],
    }

    step = 0
    while all(std > tol_std for _, std in phi_masses.values()):

        # 2) Bayesian Optimization
        

        # search space for masses: mu ± 2σ 
        search_space = [
            Real(phi_masses["thigh"][0] - 2 * phi_masses["thigh"][1], phi_masses["thigh"][0] + 2 * phi_masses["thigh"][1], name="thigh"),
            Real(phi_masses["leg"][0]   - 2 * phi_masses["leg"][1],   phi_masses["leg"][0]   + 2 * phi_masses["leg"][1],   name="leg"),
            Real(phi_masses["foot"][0]  - 2 * phi_masses["foot"][1],  phi_masses["foot"][0]  + 2 * phi_masses["foot"][1],  name="foot")]

        #search space for masses within a range of 0.5 to 2 times the real value masses (ground truth)
        #hopper = gym.make('CustomHopper-target-v0')
        #hopper_masses = hopper.get_parameters() 
        #search_space = [
        #    Real(0.5* hopper_masses[1], 2*hopper_masses[1] , name="thigh"),
        #    Real(0.5* hopper_masses[2], 2*hopper_masses[2] , name="leg"),
        #    Real(0.5* hopper_masses[3], 2*hopper_masses[3] , name="foot")
        #]
        #check
        print("Search space for Bayesian Optimization:")
        for dim in search_space:
            print(f"{dim.name}: [{dim.low:.3f}, {dim.high:.3f}]")       

        print(f"\nStarting Bayesian Optimization step {step}")

        res = gp_minimize(
            func=lambda x: BO_obj({
                "thigh": [x[0], phi_masses["thigh"][1]],
                "leg":   [x[1], phi_masses["leg"][1]],
                "foot":  [x[2], phi_masses["foot"][1]],
            }),     
            dimensions=search_space,
            n_calls=BO_calls,   
            random_state=SEED,
            verbose=True
        )

        # Plot convergence
        plt.figure(figsize=(10, 5))
        plot_convergence(res)
        plt.title(f"Bayesian Optimization Convergence (Step {step})")
        plt.savefig(f"bo_convergence_step_{step}.png")
        plt.close()

        # BO results 
        
        print("Best found parameters:", res.x)
        print("Best found value (discrepancy):", res.fun)
        min_gap = []
        min_gap.append(res.fun)
        print(f"Minimum gap found until {step}:", min_gap)
        rec = {"thigh": res.x[0], "leg": res.x[1], "foot": res.x[2]}
        print("Recommended masses from Bayesian Optimization:", rec)    #scrivere meglio

        #HA SENSO FARLO COSì O MAGARI RIMPIAZZIAMO LE MEDIE DELL'ITERAZIONE PRECEDENTE?
        #possibile cambiamento di strategia, media pesata?
        # Update masses distributions
        for key in phi_masses.keys():
            samples = np.random.normal(phi_masses[key][0], phi_masses[key][1], 300)
            samples = np.append(samples, rec[key])
            phi_masses[key][0] = float(np.mean(samples))   # μ
            phi_masses[key][1] = float(np.std(samples))    # σ 

        # Update masses distributions with the recommended values
        #weighted mean and std
        #for key in phi_masses.keys():
        #    mu_old, std_old = phi_masses[key] ########non so se abbia senso
        #    mu_new, std_new = update_distribution(mu_old,std_old,rec[key]) 
        #    phi_masses[key][0] = mu_new       # μ
        #    phi_masses[key][1] = std_new      # σ   

            print(f"Updated {key} mass distribution: μ={phi_masses[key][0]:.3f}, σ={phi_masses[key][1]:.3f}")

        step += 1
    
    # After convergence, phi_masses contains the final distributions
    phi_optimal = phi_masses.copy()
    print("\nConverged distributions:")
    for key, (mu, std) in phi_optimal.items():
        print(f"{key} mass: μ={mu:.3f}, σ={std:.3f}")

    # Final training with converged distributions
    print("\nStarting final training phase …")
    train_and_save(
        'CustomHopper-source-v0', 
        './simopt_hopper_logs_source',  # Log directory for training
        './simopt_hopper_final_source', # Model path for saving the final model
        total_timesteps=2_000_000,
        phi=phi_optimal
    )

    #
    

if __name__ == "__main__":
    main()
