from typing import Any, Dict, Optional
import torch
import random
import os 

from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from botorch.models.gp_regression import SingleTaskGP
from tqdm import tqdm
from torch import Tensor
from botorch.acquisition.active_learning import (
    MCSampler,
    qNegIntegratedPosteriorVariance,
)

from botorch.fit import fit_gpytorch_mll
from sklearn.model_selection import train_test_split
from botorch.models.gp_regression import SingleTaskGP

from sklearn.metrics import mean_absolute_error



from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import warnings


# warnings.filterwarnings("ignore", category=botorch.exceptions.BotorchWarning)

from botorch.exceptions.warnings import BotorchTensorDimensionWarning, InputDataWarning
warnings.filterwarnings(
            "ignore",
            message="Input data is not standardized.",
            category=InputDataWarning,
        )
import warnings
warnings.filterwarnings("ignore")



from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition.active_learning import (
    MCSampler,
    qNegIntegratedPosteriorVariance,
)

import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


seeds = np.load('notebooks/seeds.npy')
xtest = np.load('notebooks/xtest.npy')
ytest = np.load('notebooks/ytest.npy')


with open('notebooks/xcandidates_original.pkl', 'rb') as f:
    xcandidates_original = pickle.load(f)
    
with open('notebooks/ycandidates_original.pkl', 'rb') as f:
    ycandidates_original = pickle.load(f)
    
xtest = torch.tensor(xtest, dtype=dtype,device=device)
ytest = torch.tensor(ytest, dtype=dtype,device=device)




def random_initial_data(x, y, initial_percent, seed):
    np.random.seed(seed)
    n = int(len(x)*initial_percent)
    idx = np.random.choice(len(x), n, replace=False).tolist()
    x_initial = [x[i] for i in idx]
    y_initial = [y[i] for i in idx]
    xcandidates = [x[i] for i in range(len(x)) if i not in idx]
    ycandidates = [y[i] for i in range(len(y)) if i not in idx]
    
    return x_initial, y_initial, xcandidates, ycandidates

uncr_mae_runs = []

timing_per_iteration = []
timing_per_run = []

for seed in seeds:
    # random.seed(seed)
    iteration_times = []
    uncr_pred_mae = []
    uncr_pred_std = []
    uncr_pred_mean = []
    
    xcandidates_uncr = xcandidates_original.copy()
    ycandidates_uncr = ycandidates_original.copy()
    
    
    xinit, yinit, xcandidates, ycandidates = random_initial_data(xcandidates_uncr, ycandidates_uncr, 0.05, seed=seed)
    

    
    xinit = torch.cat(xinit,dim=0).to(device)
    yinit = torch.cat(yinit,dim=0).to(device)

    
    gp = SingleTaskGP(xinit, yinit).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
    fit_gpytorch_mll(mll)
    
    
    
    # Predict on the test set initially
    uncr_ypred = gp(xtest)
    uncr_ypred_mean = uncr_ypred.mean.detach().numpy()
    # uncr_pred_mean.append(uncr_ypred_mean)
    
    uncr_ymae = mean_absolute_error(ytest, uncr_ypred_mean)
    
    uncr_pred_mae.append(uncr_ymae)
    
    
    
    # for i in tqdm(range(100)):
    for inner_i in tqdm(range(len(xcandidates))):
        start_time = time.time()
        if not xcandidates:
            break
        
        uncertainties_list = []
        for i, candidate_tensor in enumerate(xcandidates):
            # Get posterior for this tensor
            posterior = gp(candidate_tensor)
            # Calculate mean uncertainty for this tensor
            tensor_uncertainty = posterior.stddev.mean().detach().numpy()
            uncertainties_list.append(tensor_uncertainty)
    
    # Convert list to numpy array for argmax
        uncertainties = np.array(uncertainties_list)
        
        # posterior_candidates = gp([i for i in xcandidates])
        
        # uncertainties = [posterior_candidates.stddev.detach().numpy() for i in posterior_candidates]  

        # Find the index of the candidate point with the highest uncertainty
        max_uncertainty_idx = uncertainties.argmax()
        
        xinit= torch.cat((xinit, xcandidates[max_uncertainty_idx]), 0).to(device)
        yinit = torch.cat((yinit, ycandidates[max_uncertainty_idx]), 0).to(device)
        
        # print('len of new train:', len(xtrain_rand))
        del xcandidates[max_uncertainty_idx]
        del ycandidates[max_uncertainty_idx]
        
        gp = SingleTaskGP(xinit, yinit).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
        fit_gpytorch_mll(mll)
        
        uncr_ypred = gp(xtest)
        uncr_ypred_mean = uncr_ypred.mean.detach().numpy()
        
        uncr_ymae = mean_absolute_error(ytest, uncr_ypred_mean)
        uncr_pred_mae.append(uncr_ymae)
        
        end_time = time.time()  # End timing
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        
    timing_per_run.append(iteration_times)
    uncr_mae_runs.append(uncr_pred_mae)
    
    np.save('uncr_nmr_runs.npy', np.array(uncr_mae_runs))
    np.save('timing_uncertainty_nmr.npy', np.array(timing_per_run))

  