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

from botorch.utils.transforms import normalize, standardize



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


# def calc_predictions(candidates,committee):
#     predicted_list = []
#     for model in committee:
#         predicted_val = []
#         for candidate in candidates:
#             predicted = model.predict(candidate)
#             predicted_val.append(predicted)
#         predicted_list.append(predicted_val)
#     return predicted_list    
        
        
def process_batch(batch, committee):
    # Convert torch tensor to numpy array
    batch_np = batch.numpy()
    
    predictions = np.array([model.predict(batch_np) for model in committee])
    
    disagreement_batch = np.var(predictions, axis=0)
    disagreement_scores = sum(disagreement_batch) / len(disagreement_batch)
        
    return disagreement_scores



gp_commit_lst = []

committee = [
    RandomForestRegressor(),
    SVR(),
    DecisionTreeRegressor()
]

comit_pred_mae = []

commit_seeds = []

qbc_runs = []
timing_per_iteration = []
timing_per_run = []

for seed in seeds:
    gp_commit_lst = []
    iteration_times = []
    xcandidates_qbc = xcandidates_original.copy()
    ycandidates_qbc = ycandidates_original.copy()
    
    xinit, yinit, xcandidates, ycandidates = random_initial_data(xcandidates_qbc, ycandidates_qbc, 0.05, seed=seed)
    
    xinit = torch.cat(xinit,dim=0).to(device)
    yinit = torch.cat(yinit,dim=0).to(device)

    gp = SingleTaskGP(xinit, yinit).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
    fit_gpytorch_mll(mll)
    
    posterior = gp(xtest)
    ypred = posterior.mean.detach().numpy()
    
    # gp_commit_lst.append(ypred)
    comitt_ymae = mean_absolute_error(ytest, ypred)
    gp_commit_lst.append(comitt_ymae)
    # for i in tqdm(range(25)):
    for inner_i in tqdm(range(100)):
        start_time = time.time()
        if not xcandidates:
            break
    
        for model in committee:
            # print(model)
            model.fit(xinit, yinit)
        
        all_disagreement_scores = []
        for batch in xcandidates:
            # print(batch)
            batch_disagreement_scores = process_batch(batch, committee)
            # print(batch_disagreement_scores)
            all_disagreement_scores.append(batch_disagreement_scores)

        # N = 1 
        top_N_indx = np.argsort(all_disagreement_scores)[-1]
        # top_N_indx = top_N_indices[0]
        
        X_to_query = xcandidates[top_N_indx]
        
        ylabel = ycandidates[top_N_indx]
        
        xinit = torch.cat((xinit, X_to_query), 0)
        yinit = torch.cat((yinit, ylabel), 0)
        
        del xcandidates[top_N_indx]
        del ycandidates[top_N_indx]
        
        # xcandidates.pop(top_N_indx)
        # ycandidates.pop(top_N_indx)
        
        gp = SingleTaskGP(xinit, yinit).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
        fit_gpytorch_mll(mll)
        posterior = gp(xtest)
        ypred = posterior.mean.detach().numpy()
            
        comitt_ymae = mean_absolute_error(ytest, ypred)
        gp_commit_lst.append(comitt_ymae)
        
        end_time = time.time()  # End timing
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        
        # print(f'length of gp_commit_lst: {len(gp_commit_lst)}')
    timing_per_run.append(iteration_times)
    qbc_runs.append(gp_commit_lst)
    np.save('qbc_nmr_runs.npy', np.array(qbc_runs))
    np.save('timing_qbc_nmr.npy', np.array(timing_per_run))