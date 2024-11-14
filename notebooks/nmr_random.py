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


seeds = np.load('seeds.npy')
xtest = np.load('xtest.npy')
ytest = np.load('ytest.npy')

with open('xcandidates_original.pkl', 'rb') as f:
    xcandidates_original = pickle.load(f)
    
with open('ycandidates_original.pkl', 'rb') as f:
    ycandidates_original = pickle.load(f)
    
xtest = torch.tensor(xtest, dtype=dtype,device=device)
ytest = torch.tensor(ytest, dtype=dtype,device=device)


# torch.manual_seed(13)
# random.seed(13)

rand_xmax_candidates = []
rand_pred_mae = []
rand_pred_std = []
rand_pred_mean = []
rand_mae_runs =[]


def random_initial_data(x, y, initial_percent, seed=i):
    np.random.seed(seed)
    n = int(len(x)*initial_percent)
    idx = np.random.choice(len(x), n, replace=False).tolist()
    x_initial = [x[i] for i in idx]
    y_initial = [y[i] for i in idx]
    xcandidates = [x[i] for i in range(len(x)) if i not in idx]
    ycandidates = [y[i] for i in range(len(y)) if i not in idx]
    
    return x_initial, y_initial, xcandidates, ycandidates


for seed in seeds:
    # random.seed(seed)
    rand_xmax_candidates = []
    rand_pred_mae = []
    rand_pred_std = []
    rand_pred_mean = []
    xcandidates_rand = xcandidates_original.copy()
    ycandidates_rand = ycandidates_original.copy()
    
    
    xinit, yinit, xcandidates, ycandidates = random_initial_data(xcandidates_rand, ycandidates_rand, 0.05, seed=i)
    
    
    xinit = torch.cat(xinit,dim=0).to(device)
    yinit = torch.cat(yinit,dim=0).to(device)

    
    gp = SingleTaskGP(xinit, yinit).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
    fit_gpytorch_mll(mll)
    
    rand_ypred = gp(xtest)
    ypred = rand_ypred.mean.detach().numpy()
    # ystd = rand_ypred.stddev.detach().numpy()
    ymae = mean_absolute_error(ytest, ypred)
    
    pred_mae = []
    
    # pred_y.append(ypred)
    # pred_std.append(ystd)
    pred_mae.append(ymae)
    
    
    for i in tqdm(range(100)):
        if not xcandidates:
            break
        
        rand_select = random.choice(range(0, len(xcandidates)))
        
        xinit= torch.cat((xinit, xcandidates[max_index]), 0)
        yinit = torch.cat((yinit, ycandidates[max_index]), 0)
        
        # print('len of new train:', len(xtrain_rand))
        del xcandidates[rand_select]
        del ycandidates[rand_select]
        
        
        gp = SingleTaskGP(xinit, yinit).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
        fit_gpytorch_mll(mll)
        #predict the y values for the test set
        rand_ypred = gp(xtest)
        rand_ypred_mean = rand_ypred.mean.detach().numpy()
        # rand_pred_mean.append(rand_ypred_mean)
        
        
        ymae = mean_absolute_error(ytest, rand_ypred_mean)
        pred_mae.append(ymae)
        
    rand_mae_runs.append(pred_mae)