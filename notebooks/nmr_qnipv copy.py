from typing import Any, Dict, Optional
import torch
import random
import os 
import argparse
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
from botorch.utils.sampling import draw_sobol_samples

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

import pickle

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = torch.double
# dtype = torch.float32


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
parser.add_argument("--data_dir", type=str, required=True, help="Directory where data files are located")
args = parser.parse_args()

output_dir = args.output_dir
data_dir = args.data_dir

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], device=device, dtype=dtype)

mcp = draw_sobol_samples(bounds=bounds, n=1024, q=1, seed=42).squeeze(1)
mcp.to(device=device, dtype=dtype)
# bounds


seeds = np.load('notebooks/seeds.npy')
xtest = np.load('notebooks/xtest.npy')
ytest = np.load('notebooks/ytest.npy')


with open('notebooks/xcandidates_original.pkl', 'rb') as f:
    xcandidates_original = pickle.load(f)
    
with open('notebooks/ycandidates_original.pkl', 'rb') as f:
    ycandidates_original = pickle.load(f)
    
xtest = torch.tensor(xtest, dtype=dtype,device=device)
ytest = torch.tensor(ytest, dtype=dtype,device=device)


def find_max_normalized_acqval(tensor_list, qNIVP):
    max_value = None
    max_index = -1
    acq_val_lst = []
    # torch.manual_seed(13)
    for i, tensor in enumerate(tensor_list):
        tensor_len = len(tensor)
        qNIVP_val = qNIVP(tensor)
        normalized_qNIVP_val = qNIVP_val / tensor_len
        acq_val_lst.append(normalized_qNIVP_val.item())  # Assuming it's a scalar tensor

        # Check if this is the maximum value so far
        if max_value is None or normalized_qNIVP_val > max_value:
            max_value = normalized_qNIVP_val
            max_index = i

    return max_value, max_index, acq_val_lst


def random_initial_data(x, y, initial_percent, seed=i):
    np.random.seed(seed)
    n = int(len(x)*initial_percent)
    idx = np.random.choice(len(x), n, replace=False).tolist()
    x_initial = [x[i] for i in idx]
    y_initial = [y[i] for i in idx]
    xcandidates = [x[i] for i in range(len(x)) if i not in idx]
    ycandidates = [y[i] for i in range(len(y)) if i not in idx]
    
    return x_initial, y_initial, xcandidates, ycandidates




rand_selection_mae = []
xmax_candidates = []
pred_mae = []
pred_y = []
pred_std = []
qnipv_runs =[]



def find_max_normalized_acqval(tensor_list, qNIVP):
    max_value = None
    max_index = -1
    acq_val_lst = []
    # torch.manual_seed(13)
    for i, tensor in enumerate(tensor_list):
        tensor_len = len(tensor)
        qNIVP_val = qNIVP(tensor)
        normalized_qNIVP_val = qNIVP_val / tensor_len
        acq_val_lst.append(normalized_qNIVP_val.item())
        if max_value is None or normalized_qNIVP_val > max_value:
            max_value = normalized_qNIVP_val
            max_index = i

    return max_value, max_index, acq_val_lst


for i in tqdm(seeds):
    xcandidates = xcandidates_original.copy()
    ycandidates = ycandidates_original.copy()
    
    
    
    xinit, yinit, xcandidates, ycandidates = random_initial_data(xcandidates, ycandidates, 0.05, seed=i)
    
    
    xinit = torch.cat(xinit,dim=0)
    yinit = torch.cat(yinit,dim=0)

    
    gp = SingleTaskGP(xinit, yinit)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    posterior = gp(xtest)
    ypred = posterior.mean.detach().numpy()
    ystd = posterior.stddev.detach().numpy()
    
    
    ymae = mean_absolute_error(ytest, ypred)
    
    pred_mae = []
    pred_y.append(ypred)
    pred_std.append(ystd)
    pred_mae.append(ymae)

    for inner_i in tqdm(range(len(xcandidates))):
        if not len(xcandidates):
            break
        
        qNIVP = qNegIntegratedPosteriorVariance(gp, mc_points= mcp)
        
        
        max_value, max_index, acq_val_lst = find_max_normalized_acqval(xcandidates, qNIVP)
        xmax_candidates.append(max_index)
        
        # print(f'pre-addtion of new ten',len(xinit))
        xinit= torch.cat((xinit, xcandidates[max_index]), 0)
        yinit = torch.cat((yinit, ycandidates[max_index]), 0)
        
        
        del xcandidates[max_index]
        del ycandidates[max_index]
        
       
        
        gp = SingleTaskGP(xinit, yinit) 
        # gp = SingleTaskGP(xinit, ytrain_,covar_module=rbf_kernel)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        #predict the y values for the test set
        ypred = gp(xtest)
        ypred_mean = ypred.mean.detach().numpy()
        pred_y.append(ypred_mean)

        ymae = mean_absolute_error(ytest, ypred_mean)
        # print('mean absolute error: ', ymae)
        pred_mae.append(ymae)
        ystd = gp(xtest).stddev
        ystd = ystd.detach().numpy()
        pred_std.append(ystd)
    qnipv_runs.append(pred_mae)
    

# np.save('qnipv_runs.txt', np.array(qnipv_runs))
np.save(os.path.join(output_dir, 'qnipv_runs.txt'), np.array(qnipv_runs))


