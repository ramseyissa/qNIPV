import torch
import random
import numpy as np
import pickle
import warnings

from tqdm import tqdm
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions.warnings import InputDataWarning
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings(
    "ignore",
    message="Input data is not standardized.",
    category=InputDataWarning,
)
warnings.filterwarnings("ignore")


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
    ymae = mean_absolute_error(ytest, ypred)
    
    pred_mae = []
    pred_mae.append(ymae)
    
    for i in tqdm(range(100)):
        if not xcandidates:
            break
        
        rand_select = random.choice(range(0, len(xcandidates)))
        
        xinit= torch.cat((xinit, xcandidates[max_index]), 0)
        yinit = torch.cat((yinit, ycandidates[max_index]), 0)
        
        del xcandidates[rand_select]
        del ycandidates[rand_select]
        
        gp = SingleTaskGP(xinit, yinit).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
        fit_gpytorch_mll(mll)
        rand_ypred = gp(xtest)
        rand_ypred_mean = rand_ypred.mean.detach().numpy()
        
        ymae = mean_absolute_error(ytest, rand_ypred_mean)
        pred_mae.append(ymae)
        
    rand_mae_runs.append(pred_mae)