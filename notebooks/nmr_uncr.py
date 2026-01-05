import torch
import time
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
    
    uncr_ypred = gp(xtest)
    uncr_ypred_mean = uncr_ypred.mean.detach().numpy()
    
    uncr_ymae = mean_absolute_error(ytest, uncr_ypred_mean)
    
    uncr_pred_mae.append(uncr_ymae)
    
    
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
    
        uncertainties = np.array(uncertainties_list)
     
        max_uncertainty_idx = uncertainties.argmax()
        
        xinit= torch.cat((xinit, xcandidates[max_uncertainty_idx]), 0).to(device)
        yinit = torch.cat((yinit, ycandidates[max_uncertainty_idx]), 0).to(device)
        
        del xcandidates[max_uncertainty_idx]
        del ycandidates[max_uncertainty_idx]
        
        gp = SingleTaskGP(xinit, yinit).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
        fit_gpytorch_mll(mll)
        
        uncr_ypred = gp(xtest)
        uncr_ypred_mean = uncr_ypred.mean.detach().numpy()
        
        uncr_ymae = mean_absolute_error(ytest, uncr_ypred_mean)
        uncr_pred_mae.append(uncr_ymae)
        
        end_time = time.time()  
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        
    timing_per_run.append(iteration_times)
    uncr_mae_runs.append(uncr_pred_mae)
    
    np.save('uncr_nmr_runs.npy', np.array(uncr_mae_runs))
    np.save('timing_uncertainty_nmr.npy', np.array(timing_per_run))

  