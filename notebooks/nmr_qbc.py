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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings(
    "ignore",
    message="Input data is not standardized.",
    category=InputDataWarning,
)
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


seeds = np.load('seeds_pkl/seeds.npy')
xtest = np.load('seeds_pkl/xtest.npy')
ytest = np.load('seeds_pkl/ytest.npy')


with open('seeds_pkl/xcandidates_original.pkl', 'rb') as f:
    xcandidates_original = pickle.load(f)
    
with open('seeds_pkl/ycandidates_original.pkl', 'rb') as f:
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
        
def process_batch(batch, committee):
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
    
    comitt_ymae = mean_absolute_error(ytest, ypred)
    gp_commit_lst.append(comitt_ymae)
    # for i in tqdm(range(25)):
    for inner_i in tqdm(range(100)):
        start_time = time.time()
        if not xcandidates:
            break
    
        for model in committee:
            model.fit(xinit, yinit)
    
        all_disagreement_scores = []
        for batch in xcandidates:
            batch_disagreement_scores = process_batch(batch, committee)
            all_disagreement_scores.append(batch_disagreement_scores)

        top_N_indx = np.argsort(all_disagreement_scores)[-1]
        
        X_to_query = xcandidates[top_N_indx]
        
        ylabel = ycandidates[top_N_indx]
        
        xinit = torch.cat((xinit, X_to_query), 0)
        yinit = torch.cat((yinit, ylabel), 0)
        
        del xcandidates[top_N_indx]
        del ycandidates[top_N_indx]
        
        gp = SingleTaskGP(xinit, yinit).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
        fit_gpytorch_mll(mll)
        posterior = gp(xtest)
        ypred = posterior.mean.detach().numpy()
            
        comitt_ymae = mean_absolute_error(ytest, ypred)
        gp_commit_lst.append(comitt_ymae)
        
        end_time = time.time()  
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        
    timing_per_run.append(iteration_times)
    qbc_runs.append(gp_commit_lst)
    