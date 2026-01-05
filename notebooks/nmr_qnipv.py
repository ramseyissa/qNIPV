import torch
import random
import os
import argparse
import pandas as pd
import numpy as np
import warnings

from tqdm import tqdm
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions.warnings import InputDataWarning
from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, standardize
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
# dtype = torch.float32


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
parser.add_argument("--data_dir", type=str, required=True, help="Directory where data files are located")
args = parser.parse_args()

output_dir = args.output_dir
data_dir = args.data_dir

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define paths to data files using the provided data directory
labels_path = os.path.join(data_dir, "labels.txt")
colvar_path = os.path.join(data_dir, "COLVAR_NoOpt.csv")

with open(labels_path, "r") as file:
    contents = file.readlines()

list_of_peaks = []
for items in contents:
    items = items.strip()
    new_list = [items]
    list_of_peaks.append(new_list)

peak_lists = [
    [round(float(val), 2) for val in sublist[0].split()] for sublist in list_of_peaks
]

df = pd.read_csv(colvar_path)
peak_lists.pop(0)

group_id = 0
group_id_list_ = []
for lst in peak_lists:
    group_id += 1
    for i in range(len(lst)):
        group_id_list_.append(group_id)      


df_loc_ = df.loc[1:,['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'l1', 'l2', 'l3',
       'l4']]

group_id_list = [data for idx, data in enumerate(group_id_list_) if idx in df_loc_.index]
y_values_cleaned = [data for idx, data in enumerate(df["ppm"]) if idx in df_loc_.index]

x_train_pre_Normalize = df_loc_

min_val_list = []
max_val_list = []
for col in x_train_pre_Normalize.columns:
    min_val = x_train_pre_Normalize[col].min()
    max_val = x_train_pre_Normalize[col].max()
    min_val_list.append(min_val)
    max_val_list.append(max_val)
    
xlower_bounds = torch.tensor(min_val_list, dtype=torch.double)
xupper_bounds = torch.tensor(max_val_list, dtype=torch.double)

xbounds = torch.stack([xlower_bounds, xupper_bounds])

x_tensors = torch.tensor(x_train_pre_Normalize.values, dtype=dtype,device=device)
y_tensors = torch.tensor(y_values_cleaned, dtype=dtype, device=device).unsqueeze(-1)



x_train_normalized = normalize(x_tensors, bounds=xbounds)  
y_train_standardized = standardize(y_tensors)

unique_values = set(group_id_list)
uniq_grps = list(unique_values)


x_dict = {grp_id: [] for grp_id in uniq_grps}
y_dict = {grp_id: [] for grp_id in uniq_grps}

for grp_id, x_train_val in zip(group_id_list, x_train_normalized):
    x_dict[grp_id].append(x_train_val.unsqueeze(0))

for grp_id, y_train_val in zip(group_id_list, y_train_standardized):
    y_dict[grp_id].append(y_train_val.unsqueeze(0).unsqueeze(1))
      
bounds = torch.stack([torch.min(x_train_normalized, axis=0).values, 
                      torch.max(x_train_normalized, axis=0).values])

mcp = draw_sobol_samples(bounds=bounds, n=1024, q=1, seed=42).squeeze(1)
mcp.to(device=device, dtype=dtype)

tensor_xlist = []
for key, value in x_dict.items():
    processed_tensor = torch.stack(value).squeeze(1)
    tensor_xlist.append(processed_tensor)
    
tensor_ylist = []
for key, value in y_dict.items():
    processed_tensor = torch.stack(value).squeeze(1)
    tensor_ylist.append(processed_tensor)
    
random.seed(13)

xtest_indexes = random.sample(range(0, 530), 159)
xtest_indexes

xtest = [tensor_xlist[i] for i in xtest_indexes]
ytest = [tensor_ylist[i] for i in xtest_indexes]

xtest = torch.cat(xtest, dim=0)
ytest = torch.cat(ytest, dim=0)
ytest = ytest.squeeze(1)

xcandidates_indexes = [i for i in range(0,529) if i not in xtest_indexes]
ycandidates_indexes = [i for i in range(0,529) if i not in xtest_indexes]

xcandidates_original = [tensor_xlist[i] for i in xcandidates_indexes]
ycandidates_original = [tensor_ylist[i].squeeze(1) for i in ycandidates_indexes]

xcandidates_full_set = torch.cat(xcandidates_original, dim=0)
ycandidates_full_set = torch.cat(ycandidates_original, dim=0)

gp = SingleTaskGP(xcandidates_full_set, ycandidates_full_set).to(device)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
fit_gpytorch_mll(mll).to(device)

ypred = gp(xtest)
ypred_mean = ypred.mean.detach().numpy()

ymae = mean_absolute_error(ytest, ypred_mean)
ymae

def find_max_normalized_acqval(tensor_list, qNIVP):
    max_value = None
    max_index = -1
    acq_val_lst = []
    
    for i, tensor in enumerate(tensor_list):
        tensor_len = len(tensor)
        qNIVP_val = qNIVP(tensor)
        normalized_qNIVP_val = qNIVP_val / tensor_len
        acq_val_lst.append(normalized_qNIVP_val.item())  

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


seeds = [random.randint(1, 5000) for _ in range(25)]
xcandidates = xcandidates_original.copy()
ycandidates = ycandidates_original.copy()

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
        
        xinit= torch.cat((xinit, xcandidates[max_index]), 0)
        yinit = torch.cat((yinit, ycandidates[max_index]), 0)
        
        
        del xcandidates[max_index]
        del ycandidates[max_index]
        
        gp = SingleTaskGP(xinit, yinit) 
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        ypred = gp(xtest)
        ypred_mean = ypred.mean.detach().numpy()
        pred_y.append(ypred_mean)

        ymae = mean_absolute_error(ytest, ypred_mean)
        pred_mae.append(ymae)
        ystd = gp(xtest).stddev
        ystd = ystd.detach().numpy()
        pred_std.append(ystd)
    qnipv_runs.append(pred_mae)
    

np.save(os.path.join(output_dir, 'qnipv_runs.txt'), np.array(qnipv_runs))


