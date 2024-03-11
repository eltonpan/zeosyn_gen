import os
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
import pdb
from geomloss import SamplesLoss
if os.path.basename(os.getcwd()) == 'zeosyn_gen': # if running from the main directory
    os.chdir('data') # change directory into the data folder to allow imports below
from syn_variables import zeo_cols

def clean_cbus(cbu_str):
    # This fixes cbus column contains a str representation of list, instead of list itself #
    if type(cbu_str) == str: # if str
        return literal_eval(cbu_str)
    else: # if not str, eg. NaN
        cbu_str

def calculate_tanimoto_similarity(molecule_smiles1, molecule_smiles2, plot=False, verbose=False):
    '''Calculate Tanimoto similarity between two molecules.'''

    # Create RDKit molecules from SMILES strings
    mol1 = Chem.MolFromSmiles(molecule_smiles1)
    mol2 = Chem.MolFromSmiles(molecule_smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string(s)")

    # Generate Morgan fingerprints with radius 2 (you can adjust the radius if needed)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3)

    if plot:
        plt.figure()
        plt.imshow(Draw.MolsToImage(
            [mol1, mol2]))
        plt.axis('off')
        plt.show()

    # Calculate Tanimoto similarity
    tanimoto_similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    if verbose:
        print('Tanimoto similarity:', round(tanimoto_similarity,3))

    return tanimoto_similarity

# Zeolite physicochemical descriptors 
df_zeo = pd.read_csv('zeolite_descriptors.csv').drop(columns = ['Unnamed: 0'])
df_zeo = df_zeo[['Code']+zeo_cols] # select specific features
codes = df_zeo['Code'] # save codes
scaler = StandardScaler()
df_zeo = pd.DataFrame(scaler.fit_transform(df_zeo.drop(columns=['Code'])), columns=zeo_cols)
df_zeo['Code'] = codes # add back codes
zeo2feat = {}
for zeo in df_zeo['Code']:
    zeo2feat[zeo] = np.array(df_zeo[df_zeo['Code'] == zeo][zeo_cols])

# Zeolite EGNN
df_zeo_egnn = pd.read_csv('CVAE_EGNN_embeddings_2023-07-13.csv').drop(columns = ['Unnamed: 0'])
# egnn_cols = df_zeo_egnn.drop(columns=['Code']).columns
# codes = df_zeo_egnn['Code'] # save codes
# scaler = StandardScaler()
# df_zeo_egnn = pd.DataFrame(scaler.fit_transform(df_zeo_egnn.drop(columns=['Code'])), columns=egnn_cols)
# df_zeo_egnn['Code'] = codes # add back codes
zeo2egnn = {}
for zeo in df_zeo_egnn['Code']:
    zeo2egnn[zeo] = np.array(df_zeo_egnn[df_zeo_egnn['Code'] == zeo].drop(columns=['Code']))

# Zeolite binding energies
df_zeo = pd.read_csv('zeolite_binding_energy.csv').drop(columns = ['Unnamed: 0'])
codes = df_zeo['zeo'] # save codes
scaler = StandardScaler()
df_zeo = pd.DataFrame(scaler.fit_transform(df_zeo.drop(columns=['zeo'])))
df_zeo['zeo'] = codes
zeo2be = {}
for zeo in df_zeo['zeo']:
    zeo2be[zeo] = np.array(df_zeo[df_zeo['zeo'] == zeo].drop(columns=['zeo']))

# Zeolite graph distance
df_zeo_graph = pd.read_csv('zeolite_graph_distance.csv')

# # Zeolite CBUs
# df_cbu = pd.read_csv('cbus.csv').rename(columns = {'Unnamed: 0': 'Code'})
# df_cbu['cbu'] = list(map(clean_cbus, df_cbu['cbu'].values))
# cbus_unique = []
# for cbus in df_cbu['cbu'].values:
#     for cbu in cbus:
#         cbus_unique.append(cbu)
# cbus_unique = set(cbus_unique)
# cbu_oh = np.zeros([len(df_without_na), len(cbus_unique)])
# pdb.set_trace()

def get_zeolite_similarity(zeo1, zeo2, feat_type='physicochemical'):

    assert feat_type in ['physicochemical', 'egnn', 'be', 'graph', 'soap'], "feat_type must be one of the following: 'physicochemical', 'egnn', 'be', 'graph', 'soap'"

    if feat_type in ['graph', 'soap']:
        zeo1, zeo2 = sorted([zeo1, zeo2]) # because the csv file is triangular i.e. lower character comes first eg. (AEI, CHA) not (CHA, AEI)
        
        if zeo1 == zeo2:
            return 0.
        else:
            if (zeo1 in df_zeo_graph['Zeo1'].tolist()) and (zeo2 in df_zeo_graph['Zeo2'].tolist()):
                if feat_type == 'graph':
                    return df_zeo_graph[(df_zeo_graph['Zeo1']==zeo1) & (df_zeo_graph['Zeo2']==zeo2)]['D'].item()
                elif feat_type == 'soap':
                    return df_zeo_graph[(df_zeo_graph['Zeo1']==zeo1) & (df_zeo_graph['Zeo2']==zeo2)]['SOAP'].item()
            else:
                return None
    else:

        if feat_type == 'physicochemical':
            mapping = zeo2feat
        elif feat_type == 'egnn':
            mapping = zeo2egnn
        elif feat_type == 'be':
            mapping = zeo2be

        if (zeo1 in mapping.keys()) and (zeo2 in mapping.keys()):
            zeo1 = mapping[zeo1]
            zeo2 = mapping[zeo2]
            # 2A) Mean cosine similarity of all pair of zeos
            zeo_sim = cosine_similarity(zeo1, zeo2)[0][0]
            return zeo_sim
        else:
            return None

# Early version - to be deleted
# def maximum_mean_discrepancy(X, Y, kernel_type='gaussian', sigmas=None):
#     """
#     Calculate the Maximum Mean Discrepancy (MMD) between two distributions.

#     Parameters:
#     - X: Samples from the first distribution (numpy array or PyTorch tensor).
#     - Y: Samples from the second distribution (numpy array or PyTorch tensor).
#     - kernel_type: Type of kernel function ('linear', 'gaussian', etc.).
#     # - gamma: Parameter for the Gaussian kernel (if kernel_type is 'gaussian').
#     - sigmas: List of floats. Standard deviation(s) of the Gaussian kernel (if kernel_type is 'gaussian').

#     Returns:
#     - mmd: Maximum Mean Discrepancy between X and Y.
#     """
#     X, Y = np.asarray(X), np.asarray(Y)

#     if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
#         X, Y = torch.from_numpy(X), torch.from_numpy(Y)

#     if not torch.is_tensor(X):
    #     raise ValueError("Input X should be a numpy array or a PyTorch tensor.")

    # if not torch.is_tensor(Y):
    #     raise ValueError("Input Y should be a numpy array or a PyTorch tensor.")

    # if X.dim() != 2 or Y.dim() != 2:
    #     raise ValueError("Input tensors should be 2-dimensional.")

    # if X.size(1) != Y.size(1):
    #     raise ValueError("Input tensors should have the same number of features.")

    # if kernel_type == 'linear':
    #     K_XX = torch.mm(X, X.t())
    #     K_YY = torch.mm(Y, Y.t())
    #     K_XY = torch.mm(X, Y.t())

    # elif kernel_type == 'gaussian':
    #     if sigmas is None:
    #         sigmas = [3., 4., 5., 7.]

    #     X_norm = torch.sum(X * X, dim=1).unsqueeze(1)
    #     Y_norm = torch.sum(Y * Y, dim=1).unsqueeze(1)

    #     d_XX = X_norm - 2 * torch.mm(X, X.t()) + X_norm.t()
    #     d_YY = Y_norm - 2 * torch.mm(Y, Y.t()) + Y_norm.t()
    #     # print(X_norm.shape, Y_norm.t().shape, torch.mm(X, Y.t()).shape)
    #     d_XY = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()

    #     # K_XX = torch.exp(-0.5*d_XX/(sigma**2)) # next step: remove diagonal, diagonal should be np.ones
    #     # K_YY = torch.exp(-0.5*d_YY/(sigma**2)) # same as above
    #     # K_XY = torch.exp(-0.5*d_XY/(sigma**2)) # no need to remove diagonal because no self-distance
        
    #     K_XX, K_YY, K_XY = torch.zeros(d_XX.shape), torch.zeros(d_YY.shape), torch.zeros(d_XY.shape)
    #     for sigma in sigmas: # different bandwidths as implemented in https://arxiv.org/pdf/2402.03008.pdf
    #         K_XX += torch.exp(-0.5*d_XX/(sigma**2)) # next step: remove diagonal, diagonal should be np.ones
    #         K_YY += torch.exp(-0.5*d_YY/(sigma**2)) # same as above
    #         K_XY += torch.exp(-0.5*d_XY/(sigma**2)) # no need to remove diagonal because no self-distance
    #     K_XX.fill_diagonal_(0.) # remove 1.0s from diagonals (no self-distance)
    #     K_YY.fill_diagonal_(0.) # remove 1.0s from diagonals (no self-distance)
    #     # print(K_XX.shape), print(K_YY.shape), print(K_XY.shape)
    #     print(K_XX), print(K_YY), print(K_XY)

    # else:
    #     raise ValueError("Unsupported kernel type. Supported types are 'linear' and 'gaussian'.")

    # # mmd = torch.mean(K_XX) - 2 * torch.mean(K_XY) + torch.mean(K_YY) # square root this MMD^2

    # mmd = K_XX.sum()/(K_XX.size(0)**2 - len(K_XX.diagonal())) - 2*torch.mean(K_XY) + K_YY.sum()/(K_YY.size(0)**2 - len(K_YY.diagonal())) # square root this MMD^2
    
    # # K_XY.fill_diagonal_(0.)
    # # mmd = K_XX.sum()/(K_XX.size(0)**2 - len(K_XX.diagonal())) - 2 * K_XY.sum()/(K_XY.shape[0]*K_YY.shape[1] - len(K_XY.diagonal())) + K_YY.sum()/(K_YY.size(0)**2 - len(K_YY.diagonal())) # square root this MMD^2

    # return mmd.item()

def maximum_mean_discrepancy(x, y, kernel='rbf'):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Adapted from https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    For more background: https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    x, y = np.asarray(x, dtype=np.double), np.asarray(y, dtype=np.double)

    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), 'needs to be in numpy'

    assert x.ndim == 2 and y.ndim == 2, 'matrices needs to be 2D'
    assert x.shape[1] == y.shape[1], 'number of features need to be the same'

    if x.shape[0] == 1: # if distribution only has 1 datapoint - duplicate it to allow metric to work
        x = np.concatenate([x, x], axis=0)
    if y.shape[0] == 1: # if distribution only has 1 datapoint - duplicate it to allow metric to work
        y = np.concatenate([y, y], axis=0)

    x, y = torch.from_numpy(x), torch.from_numpy(y)

    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    rx_ = (xx.diag().unsqueeze(1).expand_as(xy)) # rx in shape of xy
    ry_ = (yy.diag().unsqueeze(0).expand_as(xy)) # ry in shape of xy

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    # dxy = rx.t() + ry - 2. * xy # Used for C in (1)
    dxy = rx_ + ry_ - 2. * xy # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape),
                  torch.zeros(yy.shape),
                  torch.zeros(xy.shape))

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":
        bandwidth_range = [2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2, 2e3]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    # # include diagonals (of ones) of XX and YY - metric dependent on sample sizes of X and Y
    # return (torch.mean(XX) + torch.mean(YY) - 2. * torch.mean(XY)).item()

    # exclude diagonals (of ones) of XX and YY - metric independent of sample sizes of X and Y
    XX.fill_diagonal_(0.) # remove 1.0s from diagonals (no self-distance)
    YY.fill_diagonal_(0.) # remove 1.0s from diagonals (no self-distance)
    return (torch.sum(XX)/(XX.shape[0]**2 - len(XX.diagonal())) + torch.sum(YY)/(YY.shape[0]**2 - len(YY.diagonal())) - 2. * torch.mean(XY)).item()

ws_distance = SamplesLoss("sinkhorn", blur=0.05,)
def wasserstein_distance(x, y):
    """Emprical maximum mean discrepancy. The lower the result, the more evidence that distributions are the same.

    Adapted from https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    For more background: https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """

    if isinstance(x, pd.DataFrame):
        x = torch.from_numpy(x.to_numpy())
    if isinstance(y, pd.DataFrame):
        y = torch.from_numpy(y.to_numpy())

    if isinstance(x, np.ndarray): 
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): 
        y = torch.from_numpy(y)

    assert x.ndim == 2 and y.ndim == 2, 'matrices needs to be 2D'
    assert x.shape[1] == y.shape[1], 'number of features need to be the same'

    if x.shape[0] == 1: # if distribution only has 1 datapoint - duplicate it to allow metric to work
        x = torch.cat([x, x], axis=0)
    if y.shape[0] == 1: # if distribution only has 1 datapoint - duplicate it to allow metric to work
        y = torch.cat([y, y], axis=0)

    x, y = x.double().contiguous(), y.double().contiguous()
    
    return ws_distance(x, y).item()

if os.path.basename(os.getcwd()) == 'data': 
    os.chdir('..') # switch back to main directory after all

if __name__ == '__main__':
    # _ = calculate_tanimoto_similarity('CCC[N+](CCC)(CCC)CCC', 'CCC[N+](CCC)(CCC)CCCCCC[N+](CCC)(CCC)CCC')
    print(get_zeolite_similarity('CHA', 'AEI', feat_type = 'soap'))