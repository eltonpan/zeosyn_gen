from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from syn_variables import zeo_cols
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
import pdb

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


def visualize_smiles(smiles_list):
    '''Visualize SMILES strings of molecules.
    
    Args:
        smiles_list. List of SMILES strings.
    '''

    mol_list = [Chem.MolFromSmiles(s) for s in smiles_list]

    plt.figure(dpi = 100)
    plt.imshow(Draw.MolsToImage(mol_list))
    plt.axis('off')
    plt.show()

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




def maximum_mean_discrepancy(X, Y, kernel_type='gaussian', gamma=None):
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two distributions.

    Parameters:
    - X: Samples from the first distribution (numpy array or PyTorch tensor).
    - Y: Samples from the second distribution (numpy array or PyTorch tensor).
    - kernel_type: Type of kernel function ('linear', 'gaussian', etc.).
    - gamma: Parameter for the Gaussian kernel (if kernel_type is 'gaussian').

    Returns:
    - mmd: Maximum Mean Discrepancy between X and Y.
    """
    X, Y = np.asarray(X), np.asarray(Y)

    if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)

    if not torch.is_tensor(X):
        raise ValueError("Input X should be a numpy array or a PyTorch tensor.")

    if not torch.is_tensor(Y):
        raise ValueError("Input Y should be a numpy array or a PyTorch tensor.")

    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("Input tensors should be 2-dimensional.")

    if X.size(1) != Y.size(1):
        raise ValueError("Input tensors should have the same number of features.")

    if kernel_type == 'linear':
        K_XX = torch.mm(X, X.t())
        K_YY = torch.mm(Y, Y.t())
        K_XY = torch.mm(X, Y.t())

    elif kernel_type == 'gaussian':
        if gamma is None:
            gamma = 1.0 / X.size(1)  # Default value for gamma

        X_norm = torch.sum(X * X, dim=1).unsqueeze(1)
        Y_norm = torch.sum(Y * Y, dim=1).unsqueeze(1)

        K_XX = torch.exp(-gamma * (X_norm - 2 * torch.mm(X, X.t()) + X_norm.t()))
        K_YY = torch.exp(-gamma * (Y_norm - 2 * torch.mm(Y, Y.t()) + Y_norm.t()))
        K_XY = torch.exp(-gamma * (X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()))
    else:
        raise ValueError("Unsupported kernel type. Supported types are 'linear' and 'gaussian'.")

    mmd = torch.mean(K_XX) - 2 * torch.mean(K_XY) + torch.mean(K_YY)
    
    return mmd.item()

if __name__ == '__main__':
    # _ = calculate_tanimoto_similarity('CCC[N+](CCC)(CCC)CCC', 'CCC[N+](CCC)(CCC)CCCCCC[N+](CCC)(CCC)CCC')
    print(get_zeolite_similarity('CHA', 'AEI', feat_type = 'soap'))
    # print(maximum_mean_discrepancy(np.array([[0. ,1.],
    #                                          [0. ,0.],
    #                                          [0., 2.]]),
    #                                 np.array([[0. ,1.],
    #                                          [1. ,0.]])
    #                                          )
                                            #  )