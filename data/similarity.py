from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

from syn_variables import zeo_cols
from sklearn.preprocessing import StandardScaler

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

# Load zeolite descriptors
df_zeo = pd.read_csv('zeolite_descriptors.csv').drop(columns = ['Unnamed: 0'])
df_zeo = df_zeo[['Code']+zeo_cols] # select specific features
codes = df_zeo['Code'] # save codes

scaler = StandardScaler()
df_zeo = pd.DataFrame(scaler.fit_transform(df_zeo.drop(columns=['Code'])), columns=zeo_cols)
df_zeo['Code'] = codes # add back codes

zeo2feat = {}
for zeo in df_zeo['Code']:
    zeo2feat[zeo] = np.array(df_zeo[df_zeo['Code'] == zeo][zeo_cols])

def get_zeolite_similarity(zeo1, zeo2):

    if (zeo1 in zeo2feat.keys()) and (zeo2 in zeo2feat.keys()):
        zeo1 = zeo2feat[zeo1]
        zeo2 = zeo2feat[zeo2]

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
    # print(get_zeolite_similarity('AEI', 'CHA'))
    print(maximum_mean_discrepancy(np.array([[0. ,1.],
                                             [0. ,0.],
                                             [0., 2.]]),
                                    np.array([[0. ,1.],
                                             [1. ,0.]])
                                             )
                                             )