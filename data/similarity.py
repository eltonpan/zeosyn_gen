from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

if __name__ == '__main__':
    # _ = calculate_tanimoto_similarity('CCC[N+](CCC)(CCC)CCC', 'CCC[N+](CCC)(CCC)CCCCCC[N+](CCC)(CCC)CCC')
    print(get_zeolite_similarity('AEI', 'CHA'))