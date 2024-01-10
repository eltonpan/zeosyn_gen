from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
import matplotlib.pyplot as plt

def calculate_tanimoto_similarity(molecule_smiles1, molecule_smiles2, plot=False, verbose=False):
    '''Calculate Tanimoto similarity between two molecules.'''

    # Create RDKit molecules from SMILES strings
    mol1 = Chem.MolFromSmiles(molecule_smiles1)
    mol2 = Chem.MolFromSmiles(molecule_smiles2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string(s)")

    # Generate Morgan fingerprints with radius 2 (you can adjust the radius if needed)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)

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

if __name__ == '__main__':
    _ = calculate_tanimoto_similarity('CCC[N+](CCC)(CCC)CCC', 'CCC[N+](CCC)(CCC)CCCCCC[N+](CCC)(CCC)CCC')