import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
import pdb
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
import pickle

def check_nans(df):
    return f'Number of NaNs: {df.isna().sum()}'

def check_infs(df):
    return f'Number of Infs: {df.isin([np.inf, -np.inf]).sum()}'

def preprocess_gel(df, x, y, quantile, plot = False):
    '''Converts a dataframe column x and column y, and adds a new column with the ratio x/y. Then caps the value of ratio x/y at a certain quantile. Then applies a quantile transform to the ratio x/y. Also saves the quantile transformer to a pickle file.

    Args:
        df (pd.DataFrame): dataframe
        x (str): column name
        y (str): column name
        quantile (float): quantile to cap the ratio x/y at
    Returns:
        df (pd.DataFrame): dataframe with new columns
    '''
    ratio = f"{x}/{y}"
    df[ratio] = df[x]/df[y]

    # Fix NaNs from 0.0/0.0
    idxs = df[df[ratio].isna()].index
    for idx in idxs:
        df.loc[idx, ratio] = 0. # set to 0.0

    # Fix infs
    idxs = df[df[ratio] == np.inf].index # Inf from by non-zero divided by zero
    high_val = np.quantile(df[(df[ratio] != np.inf) & (df[ratio] != 0.)][ratio], quantile)
    print(ratio)
    print('High val:', high_val)
    for idx in idxs:
        df.loc[idx, ratio] = high_val # set to about 400.0

    # Set upper limit
    idxs = df[df[ratio] >= high_val].index # High values
    for idx in idxs:
        df.loc[idx, ratio] = high_val # set to about 400.0

    print(check_nans(df[ratio]))
    print(check_infs(df[ratio]))

    # Quantile transform
    qt = QuantileTransformer(n_quantiles=1000, random_state=0)
    df[f'{ratio}_qt'] = qt.fit_transform(np.array(df[ratio]).reshape(-1, 1)).reshape(-1)
    with open(f'qt/{x}{y}_qt.pkl', 'wb') as f:
        pickle.dump(qt, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'qt/{x}{y}_qt.pkl', 'rb') as f:
        qt = pickle.load(f)

    if plot:
        plt.figure(figsize=(15,7))
        sns.displot(df[ratio], bins=30) 
        sns.displot(df[f'{ratio}_qt'], bins=30) 
        plt.show()

    return df

def preprocess_conditions(df, ratios, conds, plot = False):
    '''Imputes conditions based on gel compositions. Applies quantile transformation to conditions. Also saves the quantile transformer to a pickle file.

    Args:
        df (pd.DataFrame): dataframe
        ratios (dict): ratios
        conds (dict): conditions
    Returns:
        df (pd.DataFrame): dataframe with new columns
    '''

    # Conditions
    syn_cols = list(ratios.keys()) + list(conds.keys()) 
    df_syn = df[syn_cols] # create a temporary dataframe with only the synthesis conditions
    # Impute missing values in conditions
    imp = IterativeImputer(min_value = 0., sample_posterior = True, skip_complete = True, random_state=0) # Sample posterior to get a distribution, skip_complete to speed up, min value is 0. since time and temp are +ve
    df_syn = imp.fit_transform(df_syn)
    df_syn = pd.DataFrame(df_syn, columns = syn_cols, index = df.index)
    df['cryst_temp'] = df_syn['cryst_temp'] # replace the original columns with the imputed columns
    df['cryst_time'] = df_syn['cryst_time'] # replace the original columns with the imputed columns

    for cond in conds.keys():
        print(cond)
        print(check_nans(df[cond]))
        print(check_infs(df[cond]))

        # Quantile transform
        qt = QuantileTransformer(n_quantiles=1000, random_state=0)
        df[f'{cond}_qt'] = qt.fit_transform(np.array(df[cond]).reshape(-1, 1)).reshape(-1)
        with open(f'qt/{cond}_qt.pkl', 'wb') as f:
            pickle.dump(qt, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'qt/{cond}_qt.pkl', 'rb') as f:
            qt = pickle.load(f)

        if plot:
            plt.figure(figsize=(15,7))
            sns.displot(df[cond], bins=30) 
            sns.displot(df[f'{cond}_qt'], bins=30) 
            plt.show()

    return df

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    # adapted from OPIG https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    """
    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    # adapted from OPIG https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles(x_smiles):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    # adapted from OPIG https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-pytorch-geometric/
    """
    
    data_list = []
    
    for smiles in x_smiles:
        if smiles == 'OSDA-free': # If OSDA-free, initialize with zeros
            n_nodes = 2
            n_edges = 2
            n_node_features = 79
            n_edge_features = 10

            X = torch.zeros((n_nodes, n_node_features), dtype = torch.float)
            E = torch.tensor([
                            [0, 1],
                            [1, 0]
                            ])
            EF = torch.zeros((n_edges, n_edge_features), dtype = torch.float)

        else: # If SMILES present
            # convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(smiles)
            # get feature dimensions
            n_nodes = mol.GetNumAtoms()
            n_edges = 2*mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
            # construct node feature matrix X of shape (n_nodes, n_node_features)
            X = np.zeros((n_nodes, n_node_features))
            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = get_atom_features(atom)
                
            X = torch.tensor(X, dtype = torch.float)
            
            # construct edge index array E of shape (2, n_edges)
            (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim = 0)
            
            # construct edge feature array EF of shape (n_edges, n_edge_features)
            EF = np.zeros((n_edges, n_edge_features))
            
            for (k, (i,j)) in enumerate(zip(rows, cols)):
                
                EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
            
            EF = torch.tensor(EF, dtype = torch.float)
            
        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, smiles = smiles))
    return data_list

class ZeoSynGenDataset:
    def __init__(self, 
                 # Gel composition + Reaction conditions
                 x_syn_frac, x_syn_ratio, # molar fraction, molar ratio

                 # Precursors
                 x_prec,

                 # Zeolite code, zeolite features
                 y_zeo1_code, y_zeo1_graph, y_zeo1_feat, y_zeo1_graph_present, y_zeo1_feat_present,
                 y_zeo2_code, y_zeo2_graph, y_zeo2_feat, y_zeo2_graph_present, y_zeo2_feat_present,

                 # OSDA SMILES, OSDA graph, OSDA features
                 y_osda1_smiles, y_osda1_graph, y_osda1_feat, y_osda1_graph_present, y_osda1_feat_present,
                 y_osda2_smiles, y_osda2_graph, y_osda2_feat, y_osda2_graph_present, y_osda2_feat_present,
                 
                #  y_zeo3_code, y_zeo3_graph, y_zeo3_feat, y_zeo3_graph_present, y_zeo3_feat_present,
                #  y_osda3_smiles, y_osda3_graph, y_osda3_feat, y_osda3_graph_present, y_osda3_feat_present,

                 # Metadata
                 year,
                 doi,
                 idxs,
                 is_lit,

                 # Scalers
                 qts, # Dict eg. {'Si/Al': QuantileTransformer()}
                 osda_feat_scaler,
                 zeo_feat_scaler,

                 # Feature names
                 frac_names, ratio_names, cond_names, zeo_feat_names, osda_feat_names 

                 ):
        self.x_syn_frac, self.x_syn_ratio = x_syn_frac, x_syn_ratio
        self.x_prec = x_prec
        self.y_zeo1_code, self.y_zeo1_graph, self.y_zeo1_feat, self.y_zeo1_graph_present, self.y_zeo1_feat_present = y_zeo1_code, y_zeo1_graph, y_zeo1_feat, y_zeo1_graph_present, y_zeo1_feat_present
        self.y_zeo2_code, self.y_zeo2_graph, self.y_zeo2_feat, self.y_zeo2_graph_present, self.y_zeo2_feat_present = y_zeo2_code, y_zeo2_graph, y_zeo2_feat, y_zeo2_graph_present, y_zeo2_feat_present
        self.y_osda1_smiles, self.y_osda1_graph, self.y_osda1_feat, self.y_osda1_graph_present, self.y_osda1_feat_present = y_osda1_smiles, y_osda1_graph, y_osda1_feat, y_osda1_graph_present, y_osda1_feat_present
        self.y_osda2_smiles, self.y_osda2_graph, self.y_osda2_feat, self.y_osda2_graph_present, self.y_osda2_feat_present = y_osda2_smiles, y_osda2_graph, y_osda2_feat, y_osda2_graph_present, y_osda2_feat_present

        # self.y_zeo3_code, self.y_zeo3_graph, self.y_zeo3_feat, self.y_zeo3_graph_present, self.y_zeo3_feat_present = y_zeo3_code, y_zeo3_graph, y_zeo3_feat, y_zeo3_graph_present, y_zeo3_feat_present
        # self.y_osda3_smiles, self.y_osda3_graph, self.y_osda3_feat, self.y_osda3_graph_present, self.y_osda3_feat_present = y_osda3_smiles, y_osda3_graph, y_osda3_feat, y_osda3_graph_present, y_osda3_feat_present

        self.year = year
        self.doi = doi
        self.idxs = idxs
        self.is_lit = is_lit

        self.qts = qts # Dict of quantile transformers for synthesis conditions eg. {'Si/Al': QuantileTransformer()}
        self.osda_feat_scaler = osda_feat_scaler
        self.zeo_feat_scaler = zeo_feat_scaler

        self.frac_names, self.ratio_names, self.cond_names, self.zeo_feat_names, self.osda_feat_names = frac_names, ratio_names, cond_names, zeo_feat_names, osda_feat_names

    def __getitem__(self, idx):
        return self.x_syn_frac[idx], self.x_syn_ratio[idx], \
               self.x_prec[idx], \
               self.y_zeo1_code[idx], self.y_zeo1_graph[idx], self.y_zeo1_feat[idx], self.y_zeo1_graph_present[idx], self.y_zeo1_feat_present[idx], \
               self.y_zeo2_code[idx], self.y_zeo2_graph[idx], self.y_zeo2_feat[idx], self.y_zeo2_graph_present[idx], self.y_zeo2_feat_present[idx], \
               self.y_osda1_smiles[idx], self.y_osda1_graph[idx], self.y_osda1_feat[idx], self.y_osda1_graph_present[idx], self.y_osda1_feat_present[idx], \
               self.y_osda2_smiles[idx], self.y_osda2_graph[idx], self.y_osda2_feat[idx], self.y_osda2_graph_present[idx], self.y_osda2_feat_present[idx], \
               self.year[idx], \
               self.doi[idx], \
               self.idxs[idx], \
               self.is_lit[idx], \
               
               
            #    self.y_zeo3_code[idx], self.y_zeo3_graph[idx], self.y_zeo3_feat[idx], self.y_zeo3_graph_present[idx], self.y_zeo3_feat_present[idx], \
            #    self.y_osda3_smiles[idx], self.y_osda3_graph[idx], self.y_osda3_feat[idx], self.y_osda3_graph_present[idx], self.y_osda3_feat_present[idx], \

    def get_datapoints_by_index(self, dataset_idxs, scaled=True, return_dataframe=False):
        '''
        Args:
            dataset_idxs: List. list of indexes
            scaled: Bool. Whether to scale the features.
            return_dataframe: Bool. Whether to return a dataframe or just tensors

        Returns:
            final_result: Tuple of lists or tensors or Dataframes
        '''

        if len(dataset_idxs) == 1: # only 1 datapoint
            result = self[dataset_idxs[0] : dataset_idxs[0]+1] # +1 to get an extra dimension (to be consistent with multiple datapoint dimensions)
        else: # multiple datapoints
            lists = [[] for _ in range(len(self[0]))] # create list containing lists, with each list containing n_datapoints of datapoints
            for dataset_idx in dataset_idxs:
                datapoint = self[dataset_idx]
                for info_idx, info in enumerate(datapoint):
                    lists[info_idx].append(info)
            result = [torch.stack(info) if type(info[0]) == torch.Tensor else info for info in lists]
        
        final_result = []
        for info in result:
            if type(info) == torch.Tensor:
                n_cols = info.shape[1]
                if n_cols == len(self.frac_names)+2:
                    if return_dataframe:
                        info = pd.DataFrame(info, columns=self.frac_names+self.cond_names)
                elif n_cols == len(self.ratio_names)+2:
                    if scaled == False: # scale back
                        unscaled_info = torch.zeros_like(info)
                        for ratio_idx, ratio in enumerate(self.ratio_names):
                            qt = self.qts[ratio] # load quantile transformer
                            unscaled_info[:,ratio_idx] = torch.tensor(qt.inverse_transform(info[:,ratio_idx].reshape(-1, 1)), dtype=torch.float32).reshape(-1) # transform back
                        for cond_idx, cond in enumerate(self.cond_names):
                            qt = self.qts[cond] # load quantile transformer
                            cond_idx += len(self.ratio_names)
                            unscaled_info[:,cond_idx] = torch.tensor(qt.inverse_transform(info[:,cond_idx].reshape(-1, 1)), dtype=torch.float32).reshape(-1) # transform back

                        info = unscaled_info
                    if return_dataframe:
                        info = pd.DataFrame(info, columns=self.ratio_names+self.cond_names)
                elif n_cols == len(self.zeo_feat_names):
                    if scaled == False: # scale back
                        zeo_feat_scaler = self.zeo_feat_scaler # load standard scaler
                        info = torch.tensor(zeo_feat_scaler.inverse_transform(info), dtype=torch.float32)
                    if return_dataframe:
                        info = pd.DataFrame(info, columns=self.zeo_feat_names)
                elif n_cols == len(self.osda_feat_names):
                    if scaled == False: # scale back
                        osda_feat_scaler = self.osda_feat_scaler
                        info = torch.tensor(osda_feat_scaler.inverse_transform(info), dtype=torch.float32)
                    if return_dataframe:
                        info = pd.DataFrame(info, columns=self.osda_feat_names)
                final_result.append(info)
            else:
                final_result.append(info)

        return final_result


    def __len__(self):
        return len(self.x_syn_frac)
    
    def get_system(self, zeo=None, osda=None, scaled=True, return_dataframe=False):
        '''Get zeolite-OSDA system or just zeolite or just OSDA

        Args:
            zeo: str. Zeolite 3-letter IZA code https://america.iza-structure.org/IZA-SC/ftc_table.php
            osda: str. OSDA SMILES
            scaled: Bool. Whether to scale the features.
            return_dataframe: Bool. Whether to return a dataframe or just tensors
        
        Returns:
            Tuple of lists or tensors or Dataframes for the zeolite-OSDA system
        '''
        assert (zeo is not None) or (osda is not None), 'Must specify at least either zeolite or OSDA.'
        
        if zeo is not None:
            assert zeo in set(self.y_zeo1_code + self.y_zeo2_code), 'Zeolite code not in dataset. Please ensure your use the only the 3-letter IZA code without hyphens or asterisks (https://america.iza-structure.org/IZA-SC/ftc_table.php)'
        if osda is not None:
            assert osda in set(self.y_osda1_smiles + self.y_osda2_smiles), 'OSDA SMILES not in dataset.'

        sys_idxs = [] # indexes of zeo-osda system
        for datapoint_idx in range(len(self)):
            if zeo is not None and osda is not None:
                if (self[datapoint_idx][3] == zeo) and (self[datapoint_idx][13] == osda):
                    sys_idxs.append(datapoint_idx)
            elif zeo is not None:
                if self[datapoint_idx][3] == zeo: # self.y_zeo1_code
                    sys_idxs.append(datapoint_idx)
            elif osda is not None:
                if self[datapoint_idx][13] == osda: # self.y_osda1_smiles
                    sys_idxs.append(datapoint_idx)

        return self.get_datapoints_by_index(sys_idxs, scaled, return_dataframe)
    
    def get_idxs_with_zeo_osda_graphs_feats_present(self):
        '''Get datapoint idxs with both graphs and features present for both zeolites and OSDAs
        
        Returns:
            all_present_idx: List of indices
        '''
    
        all_present_idx = []
        # g: graph, f: feature, p: present
        for idx, zeo_g_p, zeo_f_p, osda_g_p, osda_f_p in zip(self.idxs, self.y_zeo1_graph_present, self.y_zeo1_feat_present, self.y_osda1_graph_present, self.y_osda1_feat_present):
            if np.all(np.array([zeo_g_p, zeo_f_p, osda_g_p, osda_f_p])): # if all graphs and features present for zeolites and OSDAs
                all_present_idx.append(idx)

        self.all_present_idx = all_present_idx

        return self.all_present_idx


    def train_val_test_split(self, mode='random', both_graph_feat_present=True, random_state=0):
        '''Get datapoint idxs for train/val/test sets
        
        Returns:
            random_train_idxs: List of indices
            random_val_idxs: List of indices
            random_test_idxs: List of indices
        '''

        assert mode in ['random', 'system', 'temporal'], 'Mode must be either random, system or temporal.'

        if both_graph_feat_present: # Only use datapoints with both graphs and features present for both zeolites and OSDAs
            idxs = self.get_idxs_with_zeo_osda_graphs_feats_present()
        else: # If not, use all datapoints
            idxs = self.idxs

        if mode == 'random': # Random split
            self.random_train_idxs, self.random_test_idxs = train_test_split(idxs, test_size=0.2, random_state=random_state)
            self.random_train_idxs, self.random_val_idxs = train_test_split(self.random_train_idxs, test_size=0.125, random_state=random_state)
            
            print('train:', len(self.random_train_idxs), 'val:', len(self.random_val_idxs), 'test:', len(self.random_test_idxs))
        
            return self.random_train_idxs, self.random_val_idxs, self.random_test_idxs

def plot_gel_conds(x_syn_ratio, label=None):
    '''
    Args:
        x_syn_ratio: (pd.DataFrame) with columns of synthesis conditions
        label: (str) label for the plot
    '''
    xlims = {'Si/Al': [-5,410],
                'Al/P': [-0.1,1.8],
                'Si/Ge': [-5,100],
                'Si/B': [-10,260],
                'Na/T': [-0.2,2],
                'K/T': [-0.5,5.4],
                'OH/T': [-0.1,2.5],
                'F/T': [-0.05,1.3],
                'H2O/T': [-10,210],
                'sda1/T': [-0.3,7],
                'cryst_temp': [0,250],
                'cryst_time': [0,1000],
                }
    fontsize = 15
    fig = plt.figure(figsize=(6/3*len(x_syn_ratio.columns),3), dpi = 100)
    col_idx = 1
    for col_name in x_syn_ratio.columns:
        ax = fig.add_subplot(1, len(x_syn_ratio.columns), col_idx)
        sns.histplot(x_syn_ratio[col_name], label=label, bins=30, binrange=xlims[col_name])
        
        ax.yaxis.set_tick_params(labelleft=False)
        plt.xlim(*xlims[col_name])
        plt.xlabel(col_name, fontsize=fontsize)
        if col_idx > 1:
            plt.ylabel('')
        else:
            plt.ylabel('Density', fontsize=fontsize)
        col_idx += 1
    plt.legend()
    plt.show()

def compare_gel_conds(x_syn_ratios, labels, kde=False, common_norm=False):
    '''
    Args:
        x_syn_ratios: (List of pd.DataFrames) with columns of synthesis conditions. Each DataFrame should be have shape [n_datapoints, n_gel_comp + n_reaction_cond]
        labels: (List of str) legend labels for the above DataFrame(s). Must be same length as x_syn_ratios
        kde (bool): whether to plot kde on top of histogram
        common_norm: Bool. for two plots to have the same normalized area.
        
    '''
    assert len(x_syn_ratios) == len(labels), 'Number of DataFrames must be equal to number of legend labels.'

    if common_norm:
        stat = 'proportion'
    else:
        stat = 'count'

    fontsize = 15
    fig = plt.figure(figsize=(6/3*len(x_syn_ratios[0].columns),3), dpi = 100)
    col_idx = 1
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    for col_name in x_syn_ratios[0].columns:
        ax = fig.add_subplot(1, len(x_syn_ratios[0].columns), col_idx)
        col_max = max([x_syn_ratio[col_name].max() for x_syn_ratio in x_syn_ratios])
        col_min= min([x_syn_ratio[col_name].min() for x_syn_ratio in x_syn_ratios])

        if kde:
            for x_syn_ratio, label, color in zip(x_syn_ratios, labels, colors):
                sns.histplot(x_syn_ratio[col_name], label=label, kde=kde, kde_kws={'clip':[col_min, col_max], 'cut':100}, bins=20, binrange=[col_min, col_max], color = color, stat=stat)
        else:
            for x_syn_ratio, label, color in zip(x_syn_ratios, labels, colors):
                sns.histplot(x_syn_ratio[col_name], label=label, bins=20, binrange=[col_min, col_max], color = color, stat=stat)
        
        ax.yaxis.set_tick_params(labelleft=False)
        plt.xlabel(col_name, fontsize=fontsize, labelpad=5)
        if col_idx > 1:
            plt.ylabel('')
        else:
            plt.ylabel('Density', fontsize=fontsize)
        
        # Relabel high values to inf in xticks
        high_val = {'Si/Al': 400.000, 'Al/P': 1.717, 'Si/Ge': 98.999, 'Si/B': 250.000} # maps col_name to high value
        plt.xticks(rotation=45)
        if col_name in high_val.keys(): # only if col_name is in high_val
            if (high_val[col_name]+1e-3 >= col_min) and (high_val[col_name]-1e-3 <= col_max): # only if high value is within range
                xticks = list(ax.get_xticks()) # get current xticks
                if high_val[col_name] not in xticks: # only if high value is not in xticks
                    xticks.append(high_val[col_name])
                xticks = sorted(xticks) # sort xticks
                xticks_w_inf = [] 
                for x in xticks:
                    if x == high_val[col_name]:
                        xticks_w_inf.append(r"$\infty$") # inf if xtick = high val
                    elif x >= high_val[col_name]:
                        xticks_w_inf.append('') # empty string if xtick > high val
                    else:
                        xticks_w_inf.append(x)
                xticks_w_inf[0] = '' # ad-hoc: Reduce congestion in xticks
                plt.xticks(xticks, labels=xticks_w_inf)

        col_idx += 1

    plt.legend()
    plt.show()