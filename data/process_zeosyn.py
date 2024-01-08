import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Functions and variables for preprocessing the data
het_cols = ['Si','Al','P','Ge','B','V','Be','Ga','Ti'] # heteroatoms in the dataset
ratios = {'Si/Al': {'quantile': 0.98, 'name': 'Si/Al'}, # gel ratios in dataset. quantiles are used to cap the values at a certain value.
          'Al/P': {'quantile': 0.99, 'name': 'Al/P'},
          'Si/Ge': {'quantile': 0.995, 'name': 'Si/Ge'},
          'Si/B': {'quantile': 0.99, 'name': 'Si/B'},
          'Na/T': {'quantile': 0.95, 'name': 'Na/T'},
          'K/T': {'quantile': 0.99, 'name': 'K/T'},
          'OH/T': {'quantile': 0.97, 'name': 'OH/T'},
          'F/T': {'quantile': 0.95, 'name': 'F/T'},
          'H2O/T': {'quantile': 0.99, 'name': 'H$_2$O/T'},
          'sda1/T': {'quantile': 0.98, 'name': 'SDA/T'},
          }

conds = {'cryst_temp': {'name': 'Cryst. temp. ($^\circ$C)'}, # synthesis conditions in dataset. quantiles are used to cap the values at a certain value.
         'cryst_time': {'name': 'Cryst. time ($^\circ$C)'},
         }

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
    idxs = df[df[ratio] == np.inf].index # NaNs from 0.0/0.0
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

def preprocess_conditions(df, plot = False):
    '''Imputes conditions based on gel compositions. Applies quantile transformation to conditions. Also saves the quantile transformer to a pickle file.

    Args:
        df (pd.DataFrame): dataframe
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

if __name__ == "__main__":
    # Load data
    df = pd.read_excel('ZEOSYN.xlsx').drop(columns = ['Unnamed: 0'])
    df = df[~df['doi'].isna()] # keep only non-empty rows
    df['T'] = df[het_cols].sum(axis=1) # add sum of heteroatoms

    # Preprocess gel compositions
    # Convert to ratios, cap ratio to a certain quantile of the ratio, then apply a quantile transform to the ratio
    for ratio, config in ratios.items():
        x, y = ratio.split('/')
        df = preprocess_gel(df=df, x=x, y=y, quantile=config['quantile'])

    # Preprocess conditions
    # Impute missing values in conditions, then apply a quantile transform to the condition
    df = preprocess_conditions(df, plot = False)
    df.to_csv('ZEOSYN_preprocessed.csv')

