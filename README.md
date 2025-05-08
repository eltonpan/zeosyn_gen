# DiffSyn: A Generative Diffusion Approach to Materials Synthesis Planning

<p align="center">
<img src="/figs/denoising_diffusion.gif" width="200" />
</p>

![](/figs/denoising_diffusion.png)

## System requirements
- Python version 3.10.4
- CUDA version >= 11.3

To check your CUDA version, run `nvcc --version`.

**Note:** If your CUDA version is earlier than 11.3, you will have to change <YOUR_CUDA_VERSION> to an earlier version for the following lines in `requirements.txt`:
- --extra-index-url https://download.pytorch.org/whl/cu<YOUR_CUDA_VERSION>
- --find-links     https://data.pyg.org/whl/torch-1.12.1+cu<YOUR_CUDA_VERSION>.html
- torch==1.12.1+cu<YOUR_CUDA_VERSION>
- torchvision==0.13.1+cu<YOUR_CUDA_VERSION>
- torchaudio==0.12.1+cu<YOUR_CUDA_VERSION>

All experiments (training, sampling and evaluation) are performed on a NVIDIA RTX A5000 GPU (24GB RAM).

## Installation guide
Create environment

`conda create -n zsg python=3.10.4`

Activate environment

`conda activate zsg`

Install dependencies

`pip install -r requirements.txt`

Make conda environment visible to a jupyter notebook

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=zsg
```

## Demo

We demonstrate the DiffSyn model for prediction synthesis recipes for UFI zeolite (Fig. 5 of the manuscript).

### 1. Generating new synthesis routes using DiffSyn for a specific system
Create run folder called `run1` and download trained model weights

```
mkdir runs/diff/system/run1
cd runs/diff/system/run1
wget...
cd ../../..
```

To run inference using DiffSyn, run `python predict.py`. Configurations are defined in `predict.py`. Here, we generate synthesis recipes for the UFI zeolite with C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2 as the OSDA. This will automatically create a run directory at `predictions/UFI_C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2`.
Note: Generating 1000 synthesis routes takes ~2 min.

### 2. Visualizing generated synthesis recipes
Results can be visualized in `predictions/UFI_C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2/compare_pred_and_true.ipynb`.

## Model training and evaluation

### 1. Training
To train DiffSyn, run `python train_diff.py`. Training configurations are defined in `train_diff.py`. For example, the name of the run is defined as `"fname": "run1"` This will automatically create a run directory in `runs/diff/system/run1`. If you run into OOM errors, please lower `batch_size` in the corresponding config files for each task.

Note: Training will takes ~1 week. We recommend you download the model weights from .....

### 2. Evaluation
To evaluate run `python eval.py`.
This evaluates the trained model via a 2-step process: First, we generate synthesis recipes for every test system and save it at `runs/diff/system/run1/syn_pred_agg-cond_scale_0.75-test.csv`. Second, we compute metrics, Wasserstein distance and MAE, of the model and save them at `runs/diff/system/run1/wsd_zeo_osda.json` and `runs/diff/system/run1/reg_zeo_osda.json`, respectively.

### Baselines models
Baseline models can be trained by running the corresponding `train_<MODEL_NAME>.py` and evaluated by changing the configs in `eval.py`.

## Glossary of all files in the repo

**Main scripts**
- `train_X.py`: Train X model
- `eval.py`: Evaluate model(s)
- `predict.py`: Run synthesis parameter prediction on a specific zeolite-OSDA pair
- `data/get_zeo_graphs.ipynb`: Get zeolite graphs resulting in `data/zeo2graph.pkl`
- `data/process_zeosyn.ipynb`: Preprocess `data/ZEOSYN-2.xlsx` to give `data/ZeoSynGen_dataset.pkl` (dataset object)
- `data/augmentation.ipynb`: Visualize and analyze `data/ZeoSynGen_dataset.pkl` for augmentation purposes
- `get_diffusion_trajectory.py`: Generate and save diffusion trajectory
- `get_metrics_vs_t.py`: Calculate metrics over diffusion trajectory

**Visualizations of results in manuscript**
- `notebooks/visualize_pred_X.ipynb`: Visualizations of model X predictions
- `notebooks/compare_cvae_v9_v10.ipynb`: Qualitative validation of MMD/WSD metrics by comparing CVAE predictions (from models v9 and v10) to grouth truth
- `notebooks/tune_diff_hyperparams.ipynb`: Investigation of the effect cond_drop_prob and cond_scale on diffusion performance
- `notebooks/dendrogram.ipynb`: plot dendrogram of zeolites
- `notebooks/compare_model_outputs.ipynb`: 2D distribution of synthesis parameters visualization
- `notebooks/villaescusa.ipynb`: H2O/T vs. zeolite framework density plot (Villaescusa's Rule)
- `notebooks/temp_vs_fwd.ipynb`: Crystallization temperature distributions vs. zeolite framework density plot 
- `notebooks/2024-10-01_OSDAs_to_featurize.ipynb`: Visualize K222 and CHA OSDAs before featurization
- `notebooks/get_k222_and_CHA_osdas_features.ipynb`: Extract OSDA features from `data/241002_k222_etc_mols_osda_priors_0.pkl` and saves preprocessed features as `data/2024-10-02_K222_and_CHA_OSDA_features.csv`
- `notebooks/metrics_vs_across_hp.ipynb`: Visualize metrics w.r.t. hyperparameters (t, cond_scale, p_uncond)

**Data assets**
- `data/ZEOSYN-2.xlsx`: Cleaned ZeoSyn dataset
- `data/zeo2graph.pkl`: Dict mapping zeolite IZA code to graph
- `data/smiles2graph.pkl`: Dict mapping OSDA SMILES to graph
- `data/zeolite_descriptors.csv`: Zeolite physicochemical descriptors
- `data/zeolite_descriptors_for_dendro.csv`: Zeolite physicochemical descriptors + some extra info (ring sizes, CBUs etc) for dendrogram plotting
- `data/zeolite_binding_energy.csv`: Zeolite binding energies to literature OSDAs
- `data/zeolite_graph_distance.csv`: Zeolite graph and SOAP distances to one another
- `data/CVAE_EGNN_embeddings_2023-07-13.csv`: EGNN embeddings of zeolites obtained from pretraining CVAE-EGNN on synthesis task
- `data/cbus-to_be_deleted.csv`: Zeolite CBUs
- `data/osda_descriptors.csv`: OSDA physicochemical descriptors
- `data/iza_codes.py`: List of IZA codes
- `data/zeolite_amd_distance_matrix.csv`: AMD distance matrix featurization of zeolite topologies as reported in Schwalbe-Koda et al (2023) https://github.com/dskoda/Zeolites-AMD/blob/main/data/iza_dm.csv
- `data/241002_k222_etc_mols_osda_priors_0.pkl`: Features of K222 and CHA OSDAs (from Science paper)
- `data/2024-10-02_K222_and_CHA_OSDA_features.csv`: Preprocessed K222 version of `data/241002_k222_etc_mols_osda_priors_0.pkl`
- `data/get_dummy_graph.py`: Get placeholder graphs for amorphous phases and zeolites with no CIF files
- `data/get_bash_command_for_distance_grid.py`: script to construct bash script for Zeo++ 
- `data/syn_variables.py`: Column names of synthesis parameters
- `data/utils.py`: Helper functions for data preprocessing and visualization

**Archive: Precursor generation (for future work)**
- `prec_rec/prepare_prec_dataset.ipynb`: Prepare precursor generation dataset
- `prec_rec/precusors_raw.py`: Dictionary mapping raw text to elemental identity and common names
- `prec_rec/precusors_clean.py`: Dictionary mapping raw text to elemental identity and common names (cleaned by Soon 2024-05-08)
- `prec_rec/prec_dataset/prec_dataset_X.csv`: Specific split for precursor generation dataset
- `prec_rec/prec_dataset/prec_dataset_X.csv`: Specific split for precursor generation dataset
- `visualize_prec_dataset.ipynb`: Guide on how to access key parts of precursor generation dataset

Note: The `cde` environment is required to run training and inference of Gausssian mixture models (gmm).
