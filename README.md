# DiffSyn: A Generative Diffusion Approach to Materials Synthesis Planning

This is the official repository of

**DiffSyn: A Generative Diffusion Approach to Materials Synthesis Planning (Under Review)**

Elton Pan†, Soonhyoung Kwon‡, Sulin Liu†, Mingrou Xie‡, Alexander Hoffman†, Yifei Duan†, Thorben Prein§, Killian Sheriff†, Yuriy Roman-Leshkov‡, Manuel Moliner¶, Rafael Gomez-Bombarelli†, Elsa Olivetti†*

† MIT Materials Science & Engineering, ‡ MIT Chemical Engineering, § TUM, ¶ ITQ-UPV

A subset of the results is reported in [NeurIPS AI for Materials (Oral Spotlight) paper, 2024](https://openreview.net/forum?id=hy39qxU6CQ). The full manuscript is currently under review and will be made available upon publication.

<p align="center">
<img src="/figs/denoising_diffusion.gif" width="200" />
</p>

![](/figs/denoising_diffusion.png)

<p align="center">
  <a href="/LICENSE">
      <img alt="license" src="https://img.shields.io/badge/license-MIT-green" />
  </a>
</p>

## System requirements
- Python version 3.10.4
- CUDA version >= 11.3

To check your CUDA version, run `nvcc --version`.

**Note:** If your CUDA version is earlier than 11.3, you will have to change <YOUR_CUDA_VERSION> to an earlier version for the following lines in [`env/requirements.txt`](env/requirements.txt):
```
--extra-index-url https://download.pytorch.org/whl/cu<YOUR_CUDA_VERSION>
--find-links     https://data.pyg.org/whl/torch-1.12.1+cu<YOUR_CUDA_VERSION>.html

torch==1.12.1+cu<YOUR_CUDA_VERSION>
torchvision==0.13.1+cu<YOUR_CUDA_VERSION>
torchaudio==0.12.1+cu<YOUR_CUDA_VERSION>
```

All experiments (training, inference and evaluation) are performed on a Rocky Linux machine with a NVIDIA RTX A5000 GPU (24GB RAM).

## A) Installation guide

1. Clone the repo

```bash
git clone https://github.com/eltonpan/zeosyn_gen.git
```

2. Navigate into the repo

```bash
cd zeosyn_gen
```

3. Create environment

```bash
conda create -n zsg python=3.10.4
```

4. Activate environment

```bash
conda activate zsg
```

5. Install dependencies

```bash
pip install -r env/requirements.txt
```

6. Enable conda environment for jupyter notebook

```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=zsg
```

## B) Demo

We demonstrate the DiffSyn model for prediction synthesis recipes for UFI zeolite (Fig. 5 of the manuscript).

### 1. Generating new synthesis routes using DiffSyn for a specific system
Download trained model weights `model.pt` in run folder [`runs/diff/system/run1/`](runs/diff/system/run1)

```
wget -O runs/diff/system/run1/model.pt https://www.dropbox.com/scl/fi/vmf5ag87vszlikmlsnlg4/model.pt?rlkey=9p1d2ht0qxr32of0xizsmqxat&st=obgh0a2n&dl=1
```

To run inference using DiffSyn, run 

```bash
python predict.py
```

Configurations are defined in [`predict.py`](predict.py). Here, we generate synthesis recipes for the UFI zeolite with C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2 as the OSDA. This automatically saves the generated routes at `predictions/UFI_C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2/syn_pred.csv`.
Note: Generating 1000 synthesis routes **takes ~2 min**.

### 2. Visualizing generated synthesis recipes
In the same directory as the above, results can be visualized using [`predictions/UFI_C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2/compare_pred_and_true.ipynb`](predictions/UFI_C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2/compare_pred_and_true.ipynb) (Fig. 5a and 5e of the manuscript).

## C) Model training and evaluation

### 1. Training
The DiffSyn model is trained on the [ZeoSyn dataset](https://pubs.acs.org/doi/10.1021/acscentsci.3c01615), consisting of 23,961 zeolite synthesis routes with 233 zeolite topologies and 921 organic structure-directing agents. 

To train DiffSyn, run 

```bash
python train_diff.py
```

Training configurations are defined in [`train_diff.py`](train_diff.py). For example, the name of the run is defined as `"fname": "run1"` This will automatically create a run directory in [`runs/diff/system/run1`](runs/diff/system/run1). If you run into OOM errors, please lower `batch_size` in the corresponding config files for each task.

Note: Training will takes ~50 hours. We recommend you download the model weights (see instructions under Demo section).

### 2. Evaluation
To evaluate the DiffSyn model, run

```bash
python eval.py
```

This evaluates the trained model via a 2-step process: First, we generate synthesis recipes for every test system and save it at `runs/diff/system/run1/syn_pred_agg-cond_scale_0.75-test.csv`. This takes ~200 min. Second, we compute metrics, Wasserstein distance and MAE, of the model and save them at `runs/diff/system/run1/wsd_zeo_osda.json` and `runs/diff/system/run1/reg_zeo_osda.json`, respectively. 

You should expect to see a mean Wasserstein distance of `Mean WSD: 0.423` (Fig. 2a of manuscript).

### Baselines models
Baseline models can be trained by running the corresponding `train_<MODEL_NAME>.py` and evaluated by changing the configs in [`eval.py`](eval.py).

## Repo tree
```
├── cde
├── data
│   ├── 2024-10-02_K222_and_CHA_OSDA_features.csv
│   ├── 241002_k222_etc_mols_osda_priors_0.pkl
│   ├── augmentation.ipynb
│   ├── cbus-to_be_deleted.csv
│   ├── cifs
│   ├── CVAE_EGNN_embeddings_2023-07-13.csv
│   ├── diffusion_trajectory
│   ├── get_bash_command_for_distance_grid.py
│   ├── get_dummy_graph.py
│   ├── get_zeo_graphs.ipynb
│   ├── gpt4_generated_dict.py
│   ├── iza_codes.py
│   ├── metrics.py
│   ├── osda_descriptors.csv
│   ├── osda_enc_emb.csv
│   ├── prec_rec_vs_hp
│   ├── process_zeosyn.ipynb
│   ├── qt
│   ├── scalers
│   ├── smiles2graph.pkl
│   ├── syn_variables.py
│   ├── utils.py
│   ├── zeo2graph.pkl
│   ├── zeo_enc_emb.csv
│   ├── zeolite_amd_distance_matrix.csv
│   ├── zeolite_binding_energy.csv
│   ├── zeolite_descriptors.csv
│   ├── zeolite_descriptors_for_dendro.csv
│   ├── zeolite_graph_distance.csv
│   ├── zeo_osda_sim-syn_cos_sim.csv
│   ├── zeo_osda_sim-syn_cos_sim-zeo_egnn.csv
│   ├── zeo_osda_sim-syn_mmd_dissim.csv
│   ├── ZEOSYN-2.xlsx
│   ├── ZeoSynGen_dataset.pkl
│   └── ZEOSYN.xlsx
├── env
│   ├── cde.yml
│   ├── requirements.txt
│   ├── zeo_diffusion_metrics_eq.yml
│   ├── zeo_diffusion_metrics.yml
│   └── zeo_diffusion.yml
├── eval.py
├── figs
├── get_diffusion_trajectory.py
├── get_metrics_vs_t.py
├── LICENSE
├── models
│   ├── bnn.py
│   ├── cvae.py
│   ├── diffusion.py
│   ├── gan.py
│   ├── nf.py
│   ├── nn.py
├── notebooks
│   ├── 2024-10-01_OSDAs_to_featurize.ipynb
│   ├── compare_cvae_diff.ipynb
│   ├── compare_cvae_v9_v10.ipynb
│   ├── compare_model_metrics.ipynb
│   ├── compare_model_outputs.ipynb
│   ├── CP-CS1_FAU_LTA.ipynb
│   ├── CS1_MTT_C[N+](C)(C)CCCCCCC[N+](C)(C)C.ipynb
│   ├── CS2_MWW_CCCCCCC[N+](C)(C)C.ipynb
│   ├── CS3_BEC_C[N+](C)(C)CCCCCC[N+](C)(C)C.ipynb
│   ├── CS4_ITG_C[N+]1(C)CCC([N+]2(C)CCCC2)CC1.ipynb
│   ├── CS5_IWR.ipynb
│   ├── CS6_ATO.ipynb
│   ├── dendrogram.ipynb
│   ├── dendrogram_learned_embeds.ipynb
│   ├── get_k222_and_CHA_osdas_features.ipynb
│   ├── get_osda_embeddings.ipynb
│   ├── get_zeolite_embeddings.ipynb
│   ├── metrics_vs_across_hp.ipynb
│   ├── OPT-CS1_CHA.ipynb
│   ├── temp_vs_fwd.ipynb
│   ├── tune_diff_hyperparams.ipynb
│   ├── villaescusa.ipynb
│   ├── visualize_pred_amd.ipynb
│   ├── visualize_pred_bnn.ipynb
│   ├── visualize_pred_cvae-eq.ipynb
│   ├── visualize_pred_cvae-gnn.ipynb
│   ├── visualize_pred_cvae.ipynb
│   ├── visualize_pred_diff.ipynb
│   ├── visualize_pred_gan.ipynb
│   ├── visualize_pred_gmm.ipynb
│   ├── visualize_pred_nf.ipynb
│   ├── visualize_pred_nn.ipynb
│   └── visualize_pred_random.ipynb
├── predictions
│   ├── UFI_C1COCCN2CCOCCOCCN(CCO1)CCOCCOCC2
├── predict.py
├── README.md
├── requirements.txt
├── results
├── runs
│   ├── amd
│   ├── bnn
│   ├── cvae
│   ├── cvae-eq
│   ├── cvae-gnn
│   ├── diff
│   ├── gan
│   ├── gmm
│   ├── nf
│   ├── nn
│   └── random
├── splits
│   └── split_dataset.ipynb
├── train_amd.py
├── train_bnn.py
├── train_cvae-eq.py
├── train_cvae-gnn.py
├── train_cvae.py
├── train_diff.py
├── train_gan.py
├── train_gmm.py
├── train_nf.py
└── train_nn.py
```

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

## Contact
If you have any questions, please contact us at [eltonpan@mit.edu](mailto:eltonpan@mit.edu) or [elsao@mit.edu](mailto:elsao@mit.edu).

## To-do:
- [x] Test conda installation on Linux systems
- [x] Test conda installation on non-Linux system
- [ ] Add link to paper
- [ ] Update Bibtex (after issue is out)
- [ ] Add Colab notebook option