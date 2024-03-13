# Chemically-Guided Diffusion Denoising Enables Zeolite Synthesis Planning

<p align="center">
<img src="/figs/denoising_diffusion.gif" width="200" />
</p>

![](/figs/denoising_diffusion.png)

**Pipeline**
`data/get_zeo_graphs.ipynb`: Get zeolite graphs resulting in `data/zeo2graph.pkl`
`data/process_zeosyn.ipynb`: Preprocess `data/ZEOSYN-2.xlsx` to give `data/ZeoSynGen_dataset.pkl` (dataset object)
`data/augmentation.ipynb`: Visualize and analyze `data/ZeoSynGen_dataset.pkl` for augmentation purposes
`train_cvae.py`: Train CVAE model
`eval_cvae.py`: Evaluate CVAE model
`visualize_pred.ipynb`: Visualizations of CVAE predictions


**Data**
`data/ZEOSYN-2.xlsx`: Cleaned ZeoSyn dataset
`data/zeo2graph.pkl`: Dict mapping zeolite IZA code to graph
`data/smiles2graph.pkl`: Dict mapping OSDA SMILES to graph
`data/zeolite_descriptors.csv`: Zeolite physicochemical descriptors
`data/zeolite_binding_energy.csv`: Zeolite binding energies to literature OSDAs
`data/zeolite_graph_distance.csv`: Zeolite graph and SOAP distances to one another
`data/CVAE_EGNN_embeddings_2023-07-13.csv`: EGNN embeddings of zeolites obtained from pretraining CVAE-EGNN on synthesis task
`data/cbus-to_be_deleted.csv`: Zeolite CBUs
`data/osda_descriptors.csv`: OSDA physicochemical descriptors
`data/iza_codes.py`: List of IZA codes

**Others**
`data/get_dummy_graph.py`: Get placeholder graphs for amorphous phases and zeolites with no CIF files
`data/get_bash_command_for_distance_grid.py`: script to construct bash script for Zeo++ 
`data/syn_variables.py`: Column names of synthesis parameters
`data/utils.py`: Helper functions for data preprocessing and visualization



