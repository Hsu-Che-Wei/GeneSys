# GeneSys
**Gene**rative Modeling of Developmental **Sys**tem

---
Temporal single-cell transcriptomics enables the reconstruction of dynamic gene expression changes during development, yet its analytical power is often limited by data sparsity, technical noise, and imbalanced cell-type representation across time points. To overcome these challenges, we present GeneSys, a generative deep learning model that simulates single-cell transcriptomic landscapes under developmental constraints and informed by prior biological knowledge or user-defined hypotheses. GeneSys integrates a temporal variational autoencoder with a cell-type classifier and requires a lineage blueprint as input, allowing it to model the temporal transitions of transcriptional states with cell-type specificity. Leveraging data from Arabidopsis thaliana roots and mouse embryos, we show that GeneSys learns robust developmental trajectories, generates realistic and representative transcriptomes, and enhances gene prioritization accuracy compared to unregularized scRNA-seq data.

**Our manuscript is available on [bioRxiv](https://doi.org/10.1101/2025.08.20.671385) since August 25th, 2025.**

![Screenshot](images/Image1.png)

![Screenshot](images/Image2.png)

---

## For those comfortable with raw Python code and interested in the intricacies of the development process

The source codes of GeneSys for training and evaluation are under the code folder.

The jupyter notebooks demonstrating how to prepare, train, and evaluate the GeneSys model can be found under jupyter_notebook folder.

---

## Tutorial

### 0. Install GeneSys
```
## Create conda environment (ignore jupyterlab if you don't use jupyter notebooks)
conda create -n genesys -c conda-forge -c anaconda jupyterlab pytorch-gpu python=3.8 -y
conda activate genesys

## Clone the github repo
git clone https://github.com/Hsu-Che-Wei/GeneSys.git

## Install genesys
cd ./GeneSys/genesys
pip install -e .

## To the folder/directory where the data sets are stored
cd ../toy_data 
```

### 1. Prepare your inputs (X)
GeneSys requires these inputs to train:

**a. scRNA-seq data** : 

Filtered cell-by-gene matrix (.mtx), cell barcodes (.txt), gene ids/ feature names (.txt)

**b. Cell annotations** : 

The annotation table (.txt) should include three columns named 'barcode', 'label' and 'time'. 'barcode' for the cell barcodes in the scRNA-seq data, 'label' for the categorical labels (cell types, conditions ... etc), and 'time' for temporal steps (treatment time points, dev stages, time bins), which should be in numeric order starting from 1 (e.g. 1, 2, 3, ... n).
   
**c. Cell lineage blueprint** : 

The cell lineage table (.txt) should include how each trajectory (row) is defined. How many trajectories (rows) are there? How many temporal steps (columns) are there? And how cells should be sampled based on the annotation table for each trajectory (biological knowledge or hypothesis).

**Recommendation**:

We encourage users to provide the training data in the format of [AnnData](https://anndata.readthedocs.io/en/stable/). The annotations should be stored as 'label' and 'time' metadata columns in the anndata.obs. The expression matrix provided in anndata.X will be scaled for training. If the anndata.X provided are raw counts, they will first be log-normalized before scaling.

Example toy data can be found in **toy_data** folder.

### 2. Train GeneSys
There are options for the input data:

**a. [AnnData](https://anndata.readthedocs.io/en/stable/) as the input** : 

```
## RNA counts
genesys --train --raw_counts --anndata ./Root_Atlas_RNA_downsampled_2400_cells.h5ad --bprint ./lineage.txt --epochs 30 --batch_size 128 --verbose

## Normalized and/or corrected values
genesys --train --anndata ./Root_Atlas_SCT_downsampled_2400_cells.h5ad --bprint ./lineage.txt --epochs 30 --batch_size 128 --verbose
```

**b. Raw RNA counts cell-by-gene matrix** : 

Raw RNA counts will be log-normalized and scaled for training.

```
## Provide --anndata with directory to where matrix.mtx barcodes.tsv genes.tsv is stored
genesys --train --raw_counts --anndata ./cell_by_gene_matrix/ --anno ./annotations.txt --bprint ./lineage.txt --epochs 30 --batch_size 128 --verbose 
```

The output includes the trained model (.pth) and the training log (.pdf)

### 3. GeneSys-generated transcriptomes (P)

```
## Anndata
genesys --anndata ./Root_Atlas_RNA_downsampled_2400_cells.h5ad --bprint ./lineage.txt --batch_size 128 --verbose --device "cpu" --save_prefix "Root_Atlas_RNA_downsampled_2400_cells"

## Cell-by-gene matrix
genesys --anndata ./cell_by_gene_matrix/ --anno ./annotations.txt --bprint ./lineage.txt --batch_size 128 --verbose --device "cpu" --save_prefix "Root_Atlas_RNA_downsampled_2400_cells"
```
The output includes the generated data in anndata format (.h5ad).

### 4. Real-world applications

The toy data examples with 2,400 cells shown in sections 2 and 3 are provided only as a sanity check. In real-world applications, a dataset of 2,400 cells is not sufficient to train a meaningful GeneSys model. We recommend using at least 20,000 cells to effectively try out GeneSys.   

```
#Train
genesys --train --raw_counts --anndata ./Root_Atlas_RNA_downsampled_20000_cells.h5ad --bprint ./lineage.txt --epochs 100 --batch_size 512 --verbose

#Generate 
genesys --anndata ./Root_Atlas_RNA_downsampled_20000_cells.h5ad --bprint ./lineage.txt --batch_size 512 --verbose --device "cpu" --save_prefix "Root_Atlas_RNA_downsampled_20000_cells"
```

### 5. Parameters glossary








