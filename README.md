# GeneSys
Generative Regulatory Modeling for Systematic Development

---
Temporal single-cell transcriptomics enables the reconstruction of dynamic gene expression changes during development, yet its analytical power is often limited by data sparsity, technical noise, and imbalanced cell-type representation across time points. To overcome these challenges, we present GeneSys (Generative Modeling of Developmental System), a generative deep learning model that simulates single-cell transcriptomic landscapes under developmental constraints and informed by prior biological knowledge or user-defined hypotheses. GeneSys integrates a temporal variational autoencoder with a cell-type classifier and requires a lineage blueprint as input, allowing it to model the temporal transitions of transcriptional states with cell-type specificity. Leveraging data from Arabidopsis thaliana roots and mouse embryos, we show that GeneSys learns robust developmental trajectories, generates realistic and representative transcriptomes, and enhances gene prioritization accuracy compared to unregularized scRNA-seq data. By applying gene masking and augmentation, GeneSys reveals interpretable gene expression programs (GEPs) and serves as an in silico platform to test the impact of specific genes or gene sets on developmental outcomes. Additionally, GeneSys computes linear interaction matrices (LIMAs) to infer dynamic regulatory networks and prioritize transcription factors with spatiotemporal resolution. These features enable GeneSys to nominate key genes governing state transitions in a developmental system, supporting both mechanistic insight and hypothesis generation. Together, GeneSys provides a flexible and extensible framework to simulate single-cell data guided by developmental constraints, empowering discovery of regulatory mechanisms from high-dimensional single-cell datasets.

---

The source codes of GeneSys for training and evaluation are under the code folder.

The jupyter notebooks demonstrating how to prepare, train, and evaluate the GeneSys model can be found under jupyter_notebook folder.
