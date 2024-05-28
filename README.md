RandomForest Hiearchique (WETLANDS)
======

Overview
-----
This script can be used as a basic pipeline to perform hierarchical classification on georeferenced raster data using the Local classifier per parent node from the Hiclass package (...) 
and Random Forest classification model from Scikit-learn (...). The pipeline is given both as a .ipynb Jupyter notebook file allowing to follow the commented classification process step by step in an interactive way,
or as a .py script allowing to run it directly from the console along with a .json file used to pass the file paths to read inputs and save outputs. 

About Notebook Version
-------------------------

As inputs, this pipeline takes a multiband raster map in .geoTIFF format (for example) and a dataframe including the hierarchical labels, which are given in separate columns, along with the extracted raster values for each sample. It is important that the columns containing raster values are given in the same order as the raster bands. 
The column names containing the hierarchical labels must be given in a specified variable in the first cell of the notebook.
The script includes data preparation steps : saving the nan values from the original raster dataset as a mask to apply to the classified raster, replacement of missing labels at higher hierarchical levels by the highest existing label at lower levels, deleting samples with very rare class occurrences for accurate evaluation of model.

Validation of model is performed using a 3 fold stratified K-fold cross-validation process. 
Provided evaluation metrics are : Hierarchical precision, recall, and F1-score for global evaluation of model performance;
Weighted precision, recall, and F1-score for evaluation of model performance at each hierarchical level;
Producer’s and User’s accuracy given for each class at each level, along with the confusion matrix and overall accuracy for each level.

Features importance are calculated using SHAP values as recommended in the Hiclass package documentation (...).

The resulting model is saved as a .sav file using pickle (...) for later use.

The classified raster is saved in integer format, along with a .csv class dictionary associating each integer value in the raster to its class as given in the training dataset.

Installation
-------------

For ease of use, we recommend that you first install git.

Pre-requis:

      Git
      Anaconda or miniconda

Windows:

      #For Windows distribution, you can install with this link :
      https://git-scm.com/download/win

Linux (Debian/Ubuntu):

      # You can install git with this command:
      sudo apt install git-all
      
Linux (Fedora):
      
      # You can install git with this command:
      sudo dnf install git-all


...
...



Library Dependency
---------------

```
# Clone the repo
git clone https://github.com/JoLeClown/Rf_Hiearchique.git
cd RF_Hiearchique
conda create --name Hiclass python=3.9
conda activate Hiclass
# Prepare pip
conda install pip
pip install --upgrade pip
# Install requirements
pip install -r requirements_envs.txt

```

Getting Started
---------------
If Use Notebook:

      1. Download build from source.
      2. Activate Conda Environnement Hiclass
      3. Launching jupyter notebook
      4. Lauching .ipynb

If Use Terminal:

      1. Download build from source. 
      2. Activate Conda Environnement Hiclass
      3. Enter variables in configuration file : configuration_RF_hiearchique.json
      4. Launching via terminal Hiclass.py
     

Remerciements 
-------------

Citation
---------
If you use this software package, please cite our paper:

```
@misc{joleclown,
      title={Titre_PROVISOIRE},
      author={METTRE LES AUTEURS},
      year={ANNEE},
      eprint={DOI A METTRE},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```






