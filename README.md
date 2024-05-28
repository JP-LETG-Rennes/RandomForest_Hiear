RandomForest Hiearchique (WETLANDS)
======

Overview
-----
This script can be used as a basic pipeline to perform hierarchical classification on georeferenced raster data using the Local classifier per parent node from the Hiclass package (Miranda, F.M., Köehnecke, N. and Renard, B.Y. (2023) 'HiClass: a Python Library for Local Hierarchical Classification Compatible with Scikit-learn', Journal of Machine Learning Research, 24(29), pp. 1–17. Available at: https://jmlr.org/papers/v24/21-1518.html.) 
and Random Forest classification model from Scikit-learn (Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.). The pipeline is given both as a .ipynb Jupyter notebook file allowing to follow the commented classification process step by step in an interactive way,
or as a .py script allowing to run it directly from the console along with a .json file used to pass the file paths to read inputs and save outputs. 

About
-------------------------

As inputs, this pipeline takes a multiband raster map in .geoTIFF format (for example) and a dataframe including the hierarchical labels, which are given in separate columns, along with the extracted raster values for each sample. It is important that the columns containing raster values are given in the same order as the raster bands. 
The column names containing the hierarchical labels must be given in a specified variable in the first cell of the notebook.
The script includes data preparation steps : saving the nan values from the original raster dataset as a mask to apply to the classified raster, replacement of missing labels at higher hierarchical levels by the highest existing label at lower levels, deleting samples with very rare class occurrences for accurate evaluation of model.

Validation of model is performed using a 3 fold stratified K-fold cross-validation process. 
Provided evaluation metrics are : Hierarchical precision, recall, and F1-score for global evaluation of model performance;
Weighted precision, recall, and F1-score for evaluation of model performance at each hierarchical level;
Producer’s and User’s accuracy given for each class at each level, along with the confusion matrix and overall accuracy for each level.

Features importance are calculated using SHAP values as recommended in the Hiclass package documentation (Lundberg, Scott M and Lee, Su-In, A Unified Approach to Interpreting Model Predictions, Advances in Neural Information Processing Systems, 30, p 4765-4774, Curran Associates Inc., 2017, available at : http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) 

The resulting model is saved as a .sav file using pickle (Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.) for later use.

The classified raster is saved as a .GeoTIFF in integer format, along with a .csv class dictionary associating each integer value in the raster to its class as given in the training dataset.

The predictive data used for our test case consists of reduced time-series of optical remote sensing images provided by ESA Sentinel-2, as well as topographical variables derived from the 5m resolution RGE ALTI® DEM provided by the french national institute for geographical information (IGN). 
The training data consisted of spatial points labeled using the hierarchical EUNIS typology, provided by (...) 

Installation
-------------

For ease of use, we recommend that you first install Git and Anaconda.

Pre-requis:

      Git
      Anaconda or miniconda

Git Windows:

      #For Windows distribution, you can install with this link :
      https://git-scm.com/download/win
We recommmend to choose the Standalone Installer

Git Linux (Debian/Ubuntu):

      # You can install git with this command:
      sudo apt install git-all
      
Git Linux (Fedora):
      
      # You can install git with this command:
      sudo dnf install git-all

Anaconda: 

      # You can download and install anaconda with this link :
      https://www.anaconda.com/download/success

Library Dependency
---------------
Once the prerequisites have been installed, you can launch the next section via Anaconda Prompt

```
# Clone the repo
git clone https://github.com/JoLeClown/Rf_Hiearchique.git
cd RF_Hiearchique

# Create a new python environnement with conda  
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
If Using Notebook:

      1. Download build from source by following the previous steps in the conda prompt terminal.
      2. Activate Conda Environnement Hiclass with following command : activate Hiclass
      3. Launching jupyter notebook from conda terminal with following command : jupyter notebook
      4. Lauching via terminal Hiclass.ipynb 

If Using Terminal:

      1. Download build from source. 
      2. Activate Conda Environnement Hiclass
      3. Enter variables in configuration file : configuration_RF_hiearchique.json
      4. Launching via terminal Hiclass.py

     
Simplified usage with notebook : 
---------------
For those unfamiliar with the use of git, it is possible to avoid it by typing the following in the conda prompt :  

```
conda create --name Hiclass python=3.8 --yes 

activate Hiclass

pip install hiclass numpy pandas geopandas matplotlib rasterio scipy scikit-learn pyproj scipy notebook seaborn xarray rioxarray shap 

jupyter notebook

```

Typing this in conda terminal will create a new environment named "Hiclass", activate it, install all necessary dependencies and open jupyter notebook. All that is left to do is download the .ipynb file, navigate to it through the jupyter interface and open it. 

Acknowledgements 
-------------
Packages : 

      - Hiclass - (Miranda, F.M., Köehnecke, N. and Renard, B.Y. (2023) 'HiClass: a Python Library for Local Hierarchical Classification Compatible with Scikit-learn', Journal of Machine Learning Research, 24(29), pp. 1–17. Available at: https://jmlr.org/papers/v24/21-1518.html.)

      - Scikit-learn - (Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.)

      - SHAP - (Lundberg, Scott M and Lee, Su-In, A Unified Approach to Interpreting Model Predictions, Advances in Neural Information Processing Systems, 30, p 4765-4774, Curran Associates Inc., 2017, available at : http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf)

Modeling data used in test example (visible in .ipynb file) : 

      - ESA Sentinel-2
      - IGN RGE ALTI® 5m DEM

Citation
---------
If you use this script, please cite our paper:

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






