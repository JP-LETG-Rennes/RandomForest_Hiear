Hierarchical Random Forest classification on raster data 
======

Overview
-----
This script can be used to perform hierarchical classification of  raster data using the local classifier per parent node algorithm in Hiclass package (Miranda et al., 2023) and the random forest classification algorithm in Scikit-learn package (Pedregosa et al., 2011).

The pipeline is designed to work either with the Jupyter notebook to perform the classification interactively step by step using a .ipynb file, or directly from the command line interface using a .py python script file along with a .json text file used to indicate file paths for modeling inputs and outputs. 

About
-------------------------

The pipeline takes as inputs a multiband raster map in .geoTIFF format (or other GDAL-compatible formats) and a dataframe including the hierarchical labels, which are given in separate columns, with the spectral values of samples. It is important that columns with raster values are given in the same order as the raster bands. 
The names of the columns containing the hierarchical labels must be indicated as a list, along with the other modeling inputs information. 

The script includes three data preparation steps: 
    • saving the “nan” values of the original raster dataset as a mask to be applied to the classified raster dataset, 
    • replacing missing labels at higher hierarchical levels by the highest existing label at lower levels, 
    • removing samples with very rare class occurrence. 

The model validation is performed using a k-fold stratified cross-validation process. 
Calculated metrics are: 
    • Hierarchical precision, recall, and F1-score for global evaluation of model performance;
    • Weighted precision, recall, and F1-score for evaluation of model performance at each hierarchical level;
    • Producer’s and User’s accuracy given for each class at each hierarchical level provided with the confusion matrix and overall accuracy at each hierarchical level.

Features importance is calculated using SHAP values as recommended in the Hiclass package documentation (https://hiclass.readthedocs.io/en/latest/index.html).

The resulting model is saved as a .sav file using the pickle package from python’s native library for later use.

The classified raster is saved in integer format, along with a .csv class dictionary associating each integer value in the raster to its class as given in the training dataset.

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
git clone https://github.com/JP-LETG-Rennes/RandomForest_Hiear.git
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

Example dataset
---------------

The 5 X 5 km site study site is located along the Gironde estuary (France) in the Calupeyre catchment area (45.47°N, 1.08°W). The site comprises coastal dunes in the western part, a marsh in the central part and limestone hillsides in the eastern part.

The dataset consists of:
- a raster file containing the predictive variables; 
- a table giving the values of each predictive variable and the vegetation class for 1,629 field samples collected over the whole watershed.

The raster Calupeyre_pred_variables.tif was projected in the French projection system (Lambert 93, code EPSG 2154) at 10  10 m spatial resolution. It includes 6 bands characterizing the micro-topography (Panhelleux et al., 2023) - that were derived from an airborne digital terrain model (RGE ALTI ® IGN) -  as well as 27 bands characterizing the first three components of the functional principal component analysis for each of the 9 spectral bands (band 2, 3, 4, 5, 6, 7, 8, 10 & 11) derived from a pluriannual (2015-2021) multispectral satellite Sentinel-2 time series (® ESA). The method used to generate the functional principal component analysis components was based on Pesaresi et al (2022).

The table "Calupeyre_extracted_raster_values.csv" is in csv format (comma separator). It contains for each sample (row) the value of each predictor variable (column) as well as the corresponding natural habitat class in the European EUNIS typology (Davies et al., 2004) for each of the three hierarchical levels (level1, level2, level3). The vegetation samples were collected in the field and archived in various open access databases:
- The French national inventory of natural heritage (Poncet, 2013) accessed on the website of the national natural history museum (https://inpn.mnhn.fr/accueil/index?lg=en), 
- the French national forest inventory (Hervé, 2016), accessed from the IGN website (https://inventaire-forestier.ign.fr/dataifn/), 
- samples collected as part of wetland inventories (Gayet et al., 2022).

Each sample was automatically assigned to level 3 of the EUNIS typology based on their floristic composition and environmental characteristics according to the approach detailed in Chytrý et al. (2021). In addition, crop samples taken from the European land parcel information system (LPIS) - available on the IGN ® website (https://geoservices.ign.fr/rpg) - were assigned to the Intensive unmixed crops habitat (EUNIS code I1.1). It should be noted that, for the purposes of this tutorial and for genericity, the characterization of some samples was voluntary degraded to level 1 or 2 of the EUNIS typology.


References 
-------------
      Chytrý, M., Tichý, L., Hennekens, S. M., Knollová, I., Janssen, J. A. M., Rodwell, J. S., Peterka, T., Marcenò, C., Landucci, F., Danihelka, J., Hájek, M., Dengler, J., Novák, P., Zukal, D., Jiménez-Alfaro, B., Mucina, L., Abdulhak, S., Abramova, L., Aćić, S., … Schaminée, J. H. J. (2021, 1 juin). EUNIS-ESy: Expert system for automatic classification of European vegetation plots to EUNIS habitats. Zenodo. https://doi.org/10.5281/zenodo.4812736
      
      Davies, C. E., Moss, D. et Hill, M. O. (2004). EUNIS habitat classification revised 2004. Report to: European Environment Agency-European Topic Centre on Nature Protection and Biodiversity, 127‑143.
      
      Gayet, G., Botcazou, F., Gibeault-Rousseau, J.-M., Hubert-Moy, L., Rapinel, S. et Lemercier, B. (2022). Field dataset of punctual observations of soil properties and vegetation types distributed along soil moisture gradients in France. Data in Brief, 45, 108632. https://doi.org/10.1016/j.dib.2022.108632
      
      Hervé, J.-C. (2016). France. Dans C. Vidal, I. A. Alberdi, L. Hernández Mateo et J. J. Redmond (dir.), National Forest Inventories: Assessment of Wood Availability and Use (p. 385‑404). Springer International Publishing. https://doi.org/10.1007/978-3-319-44015-6_20
      
      Miranda, F. M., Köhnecke, N. et Renard, B. Y. (2023). Hiclass: a python library for local hierarchical classification compatible with scikit-learn. Journal of Machine Learning Research, 24(29), 1‑17.
      
      Panhelleux, L., Rapinel, S., Lemercier, B., Gayet, G. et Hubert-Moy, L. (2023). A 5 m dataset of digital terrain model derivatives across mainland France. Data in Brief, 109369. https://doi.org/10.1016/j.dib.2023.109369
      
      Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. et Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12(Oct), 2825‑2830.
      
      Pesaresi, S., Mancini, A., Quattrini, G. et Casavecchia, S. (2022). Functional analysis for habitat mapping in a special area of conservation using sentinel-2 time-series data. Remote Sensing, 14(5), 1179.
      
      Poncet, L. (2013). La diffusion de l’information sur la biodiversité en France. L’exemple de l’inventaire national du patrimoine naturel (INPN). Netcom. Réseaux, communication et territoires, (27‑1/2), 181‑189.

Citation
---------
If you use this script, please cite our paper:

Liam Loizeau-Woollgar, Julien Pellen, Laurence Hubert-Moy. Hierarchical Random Forest classification on raster data. 2024. hal-04596517

```
@misc{JP-LETG-Rennes,
      title={Hierarchical Random Forest classification on raster data},
      author={Liam Loizeau-Woollgar, Julien Pellen, Laurence Hubert-Moy},
      year={2024},
      eprint={hal-04596517},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```






