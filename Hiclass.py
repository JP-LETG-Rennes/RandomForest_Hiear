#MIT License

#Copyright (c) 2024 Liam Loizeau-Woollgar, Julien Pellen, Laurence Hubert-Moy

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import hiclass
import matplotlib.pyplot as plt
import rioxarray as rxr
import xarray as xr
import os
from sklearn.model_selection import StratifiedKFold, KFold
from shapely.geometry import Point
from hiclass import LocalClassifierPerParentNode
from sklearn.ensemble import RandomForestClassifier
from hiclass.metrics import precision, recall, f1
import json


def main_process(df_path,raster_path, raster_path2, outraster, classdicpath, listlabel, geopakg_out, sep, decimal, crs="EPSG:2154"): 
    # define paths

    df_path = df_path

    raster_path = raster_path


    # raster en sortie

    outraster = outraster

    # dictionnaire des classes en .csv
    classdictpath = classdicpath

    # class labels
    ylabel = listlabel

    # J'importe la version du df ou certains labels de classe des 2e et 3e niveaux ont ete supprimés
    df = pd.read_csv(df_path, sep=sep, decimal=decimal)  # ,header = None)

    crs = 'EPSG:2154'
    df['geometry'] = df['geometry'].str[2:-1].str.split(', ').apply(lambda x: (float(x[0]), float(x[1])))
    df['geometry'] = df['geometry'].apply(lambda coord: Point(coord))
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
    gdf.to_file(geopakg_out)

    # Supprimer colonnes inutilisées
    df = df.drop(['X', 'Y', 'geometry'], axis=1)



    im = rxr.open_rasterio(raster_path, masked=True).squeeze()

    # Le SCR a été interprété comme étant EPSG:9001, le changer pour EPSG:2154
    im = im.rio.set_crs(crs, inplace=True)

    # Sauvegarder la version du raster avec le bon SCR
    # im.rio.to_raster(raster_path2)

    # Je masque certaines parties du raster pour m'assurer que le code puisse gérer des Na values
    rasternans = gpd.read_file(geopakg_out)
    im = im.rio.clip(rasternans.geometry, drop=True, invert=True)

    # Enregistrer le masque de valeurs manquantes
    nan_mask = im.isnull()[0]

    # Remplacer les na du raster par des 0
    im = im.fillna(0)

    df = pd.concat([df.iloc[:, 0:3], df[list(im.long_name)]], axis=1)

    df = df.fillna(str('ABS'))

    # Remplacer valeurs absentes par label du niveau supérieur
    for e in range(len(ylabel[1:])):
        for b in range(len(df)):
            if df.iloc[b, e + 1] == "ABS":
                var = df.iloc[b, e]
                df.iloc[b, e + 1] = var

                # retirer classes peu présentes /!\ marche pour 3 niveaux
    ar = np.array(df[ylabel])
    y_agg = []
    for i in range(ar.shape[0]):
        y_agg.append((str(ar[i, 0]) + str(ar[i, 1]) + str(ar[i, 2])))

    dhd = np.unique(y_agg, return_counts=True)
    label_uu = np.ravel(dhd[0])
    count_label = np.ravel(dhd[1])
    l_var = []

    for e, i in zip(label_uu, count_label):
        if i <= 3:
            l_var.append(e)

    ind = []

    # for i in range(len(l_var)) :
    #    ind.append(y_agg.index(l_var[i]))

    # declare for loop
    for indice in range(len(l_var)):
        for itr in range(len(y_agg)):

            # check the condition
            if (y_agg[itr] == l_var[indice]):
                ind.append(itr)

    # for i in range(len(l_var)) :
    #    y_agg = list(filter(lambda a: a != l_var[i], y_agg))

    df = df.drop(index=ind)

    ar = np.array(df[ylabel])
    y_agg = []
    for i in range(ar.shape[0]):
        y_agg.append((str(ar[i, 0]) + str(ar[i, 1]) + str(ar[i, 2])))

    print(len(y_agg))

    y_test = []
    y_pred = []

    X, y = np.array(df.drop(columns=ylabel)), np.asarray(df[ylabel])
    skf = StratifiedKFold(n_splits=3)

    # Define and train classifier



    rf = RandomForestClassifier()

    classifier = LocalClassifierPerParentNode(local_classifier=rf, replace_classifiers=False)

    for train, test in skf.split(X, y_agg):
        # print('train -  {}   |   test -  {}'.format(y[train], y[test]))
        classifier.fit(X[train], y[train])

        pred = classifier.predict(X[test])

        # y_truth.append(y[test])
        # y_pred.append(pred)
        y_test += list(y[test])
        y_pred += list(pred)

    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    classifier.fit(X, y)


    # test classifier

    # y_pred = classifier.predict(X_test)

    precision = hiclass.metrics.precision(y_test, y_pred)
    recall = hiclass.metrics.recall(y_test, y_pred)
    f1 = hiclass.metrics.f1(y_test, y_pred)

    print("Hierarchical Precision : ", precision)
    print("Hierarchical Recall : ", recall)
    print("Hierarchical F1-score : ", f1)

    # Turn the raster into a 1 dimensional array for predictions, saving it's original shape and coords for reassembling later on
    # inclut une manip pour que les données à prédire soient au format (valeurs, bandes) au lieu de (bandes, valeurs)
    shape = im.shape
    coords = im.coords
    dims = im.dims

    im_1d = im.values.reshape(shape[0], shape[1] * shape[2])
    im_1d_np = np.zeros((shape[1] * shape[2], shape[0]))
    for i in range(shape[0]):
        im_1d_np[:, i] = im_1d[i]
    im_1d = im_1d_np
    del (im_1d_np)

    # Perfom predictions
    pred_1d = classifier.predict(im_1d)

    ### Les prédictions sont du type caractères U51, pour les affichers comme raster il faut les convertir en int
    # On génère un dictionnaire pour pouvoir déchiffrer les classes depuis leurs valeurs en int
    strmod = np.unique(pred_1d)
    intmod = np.array(range(len(np.unique(pred_1d))))

    modalities_dict = dict(zip(strmod, intmod))
    # La variable modalities_dict permet de voir les correspondances entre les classes et les valeurs int

    # Dans un nouveau data array on convertit les valeurs str en valeurs int
    pred_1d_int = np.zeros(pred_1d.shape, dtype='int')
    for i in range(pred_1d.shape[0]):
        for j in range(pred_1d.shape[1]):
            pred_1d_int[i, j] = modalities_dict[pred_1d[i, j]]

    # remettre au format (bandes, valeurs)
    pred_np = np.empty((pred_1d_int.shape[1], pred_1d_int.shape[0]), dtype='int')
    for i in range(pred_1d_int.shape[1]):
        pred_np[i] = pred_1d_int[:, i]
    pred_1d_int = pred_np
    del (pred_np)

    # On remet les données sous forme xarray géoréférencée, au format original (band, y, x)
    pred_3d = xr.DataArray(pred_1d_int.reshape(pred_1d_int.shape[0], shape[1], shape[2]),
                           dims=dims,
                           coords={'band': np.array(range(pred_1d_int.shape[0])), 'y': coords['y'], 'x': coords['x'],
                                   'spatial_ref': coords['spatial_ref']})

    return pred_3d

def lecture_config(configuration_path):

    # Récupération des paramètres contenus dans le fichier conf.json
    with open(configuration_path, "r") as fileConf:
        conf = json.load(fileConf)

    df_path = conf["df_path"]
    raster_path = conf["raster_path"]
    raster_path2 = conf['raster_path2']
    outraster = conf['outraster']
    classdicpath = conf['classdicpath']
    listlabel = conf['listlabel']
    geopakg_out = conf['geopakg_out']
    sep = conf['sep']
    decimal = conf['decimal']

    try:
        pred = main_process(df_path,raster_path,raster_path2,outraster,classdicpath,listlabel,geopakg_out,sep,decimal)

    except TypeError:
        raise('Attention une ou plusieurs des variables ne sont pas valides')
        sys.exit(1)

    return pred


if __name__ == '__main__':


    pred = lecture_config("/home/adr2.local/pellen_j/PycharmProjects/pythonProject/configuration_RF_hiearchique.json")

