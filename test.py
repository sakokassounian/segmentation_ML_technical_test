#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:08:13 2022

@author: sako
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 20:06:44 2022

@author: sako
"""




#%%_____________ 0. Packages and Function _____________

import os
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

def print_info(message):
    print(f"\n[INFO]: {message} \n")



#%% __________ 1.Obtain data __________

#path to the file that is located in the same folder as a notebook
input_folder = "."
filename = 'test_data_covid_attitudes_and_purchase_recency.sav'
df, meta = pyreadstat.read_sav(os.path.join(input_folder, filename))



#%% __________ 2.Explore content __________

#import column informaton from metadata
df_answers = pd.DataFrame(meta.variable_value_labels).T

#put in a dataframe for clear visualization
cols = ['questions']+df_answers.columns.sort_values().to_list()
df_answers['questions'] = [meta.column_names_to_labels[i] for i in df_answers.index]
df_answers = df_answers[cols]


#%% __________ 2.Pre-processing and prepreping  __________

#Extract columns only which are associated to the attitudes and measures
cols = [i for i in df.columns if 'QP07' in i] + [i for i in df.columns if 'CAT05' in i ]
df_covid = df[cols]
summary = df_covid.describe().T.sort_values(by='count')

#visualize a summary of the selection
print_info(f"Summary of selected data: {len(cols)} columns and {df_covid.shape[0]} rows")
print_info(f"Only {df_covid.dropna().shape[0]} are completely not null")
print(summary.to_string())


#create homogenous database
X = df_covid.to_numpy() 


#Standardization    
print_info("Applying Standardization ...")
scaler=StandardScaler()
x_std=scaler.fit_transform(X)


#%%________ 3. Dimension reduction _________



#PCA
print_info("Applying PCA ...")
pca = PCA(n_components=2, random_state=2021)
pca.fit(x_std)
x_pca = pca.transform(x_std)

#visualize PCA
plt.figure(figsize=(10,10))
plt.scatter(x_pca[:,0],x_pca[:,1],cmap='jet')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()


#t-SNE
print_info("Applying t-SNE to reduce dimensions ...")
tsne=TSNE(n_components = 2, random_state=20)
x_tsne = tsne.fit_transform(x_pca)




#%% _________ 4. Clustering and Segmentation 

#DBSCAN
print_info("Applying DBSCAN ...")
clustering = DBSCAN(eps = 1.25, min_samples = 15 ).fit(x_tsne)

# Plot t-SNE
print_info("Visualize ...")
plt.figure(figsize=(10,10))
plt.scatter(x_tsne[:,0],x_tsne[:,1],c=clustering.labels_,cmap='jet', picker=True)    
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar()
plt.show()


# pca t-SNE
print_info("Visualize ...")
plt.figure(figsize=(10,10))
plt.scatter(x_pca[:,0],x_pca[:,1],c=clustering.labels_,cmap='jet', picker=True)    
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar()
plt.show()




#%% ___________ 5. Visualize classes __________

df_covid['labels'] = list(clustering.labels_)


for i in df_covid.labels.unique():
    plt.pcolormesh(df_covid.query(f"labels == {i}"),vmin=0,vmax=10,cmap='jet')
    plt.title(f'class = {i}')
    plt.colorbar()
    plt.show()












    


    
    
    


    

    







 








