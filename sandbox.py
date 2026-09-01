#%%
import pandas as pd
import numpy as np

#%%
csv = pd.read_csv("/storage/SSD1/.data/milk10k/metadata.csv",index_col="isic_id")
csv = csv.iloc[np.where(csv["diagnosis_1"]!="Indeterminate")[0],:]
slice = np.where(csv["image_type"]!="dermoscopic")[0]
clinic = csv.iloc[slice,:].sort_values(by="lesion_id")
slice = np.where(csv["image_type"]=="dermoscopic")[0]
dermoscopic = csv.iloc[slice,:].sort_values(by="lesion_id")
clinic = clinic.loc[:,["lesion_id","image_type","diagnosis_1"]]
dermoscopic = dermoscopic.loc[:,["lesion_id","image_type","diagnosis_1"]]