import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt



consumer = pd.read_csv('consumerpanel_anon.csv')


brand_57_48 = consumer.loc[consumer['brand_anon'] == '573fef48']
brand_b7_38 = consumer.loc[consumer['brand_anon'] == 'b7907338']
print brand_57_48.dtypes


#PLOT CATEGORY DATA
#netincome_per_hhmember
netincome_per_hhmember_57_48 = brand_57_48["netincome_per_hhmember"].value_counts().plot(kind='bar', title ="573fef48",figsize=(15,22),legend=True, fontsize=12)
fig = netincome_per_hhmember_57_48.get_figure()
fig.savefig('netincome_per_hhmember_57_48.jpg')
plt.close()

netincome_per_hhmember_b7_38 = brand_b7_38["netincome_per_hhmember"].value_counts().plot(kind='bar', title ="b7907338",figsize=(15,22),legend=True, fontsize=12)
fig = netincome_per_hhmember_b7_38.get_figure()
fig.savefig('netincome_per_hhmember_b7_38.jpg')
plt.close()

#state
state_57_48 = brand_57_48["state"].value_counts().plot(kind='bar', title ="573fef48",figsize=(15,22),legend=True, fontsize=12)
fig = state_57_48.get_figure()
fig.savefig('state_57_48.jpg')
plt.close()

state_b7_38 = brand_b7_38["state"].value_counts().plot(kind='bar', title ="b7907338",figsize=(15,22),legend=True, fontsize=12)
fig = state_b7_38.get_figure()
fig.savefig('state_b7_38.jpg')
plt.close()

#type
type_57_48 = brand_57_48["type"].value_counts().plot(kind='bar', title ="573fef48",figsize=(15,22),legend=True, fontsize=12)
fig = type_57_48.get_figure()
fig.savefig('type_57_48.jpg')
plt.close()

type_b7_38 = brand_b7_38["type"].value_counts().plot(kind='bar', title ="b7907338",figsize=(15,22),legend=True, fontsize=12)
fig = type_b7_38.get_figure()
fig.savefig('type_b7_38.jpg')
plt.close()

#NUMERICS VALUES
type_57_48 = brand_57_48.plot(x ="date", y = "amount", style= 'o' ,title ="573fef48",figsize=(15,15),legend=True, fontsize=12)
fig = type_57_48.get_figure()
fig.savefig('date_VS_amount_57_48.jpg')
plt.close()






