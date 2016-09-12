import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

#apply label encoder for categorical data
def process(x): 
	le = preprocessing.LabelEncoder()
	for col in x:
		if ( x[col].dtypes == 'object'):
			x[col] = le.fit_transform(x[col])
	return x


#read the data
consumer = pd.read_csv('consumerpanel_anon.csv')

#split the data
brand_57_48 = consumer.loc[consumer['brand_anon'] == '573fef48']
brand_b7_38 = consumer.loc[consumer['brand_anon'] == 'b7907338']

#apply encoding to both brands
brandProcessed_57_48 = process(brand_57_48)
brandProcessed_b7_38 = process(brand_b7_38)

y = np.asarray(brand_57_48['value'], dtype="|S6")
#ignore columns
brandProcessed_57_48.drop(['district','main_provider','projectionfactor','value'],axis= 1, inplace = True)


#feature selection
print brand_57_48.shape
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
#from sklearn.svm import LinearSVC
#lsvc = LinearSVC(C=0.1, penalty="l1", dual=False)
#ftp = SelectFromModel(lsvc)
ftp = SelectPercentile(f_classif, percentile=80)
HPfeature_57_48 = ftp.fit_transform(brandProcessed_57_48.values,y)
features_selected = ftp.get_support(indices = False)
HPfeature_57_48.shape

c = []
for i in range(len(features_selected)): c.append(i) if ( features_selected[i] == True) else 0
print brand_57_48.iloc[[0],c]

#spliting data for crossvalidation
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(HPfeature_57_48,y,test_size= 0.4,random_state = 0)

#train model
from sklearn.ensemble.forest import RandomForestRegressor
clf = RandomForestRegressor()
clf.fit(X_train,y_train)

#predict
from sklearn.metrics import mean_squared_error
y_pred = clf.predict(X_test)
y_test = np.array(y_test,np.float)
print mean_squared_error(y_test,y_pred) 





sys.exit()
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
type_57_48 = brand_57_48.plot(x ="date", y = "value", style= 'o' ,title ="573fef48",figsize=(15,15),legend=True, fontsize=12)
fig = type_57_48.get_figure()
fig.savefig('date_VS_values_57_48.jpg')
plt.close()






