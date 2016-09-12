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

y = np.asarray(brand_57_48['value'], dtype="float64")
#ignore columns
brandProcessed_57_48.drop(['district','main_provider','projectionfactor','value'],axis= 1, inplace = True)


#feature selection
print brand_57_48.shape
from sklearn.feature_selection import SelectPercentile,f_regression, SelectFromModel
#from sklearn.svm import LinearSVC
#lsvc = LinearSVC(C=0.1, penalty="l1", dual=False)
#ftp = SelectFromModel(lsvc)
ftp = SelectPercentile(f_regression, percentile=80)
HPfeature_57_48 = ftp.fit_transform(brandProcessed_57_48.values,y)
features_selected = ftp.get_support(indices = False)
HPfeature_57_48.shape

c = []
for i in range(len(features_selected)): c.append(i) if ( features_selected[i] == True) else 0
print ">>>> Slected Features <<<<<<"
print brand_57_48.iloc[[0],c],'\n'

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
print "score", mean_squared_error(y_test,y_pred) 





sys.exit()






