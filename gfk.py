import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import preprocessing


consumer = pd.read_csv('consumerpanel_anon.csv')


print consumer.shape
print consumer.columns