# ML-project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from sklearn.imput import simpelImputer
data_set=pd.read_csv("/content/aisles.csv")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#_1     memory optimization
 #Delete unused variables
del data_set_old
gc.collect()    
 #Transferring decimal points from 64 to 32 bits
data_set['float_col'] = data_set['float_col'].astype('float32')  
 #Transforming the correct numbers
data_set['int_col'] = data_set['int_col'].astype('int32')
 #Converting Recurring Texts to Category
data_set['aisle'] = data_set['aisle'].astype('category')
