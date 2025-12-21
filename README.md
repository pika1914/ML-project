# ML-project
import pandas as pd
import numpy as np
import seaborn as sns # to make boxplot
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
#_2 data_cleaning
 #first look
  print(data_set.info())
  print(data_set.describe())

 #delet Duplicates
  print(f"Number of duplicate IDs: {data_set.duplicated().sum()}")
  data_set.drop_duplicates(inplace=True)


 #processe Missing Values
  print(data_set.isnull().sum())  #test for missing values
  data_set['column_name'].fillna(data_set['column_name'].mean(), inplace=Tru #replace missing values to mean

 #Standardizing Columns
  data_set.columns = data_set.columns.str.strip().str.lower().str.replace(' ', '_') # makes all names lower case

#_3 joins and merging
 new_data= pd.merge(left_data,right_data,on='aisles-id',how='left')
  #to avoid suffixes problem
   pd.merge(data_1,data_2,on='id',suffixes=('_prod', '_aisle'))


#_4 outliers
   
   
  
  

























 
