# ML-project
import pandas as pd
import numpy as np
import seaborn as sns # to make boxplot ,histogram and heatmap
import matplotlib.pyplot as plt
from collections import Counter #for analyis most common names
import gc
from sklearn.imput import simpelImputer
data_set=pd.read_csv("/content/aisles.csv")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
                                        #1_data_preprocessing
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


#_4 outliers -----------> i will use boxplot to notice the outliers
 plt.figure(figsize=(10, 2))
 np.random.seed(42)
 
data_set['daily_sales'] = np.random.normal(50, 10, len(data_set))---> add fake cloumn

sns.boxplot(x=data_set['daily_sales'])
Q1 = data_set['daily_sales'].quantile(0.25)
Q3 = data_set['daily_sales'].quantile(0.75)
IQR = Q3 - Q1   
   
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR  
  
clean_data = data_set[(data_set['daily_sales'] >= lower_bound) & (data_set['daily_sales'] <= upper_bound)]

  plt.figure(figsize=(10, 2))----------------->
  sns.boxplot(x=clean_data['daily_sales'])---->
  plt.show()---------------------------------->just to make sure :)


#_5 EDA

 #discover data
  print(data_set.head())
  print(data_set.info())
  print(data_set.describe().T)----> # for easy to read
  
 #discoverd the mod
  all_words = " ".join(data_set['aisle']).split()
  word_counts = Counter(all_words).most_common(10)
  words_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
  plt.figure(figsize=(10, 5))
  sns.barplot(data=words_df, x='Count', y='Word', palette='viridis')
  plt.show()

 # Analysis the Relations
 plt.figure(figsize=(10, 5))
 sns.scatterplot(x=data_set['aisle_id'], y=data_set['daily_sales'])
 plt.show()


 #Histogram
  plt.figure(figsize=(10, 5))
  sns.histplot(data_set['daily_sales'], kde=True, color='blue')
  plt.show()
  
 #heatmap
  numeric_data = data_set.select_dtypes(include=[np.number])
  corr_matrix = numeric_data.corr()
  plt.figure(figsize=(8, 6))
  sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
  plt.show()

                                         
   #_2_







 
