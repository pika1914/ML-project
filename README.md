# ML-project
import pandas as pd
import numpy as np
import seaborn as sns # to make boxplot ,histogram and heatmap
import matplotlib.pyplot as plt
from collections import Counter #for analyis most common names
import gc
from sklearn.impute import SimpleImputer
data_set=pd.read_csv("/content/aisles.csv")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



#fake data
data_set['user_id'] = np.random.randint(1, 100, len(data_set))
data_set['order_id'] = range(len(data_set))
data_set['order_dow'] = np.random.randint(0, 7, len(data_set))
data_set['order_hour_of_day'] = np.random.randint(0, 24, len(data_set))
data_set['days_since_prior_order'] = np.random.randint(0, 30, len(data_set))
data_set['reordered'] = np.random.randint(0, 2, len(data_set))
  #1_data_preprocessing
#_1     memory optimization
 #Delete unused variables
if 'data_set_old' in locals(): del data_set_old
gc.collect()    
 #Transferring decimal points from 64 to 32 bits
#data_set['float_col'] = data_set['float_col'].astype('float32')  
 #Transforming the correct numbers
#data_set['int_col'] = data_set['int_col'].astype('int32')
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
data_set['aisle'].fillna(data_set['aisle'].mode()[0], inplace=True) #replace missing values to mode

 #Standardizing Columns
data_set.columns = data_set.columns.str.strip().str.lower().str.replace(' ', '_') # makes all names lower case

#_3 joins



#_4 outliers -----------> i will use boxplot to notice the outliers
plt.figure(figsize=(10, 2))
np.random.seed(42)
 
data_set['daily_sales'] = np.random.normal(50, 10, len(data_set)) # add fake cloumn

sns.boxplot(x=data_set['daily_sales'])
Q1 = data_set['daily_sales'].quantile(0.25)
Q3 = data_set['daily_sales'].quantile(0.75)
IQR = Q3 - Q1   
   
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR  
  
clean_data = data_set[(data_set['daily_sales'] >= lower_bound) & (data_set['daily_sales'] <= upper_bound)]

plt.figure(figsize=(10, 2))#----------------->
sns.boxplot(x=clean_data['daily_sales'])#---->
plt.show()#---------------------------------->just to make sure :)


#_5 EDA

 #discover data
print(data_set.head())
print(data_set.info())
print(data_set.describe().T)#---->  for easy to read
  
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

                                         
   #_2_Feature engineering 


#_1 user_level 
 
user_stats = data_set.groupby('user_id')['days_since_prior_order'].mean().reset_index()
user_stats.columns = ['user_id', 'user_avg_days']
 
data_set = data_set.merge(user_stats, on='user_id', how='left')
 
#_2  Product_Level
 
aisle_stats = data_set.groupby('aisle_id')['order_id'].count().reset_index()
aisle_stats.columns = ['aisle_id', 'aisle_popularity']
 
data_set = data_set.merge(aisle_stats, on='aisle_id', how='left')

#_3 user x product

interaction = data_set.groupby(['user_id', 'aisle_id']).size().reset_index()
interaction.columns = ['user_id', 'aisle_id', 'user_bought_aisle_times']
 
data_set = data_set.merge(interaction, on=['user_id', 'aisle_id'], how='left')

#_4 temporal_features
data_set['is_weekend'] = data_set['order_dow'].isin([0, 1]).astype(int) #------->To understand people behavior in weekend
def get_time_of_day(hour):
    if 6 <= hour < 12:
      return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'
data_set['part_of_day'] = data_set['order_hour_of_day'].apply(get_time_of_day)
data_set['hour_sin'] = np.sin(2 * np.pi * data_set['order_hour_of_day'] / 24)
data_set['hour_cos'] = np.cos(2 * np.pi * data_set['order_hour_of_day'] / 24)

print(data_set[['order_dow', 'order_hour_of_day', 'is_weekend', 'part_of_day', 'hour_sin', 'hour_cos']].head())

#_5 categorical_encoding
data = {
  'aisle': ['fresh fruits','fresh fruits','packaged cheese','fresh fruits','water','packaged cheese','water','fresh fruits', 'yogurt', 'fresh fruits'] }
 
dummy_df = pd.DataFrame(data)

freq_map = data_set['aisle'].value_counts(normalize=True)
data_set['aisle_freq_enc'] = data_set['aisle'].map(freq_map)

np.random.seed(42)
data_set['is_reordered'] = np.random.randint(0, 2, len(data_set))

target_map = data_set.groupby('aisle', observed=True)['is_reordered'].mean()  #----->calculat the mean
data_set['aisle_target_enc'] = data_set['aisle'].map(target_map)

print(data_set.sort_values(by='aisle'))

#_3_Classification


 #Task_A
user_orders_count = data_set.groupby('user_id')['order_id'].count().reset_index()
user_orders_count.columns = ['user_id', 'user_total_orders']
if 'user_total_orders' not in data_set.columns:
    data_set = data_set.merge(user_orders_count, on='user_id', how='left')
data_set = data_set.dropna(subset=['aisle_freq_enc', 'order_dow', 'days_since_prior_order', 'user_total_orders'])


features = ['aisle_freq_enc', 'order_dow', 'days_since_prior_order', 'user_total_orders']
X = data_set[features]
y = data_set['reordered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
  

print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))


