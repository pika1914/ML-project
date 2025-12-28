# ML-project
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import seaborn as sns # to make boxplot ,histogram and heatmap
import matplotlib.pyplot as plt
from collections import Counter #for analyis most common names
import gc
from sklearn.impute import SimpleImputer
import glob
import os
path = '/content/drive/MyDrive/ML/*.csv'
df = dd.read_csv(path)
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

#TASk_A_Classification

#import_Libraries

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import time

#1_data_preparation , feature_selection
feature_cols = ['user_avg_days', 'aisle_popularity', 'user_bought_aisle_times','is_weekend', 'hour_sin', 'hour_cos', 'aisle_freq_enc','aisle_target_enc', 'daily_sales' ]

df_model = data_set[feature_cols + ['is_reordered']].dropna()

X = df_model[feature_cols]
y = df_model['is_reordered']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}


def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    start_time = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)

    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training Time: {train_time:.4f} seconds")
    print("-" * 30)    
    results[name] = acc
#_2 logistic_regression
lr = LogisticRegression(penalty='l2', class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
evaluate_model("Logistic Regression", lr, X_train_scaled, X_test_scaled, y_train, y_test)

#_3 KNN
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model("KNN", knn, X_train_scaled, X_test_scaled, y_train, y_test)

#_4 SVM
svm_linear = SVC(kernel='linear', probability=True, random_state=42)
evaluate_model("SVM (Linear)", svm_linear, X_train_scaled, X_test_scaled, y_train, y_test)


svm_rbf = SVC(kernel='rbf', probability=True, random_state=42)
evaluate_model("SVM (RBF)", svm_rbf, X_train_scaled, X_test_scaled, y_train, y_test)
#_5 Decision_Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
evaluate_model("Decision Tree", dt, X_train, X_test, y_train, y_test)

#_6 Random_Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test)

#_7 Stacking_Classifier
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42)
evaluate_model("XGBoost", xgb, X_train, X_test, y_train, y_test)

estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=50, eval_metric='logloss', random_state=42))
]

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
evaluate_model("Stacking Classifier", stacking_clf, X_train, X_test, y_train, y_test)

df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
df = df.sort_values(by='Accuracy', ascending=False)
df

# _4_Regression (Task B)

#import_Libraries

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#_1 create_regression_target , feature_selection

np.random.seed(42)
data_set['days_to_next_order'] = np.random.randint(1, 30, len(data_set))

feature_cols_reg = ['user_avg_days', 'aisle_popularity', 'user_bought_aisle_times','is_weekend', 'hour_sin', 'hour_cos', 'aisle_freq_enc','aisle_target_enc','daily_sales']

df_reg = data_set[feature_cols_reg + ['days_to_next_order']].dropna()

X_reg = df_reg[feature_cols_reg]
y_reg = df_reg['days_to_next_order']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

reg_results = {}

def evaluate_regressor(name, model, X_tr, X_te, y_tr, y_te):
    start_time = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_te)
    
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"Time: {train_time:.4f} sec")
    print("-" * 30)
    
    reg_results[name] = rmse

print("\n" + "="*40)
print(" STARTING TASK B: REGRESSION ")
print("="*40 + "\n")

#_2_linear_regression
lr_reg = LinearRegression()
evaluate_regressor("Linear Regression", lr_reg, X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r)

#_3 lasso
lasso = Lasso(alpha=0.1, random_state=42)
evaluate_regressor("Lasso (L1)", lasso, X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r)

#_4 ridge
ridge = Ridge(alpha=1.0, random_state=42)
evaluate_regressor("Ridge (L2)", ridge, X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r)

#_5
enet = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
evaluate_regressor("Elastic Net", enet, X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r)

#_6 elastic_net
svr_lin = SVR(kernel='linear')
evaluate_regressor("SVR (Linear)", svr_lin, X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r)

#_7 SVR
svr_rbf = SVR(kernel='rbf')
evaluate_regressor("SVR (RBF)", svr_rbf, X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r)

#_8 KNN
knn_reg = KNeighborsRegressor(n_neighbors=5)
evaluate_regressor("KNN Regressor", knn_reg, X_train_r_scaled, X_test_r_scaled, y_train_r, y_test_r)

#_9 decision_tree
dt_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
evaluate_regressor("Decision Tree Regressor", dt_reg, X_train_r, X_test_r, y_train_r, y_test_r)

#_10 random_forest
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_regressor("Random Forest Regressor", rf_reg, X_train_r, X_test_r, y_train_r, y_test_r)

#_11 Gradient_boosting_regressor
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
evaluate_regressor("XGBoost Regressor", xgb_reg, X_train_r, X_test_r, y_train_r, y_test_r)

#print_all
df_reg = pd.DataFrame(list(reg_results.items()), columns=['Model', 'RMSE'])
df_reg = df_reg.sort_values(by='RMSE', ascending=True)
df_reg












