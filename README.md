# ML-project
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import seaborn as sns # to make boxplot ,histogram and heatmap
import matplotlib.pyplot as plt
from collections import Counter #for analyis most common names
import gc
import dask.dataframe as dd
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import KFold

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Initial Memory: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Final Memory: {end_mem:.2f} MB')
    return df



path = '/content/drive/MyDrive/ML/'

orders = reduce_mem_usage(pd.read_csv(path + 'orders.csv'))
products = reduce_mem_usage(pd.read_csv(path + 'products.csv'))
departments = reduce_mem_usage(pd.read_csv(path + 'departments.csv'))

order_products = reduce_mem_usage(pd.read_csv(path + 'order_products__train.csv'))
aisles = reduce_mem_usage(pd.read_csv(path + 'aisles.csv'))


print("Merging Data...")


df = pd.merge(order_products, orders, on='order_id', how='left')

df = pd.merge(df, products, on='product_id', how='left')

df = pd.merge(df, departments, on='department_id', how='left')


del orders, products, departments, order_products, aisles
gc.collect()

print("Data Loaded and Merged Successfully.")
plt.figure(figsize=(10, 5))
sns.histplot(df['days_since_prior_order'], bins=30, kde=False, color='skyblue')
plt.title('Distribution of Days Since Prior Order')
plt.xlabel('Days')
plt.ylabel('Count')
plt.show()

#Missing Value Analysis

missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
print("\nMissing Values:\n", missing_values[missing_values > 0])

plt.figure(figsize=(8, 4))
sns.barplot(x=missing_percent.index, y=missing_percent.values, palette='Reds_r')
plt.title("Percentage of Missing Values per Feature")
plt.ylabel("Percent %")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['days_since_prior_order'].dropna(), bins=30, kde=True, color='teal')
plt.title('Distribution of Days Since Prior Order')
plt.xlabel('Days')
plt.show()


top_depts = df['department'].value_counts().head(10)
plt.figure(figsize=(12, 5))
sns.barplot(x=top_depts.index, y=top_depts.values, palette='viridis')
plt.title('Top 10 Best Selling Departments (Cardinality Analysis)')
plt.xticks(rotation=45)
plt.show()


sample_heatmap = df.sample(n=min(100000, len(df)), random_state=42)
heatmap_data = sample_heatmap.groupby(['order_dow', 'order_hour_of_day']).size().unstack()


plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False)
plt.title("Seasonality Heatmap: Order Volume by Day & Hour")
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.show()


numeric_cols = ['order_number', 'order_dow', 'order_hour_of_day', 
                'days_since_prior_order', 'add_to_cart_order', 'reordered']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Numeric Features")
plt.show()


print("\nImputing Missing Values...")
df['days_since_prior_order'] = df['days_since_prior_order'].fillna(-1)
print(f"Missing values in 'days_since_prior_order' after imputation: {df['days_since_prior_order'].isnull().sum()}")


plt.figure(figsize=(10, 2))
sns.boxplot(x=df['add_to_cart_order'])
plt.title("Boxplot: Add to Cart Order (Before Winsorizing)")
plt.show()




upper_limit = df['add_to_cart_order'].quantile(0.99)
print(f"Capping outliers for 'add_to_cart_order' at 99th percentile: {upper_limit}")

df.loc[df['add_to_cart_order'] > upper_limit, 'add_to_cart_order'] = upper_limit

plt.figure(figsize=(10, 2))
sns.boxplot(x=df['add_to_cart_order'])
plt.title("Boxplot: Add to Cart Order (After Winsorizing)")
plt.show()

print("\nStep 1 (Ingestion), Step 2 (EDA), and Step 3 (Cleaning) Completed Successfully!")

#4_Encoding Categorical Variables


if 'reordered' not in df.columns:
    print("تنبيه: عمود reordered غير موجود، تأكد من دمج ملف order_products بشكل صحيح.")
else:
    print("Target variable 'reordered' found. Proceeding with Encoding...")

print("Applying One-Hot Encoding on 'department'...")

df = pd.get_dummies(df, columns=['department'], prefix='dept', drop_first=True)

print(f"New columns created: {[col for col in df.columns if 'dept_' in col][:5]} ...")

print("Applying Frequency Encoding on 'product_id'...")

product_freq = df['product_id'].value_counts(normalize=True)

df['product_freq_enc'] = df['product_id'].map(product_freq)

print("Applying Target Encoding with K-Fold on 'aisle_id'...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

df['aisle_target_enc'] = np.nan

for train_index, val_index in kf.split(df):

  X_train, X_val = df.iloc[train_index], df.iloc[val_index]

  means = X_train.groupby('aisle_id')['reordered'].mean()

  df.loc[val_index, 'aisle_target_enc'] = X_val['aisle_id'].map(means)

global_mean = df['reordered'].mean()
df['aisle_target_enc'] = df['aisle_target_enc'].fillna(global_mean)

print("\nComparing Encoding Impact (Correlation with Target 'reordered'):")

encoding_cols = ['product_freq_enc', 'aisle_target_enc']

dept_col = [col for col in df.columns if 'dept_' in col][0]
encoding_cols.append(dept_col)


correlations = df[encoding_cols + ['reordered']].corr()['reordered'].drop('reordered')
print(correlations.sort_values(ascending=False))

print("\nEncoding Step Completed Successfully.")
print(df[['product_id', 'product_freq_enc', 'aisle_id', 'aisle_target_enc', 'reordered']].head())


# 5. Feature Engineering (MANDATORY LIST)

orders_meta = pd.read_csv(path + 'orders.csv', usecols=['user_id', 'order_number', 'days_since_prior_order'])

user_stats = orders_meta.groupby('user_id').agg({'order_number': 'max', 'days_since_prior_order': 'mean' }).rename(columns={'order_number': 'user_total_orders','days_since_prior_order': 'user_avg_days_between_orders'})

user_basket_stats = df.groupby('user_id').agg({ 'reordered': 'mean','product_id': 'count' }).rename(columns={ 'reordered': 'user_reorder_ratio', 'product_id': 'avg_basket_size'})

df = df.merge(user_stats, on='user_id', how='left')
df = df.merge(user_basket_stats, on='user_id', how='left')

del orders_meta, user_stats, user_basket_stats
gc.collect()
print("User features created.")

product_stats = df.groupby('product_id').agg({
    'reordered': 'mean', 
    'add_to_cart_order': 'mean', 
    'order_id': 'count' 
}).rename(columns={
    'reordered': 'prod_reorder_rate',
    'add_to_cart_order': 'prod_avg_position',
    'order_id': 'prod_popularity'
})


df = df.merge(product_stats, on='product_id', how='left')
del product_stats
gc.collect()
print("Product features created.")

user_prod_stats = df.groupby(['user_id', 'product_id']).agg({
    'order_id': 'count', 
    'reordered': 'mean' 
}).rename(columns={
    'order_id': 'up_purchase_count',
    'reordered': 'up_reorder_prob'
})


df = df.merge(user_prod_stats, on=['user_id', 'product_id'], how='left')
del user_prod_stats
gc.collect()
print("User-Product interaction features created.")


df['hour_sin'] = np.sin(2 * np.pi * df['order_hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['order_hour_of_day'] / 24)

df['is_weekend'] = df['order_dow'].isin([0, 1]).astype(int)

df['interaction_popularity_loyalty'] = np.log1p(df['prod_popularity']) * df['user_total_orders']


print("\nStarting Feature Scaling...")
from sklearn.preprocessing import StandardScaler

scale_cols = [
    'user_total_orders', 'user_avg_days_between_orders', 'user_reorder_ratio', 'avg_basket_size',
    'prod_reorder_rate', 'prod_avg_position', 'prod_popularity',
    'up_purchase_count', 'up_reorder_prob',
    'days_since_prior_order', 'interaction_popularity_loyalty'
]

for col in scale_cols:
    df[col] = df[col].fillna(df[col].mean())

scaler = StandardScaler()

scaled_features = scaler.fit_transform(df[scale_cols])
df_scaled = pd.DataFrame(scaled_features, columns=[f'scl_{c}' for c in scale_cols], index=df.index)


df = pd.concat([df, df_scaled], axis=1)

print("Feature Engineering & Scaling Completed Successfully.")
print(f"New Scaled Columns: {df_scaled.columns.tolist()}")
print(df.head())


























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












