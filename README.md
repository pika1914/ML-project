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
data_fram = dd.read_csv(path)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import KFold

#memory_optimaiz

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def reduce_mem_usage(data_fram):
    start_mem = data_fram.memory_usage().sum() / 1024**2
    print(f'Initial Memory: {start_mem:.2f} MB')

    for col in data_fram.columns:
        col_type = data_fram[col].dtype

        if col_type != object:
            c_min = data_fram[col].min()
            c_max = data_fram[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data_fram[col] = data_fram[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data_fram[col] = data_fram[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data_fram[col] = data_fram[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data_fram[col] = data_fram[col].astype(np.float32)
                else:
                    data_fram[col] = data_fram[col].astype(np.float64)
        else:
            data_fram[col] = data_fram[col].astype('category')

    end_mem = data_fram.memory_usage().sum() / 1024**2
    print(f'Final Memory: {end_mem:.2f} MB')
    
    return data_fram



path = '/content/drive/MyDrive/ML/'

orders = reduce_mem_usage(pd.read_csv(path + 'orders.csv'))
products = reduce_mem_usage(pd.read_csv(path + 'products.csv'))
departments = reduce_mem_usage(pd.read_csv(path + 'departments.csv'))

order_products = reduce_mem_usage(pd.read_csv(path + 'order_products__train.csv'))
aisles = reduce_mem_usage(pd.read_csv(path + 'aisles.csv'))

#merg data

print("Merging Data ")


data_fram = pd.merge(order_products, orders, on='order_id', how='left')

data_fram = pd.merge(data_fram, products, on='product_id', how='left')

data_fram = pd.merge(data_fram, departments, on='department_id', how='left')


del orders, products, departments, order_products, aisles
gc.collect()

print("Data Loaded and Merged ")
plt.figure(figsize=(10, 5))
sns.histplot(data_fram['days_since_prior_order'], bins=30, kde=False, color='skyblue')
plt.title('Distribution of Days Since Prior Order')
plt.xlabel('Days')
plt.ylabel('Count')
plt.show()

#Missing Value Analysis

missing_values = data_fram.isnull().sum()
missing_percent = (missing_values / len(data_fram)) * 100
print("\nMissing Values:\n", missing_values[missing_values > 0])

plt.figure(figsize=(8, 4))
sns.barplot(x=missing_percent.index, y=missing_percent.values, palette='Reds_r')
plt.title("Percentage of Missing Values per Feature")
plt.ylabel("Percent %")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(data_fram['days_since_prior_order'].dropna(), bins=30, kde=True, color='teal')
plt.title('Distribution of Days Since Prior Order')
plt.xlabel('Days')
plt.show()


top_depts = data_fram['department'].value_counts().head(10)
plt.figure(figsize=(12, 5))
sns.barplot(x=top_depts.index, y=top_depts.values, palette='viridis')
plt.title('Top 10 Best Selling Departments (Cardinality Analysis)')
plt.xticks(rotation=45)
plt.show()


sample_heatmap = data_fram.sample(n=min(100000, len(data_fram)), random_state=42)
heatmap_data = sample_heatmap.groupby(['order_dow', 'order_hour_of_day']).size().unstack()


plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False)
plt.title("Seasonality Heatmap: Order Volume by Day and Hour")
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
plt.show()


numeric_cols = ['order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order', 'add_to_cart_order', 'reordered']
corr_matrix = data_fram[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Numeric Features")
plt.show()


print("\nImputing Missing Values ")
data_fram['days_since_prior_order'] = data_fram['days_since_prior_order'].fillna(-1)
print(f"Missing values in 'days_since_prior_order' after imputation: {data_fram['days_since_prior_order'].isnull().sum()}")


plt.figure(figsize=(10, 2))
sns.boxplot(x=data_fram['add_to_cart_order'])
plt.title("Boxplot: Add to Cart Order (Before Winsorizing)")
plt.show()




upper_limit = data_fram['add_to_cart_order'].quantile(0.99)
print(f"Capping outliers for 'add_to_cart_order' at 99th percentile: {upper_limit}")

data_fram.loc[data_fram['add_to_cart_order'] > upper_limit, 'add_to_cart_order'] = upper_limit

plt.figure(figsize=(10, 2))
sns.boxplot(x=data_fram['add_to_cart_order'])
plt.title("Boxplot: Add to Cart Order (After Winsorizing)")
plt.show()

print("\nFinished loading, exploring and cleaning the data.")

#4_Encoding Categorical Variables


if 'reordered' not in data_fram.columns:
    print("Warning: 'reordered' column not found. Ensure 'order_products' is merged correctly.")
else:
    print("Target variable 'reordered' found. Proceeding with Encoding ")

print("One-Hot Encoding in department")

data_fram = pd.get_dummies(data_fram, columns=['department'], prefix='dept', drop_first=True)

print(f"New columns created: {[col for col in data_fram.columns if 'dept_' in col][:5]}  ")

print("Encoding product_id")

product_freq = data_fram['product_id'].value_counts(normalize=True)

data_fram['product_freq_enc'] = data_fram['product_id'].map(product_freq)

print("Encoding aisle feature")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

data_fram['aisle_target_enc'] = np.nan

for train_index, val_index in kf.split(data_fram):

  X_train, X_val = data_fram.iloc[train_index], data_fram.iloc[val_index]

  means = X_train.groupby('aisle_id')['reordered'].mean()

  data_fram.loc[val_index, 'aisle_target_enc'] = X_val['aisle_id'].map(means)

global_mean = data_fram['reordered'].mean()
data_fram['aisle_target_enc'] = data_fram['aisle_target_enc'].fillna(global_mean)

print("\nComparing Encoding Impact (Correlation with Target 'reordered'):")

enc_features = ['product_freq_enc', 'aisle_target_enc']

dept_col = [col for col in data_fram.columns if 'dept_' in col][0]
enc_features.append(dept_col)


correlations = data_fram[enc_features + ['reordered']].corr()['reordered'].drop('reordered')
print(correlations.sort_values(ascending=False))

print("\nEncoding Step Completed")
print(data_fram[['product_id', 'product_freq_enc', 'aisle_id', 'aisle_target_enc', 'reordered']].head())


# 5_Feature_Engineering

orders_meta = pd.read_csv(path + 'orders.csv', usecols=['user_id', 'order_number', 'days_since_prior_order'])

user_stats = orders_meta.groupby('user_id').agg({'order_number': 'max', 'days_since_prior_order': 'mean' }).rename(columns={'order_number': 'user_total_orders','days_since_prior_order': 'user_avg_days_between_orders'})

user_basket_stats = data_fram.groupby('user_id').agg({ 'reordered': 'mean','product_id': 'count' }).rename(columns={ 'reordered': 'user_reorder_ratio', 'product_id': 'avg_basket_size'})

data_fram = data_fram.merge(user_stats, on='user_id', how='left')
data_fram = data_fram.merge(user_basket_stats, on='user_id', how='left')

del orders_meta, user_stats, user_basket_stats
gc.collect()
print("User features created.")

product_stats = data_fram.groupby('product_id').agg({'reordered': 'mean', 'add_to_cart_order': 'mean', 'order_id': 'count' }).rename(columns={'reordered': 'prod_reorder_rate','add_to_cart_order': 'prod_avg_position','order_id': 'prod_popularity'})

data_fram = data_fram.merge(product_stats, on='product_id', how='left')
del product_stats
gc.collect()
print("Product features created.")

user_prod_stats = data_fram.groupby(['user_id', 'product_id']).agg({'order_id': 'count', 'reordered': 'mean' }).rename(columns={'order_id': 'up_purchase_count','reordered': 'up_reorder_prob'})

data_fram = data_fram.merge(user_prod_stats, on=['user_id', 'product_id'], how='left')
del user_prod_stats
gc.collect()
print("User-Product interaction features created.")


data_fram['hour_sin'] = np.sin(2 * np.pi * data_fram['order_hour_of_day'] / 24)
data_fram['hour_cos'] = np.cos(2 * np.pi * data_fram['order_hour_of_day'] / 24)

data_fram['is_weekend'] = data_fram['order_dow'].isin([0, 1]).astype(int)

data_fram['interaction_popularity_loyalty'] = np.log1p(data_fram['prod_popularity']) * data_fram['user_total_orders']


print("\nStarting Feature Scaling ")
from sklearn.preprocessing import StandardScaler

scale_cols = ['user_total_orders', 'user_avg_days_between_orders', 'user_reorder_ratio', 'avg_basket_size','prod_reorder_rate', 'prod_avg_position', 'prod_popularity','up_purchase_count', 'up_reorder_prob','days_since_prior_order', 'interaction_popularity_loyalty']

for col in scale_cols:
    data_fram[col] = data_fram[col].fillna(data_fram[col].mean())

scaler = StandardScaler()

scaled_features = scaler.fit_transform(data_fram[scale_cols])
data_fram_scaled = pd.DataFrame(scaled_features, columns=[f'scl_{c}' for c in scale_cols], index=data_fram.index)


data_fram = pd.concat([data_fram, data_fram_scaled], axis=1)

print("Feature Engineering and Scaling ")
print(f"New Scaled Columns: {data_fram_scaled.columns.tolist()}")
print(data_fram.head())

#6_Advanced Feature Engineering

if 'prod_popularity' in data_fram.columns and 'user_total_orders' in data_fram.columns:
    data_fram['interaction_user_prod_rank'] = data_fram['prod_popularity'] * data_fram['user_total_orders']

data_fram['log_days_since_prior'] = np.log1p(data_fram['days_since_prior_order'])


data_fram['user_order_frequency'] = data_fram['user_total_orders'] / (data_fram['user_avg_days_between_orders'] + 1)

print("some extra features ")

#_7_Dimensionality and Multicollinearity

print("Checking multicollinearity between some numeric features")
from statsmodels.stats.outliers_influence import variance_inflation_factor


numeric_feats = ['user_total_orders', 'user_avg_days_between_orders', 'avg_basket_size', 'days_since_prior_order', 'prod_reorder_rate', 'user_reorder_ratio']


X_vif = data_fram[numeric_feats].fillna(data_fram[numeric_feats].mean())

X_vif_sample = X_vif.sample(n=10000, random_state=42)


vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_feats
vif_data["VIF"] = [variance_inflation_factor(X_vif_sample.values, i) for i in range(len(numeric_feats))]

print(vif_data.sort_values(by="VIF", ascending=False))
print("\nNote: VIF > 5 or 10 indicates high multicollinearity. Consider removing these features.")



# 8 _ Imbalanced Data Handling


target_count = data_fram['reordered'].value_counts()
print("Class Distribution:\n", target_count)
print("Ratio (0:1):", target_count[0] / target_count[1])

plt.figure(figsize=(6, 4))
sns.countplot(x='reordered', data=data_fram, palette='pastel')
plt.title('Target Class Distribution (Imbalanced)')
plt.show()



from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data_fram['reordered']),y=data_fram['reordered'])
class_weight_dict = dict(zip(np.unique(data_fram['reordered']), class_weights))
print("Computed Class Weights:", class_weight_dict)


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
print("SMOTE initialized use to avoid data leakage)")



#9_Time_Aware_Splitting


unique_users = data_fram['user_id'].unique()
train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

print(f"Total Users: {len(unique_users)}")
print(f"Train Users: {len(train_users)}")
print(f"Validation Users: {len(val_users)}")

X_train_full = data_fram[data_fram['user_id'].isin(train_users)]


X_val_full = data_fram[data_fram['user_id'].isin(val_users)]

print(f"Train Set Shape: {X_train_full.shape}")
print(f"Validation Set Shape: {X_val_full.shape}")


features_to_drop = ['order_id', 'user_id', 'product_id', 'eval_set', 'product_name', 'department', 'aisle'] 
target_col = 'reordered'



available_features = [c for c in X_train_full.columns if c not in features_to_drop and c != target_col]

print(f"\nFinal Features List ({len(available_features)}):")
print(available_features[:10])





X_train = X_train_full[available_features].fillna(0)
y_train = X_train_full[target_col]

X_val = X_val_full[available_features].fillna(0)
y_val = X_val_full[target_col]

print("\nData Splits Ready for Modeling.")


#task A Classification

import time
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report


models = {
    # Logistic Regression: L2 penalty (Ridge) default, Class Weighted
    "Logistic Regression": make_pipeline(StandardScaler(),LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)),
    
    
    "KNN (k=5)": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)),"Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42), "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1), "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss',scale_pos_weight=10, random_state=42, n_jobs=-1)}



svm_model = make_pipeline(StandardScaler(),SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42))


model_results = []

for name, model in models.items():
    print(f"\nTraining {name} ")
    start_time = time.time()
    
  
    model.fit(X_train, y_train)
    
   
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.steps[-1][1].predict_proba(X_val)[:, 1]
    
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    elapsed = time.time() - start_time
    
    print(f"Model: {name} | Accuracy: {acc:.3f} | F1 Score: {f1:.3f}")
    
    model_results.append({"Model": name,"Accuracy": acc,"F1-Score": f1,"AUC-ROC": auc,"Time (s)": elapsed})

X_train_sub = X_train.iloc[:20000]
y_train_sub = y_train.iloc[:20000]

svm_configs = [("SVM (Linear)", SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)),("SVM (RBF)", SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))]


for name, clf in svm_configs:
    start_time = time.time()
    model = make_pipeline(StandardScaler(), clf) # Scaling is crucial for SVM
    model.fit(X_train_sub, y_train_sub)
    
    y_pred = model.predict(X_val) 
    y_prob = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    elapsed = time.time() - start_time
    
    print(f" {name} Done (Subset Train). Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    model_results.append({"Model": name + " (Subset)", "Accuracy": acc, "F1-Score": f1, "AUC-ROC": auc, "Time (s)": elapsed})


estimators = [('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),('xgb', XGBClassifier(n_estimators=50, max_depth=5, eval_metric='logloss', random_state=42))]
stacking_clf = StackingClassifier(estimators=estimators,final_estimator=LogisticRegression(),cv=3 )

start_time = time.time()
stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_val)
y_prob_stack = stacking_clf.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred_stack)
f1 = f1_score(y_val, y_pred_stack)
auc = roc_auc_score(y_val, y_prob_stack)
elapsed = time.time() - start_time

print(f"Stacking Done. Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
model_results.append({"Model": "Stacking Classifier", "Accuracy": acc, "F1-Score": f1, "AUC-ROC": auc, "Time (s)": elapsed})


print("Model comparison results")
final_results = pd.DataFrame(model_results).sort_values(by="AUC-ROC", ascending=False)
print(final_results[['Model', 'AUC-ROC']])


plt.figure(figsize=(10, 5))
sns.barplot(x="AUC-ROC", y="Model", data=final_results, palette="viridis")
plt.title("Model Comparison (AUC-ROC Score)")
plt.xlim(0, 1.0)
plt.show()


# Task B: Regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_fram_reg = data_fram[data_fram['days_since_prior_order'] >= 0].copy()

target_reg = 'days_since_prior_order'


drop_cols_reg = [target_reg, 'log_days_since_prior', 'order_id', 'user_id', 
                 'product_id', 'eval_set', 'product_name', 'department', 'aisle']

features_reg = [c for c in data_fram_reg.columns if c not in drop_cols_reg]
print(f"Regression Features ({len(features_reg)}): {features_reg}")

train_users_reg, val_users_reg = train_test_split(data_fram_reg['user_id'].unique(), test_size=0.2, random_state=42)

X_train_reg = data_fram_reg[data_fram_reg['user_id'].isin(train_users_reg)][features_reg].fillna(0)
y_train_reg = data_fram_reg[data_fram_reg['user_id'].isin(train_users_reg)][target_reg]

X_val_reg = data_fram_reg[data_fram_reg['user_id'].isin(val_users_reg)][features_reg].fillna(0)
y_val_reg = data_fram_reg[data_fram_reg['user_id'].isin(val_users_reg)][target_reg]


scaler_reg = StandardScaler()
X_train_reg_s = scaler_reg.fit_transform(X_train_reg)
X_val_reg_s = scaler_reg.transform(X_val_reg)

reg_models = {
    # Ordinary Least Squares and Regularized Variants
    "Linear Regression": LinearRegression(),
    "Lasso (L1)": Lasso(alpha=0.1, random_state=42),
    "Ridge (L2)": Ridge(alpha=1.0, random_state=42),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    
    # KNN Regressor
    "KNN Regressor (k=5)": KNeighborsRegressor(n_neighbors=5),
    
    # Trees and Ensembles (No Scaling needed usually, but passing scaled is fine)
    "Decision Tree Reg": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest Reg": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
}

# 3 (Training Loop)

reg_results = []

for name, model in reg_models.items():
    print(f"Training {name} ")
    start_time = time.time()
    
 
    if name in ["Linear Regression", "Lasso (L1)", "Ridge (L2)", "Elastic Net", "KNN Regressor (k=5)"]:
        model.fit(X_train_reg_s, y_train_reg)
        y_pred = model.predict(X_val_reg_s)
    else:
        model.fit(X_train_reg, y_train_reg) 
        y_pred = model.predict(X_val_reg)
        
   
    mae = mean_absolute_error(y_val_reg, y_pred)
    mse = mean_squared_error(y_val_reg, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val_reg, y_pred)
    elapsed = time.time() - start_time
    
    print(f"--> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f} ({elapsed:.2f}s)")
    
    reg_results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
        "Time (s)": elapsed
    })

X_train_sub_s = X_train_reg_s[:10000]
y_train_sub = y_train_reg.iloc[:10000]

svr_models = [
    ("SVR (Linear)", SVR(kernel='linear')),
    ("SVR (RBF)", SVR(kernel='rbf'))
]

for name, model in svr_models:
    start_time = time.time()
    model.fit(X_train_sub_s, y_train_sub)
    y_pred = model.predict(X_val_reg_s[:2000]) 
    
    rmse = np.sqrt(mean_squared_error(y_val_reg.iloc[:2000], y_pred))
    mae = mean_absolute_error(y_val_reg.iloc[:2000], y_pred)
    r2 = r2_score(y_val_reg.iloc[:2000], y_pred)
    
    print(f"--> {name} (Subset): RMSE: {rmse:.4f}, R2: {r2:.4f}")
    reg_results.append({"Model": name + " (Subset)", "RMSE": rmse, "MAE": mae, "R2 Score": r2, "Time (s)": time.time() - start_time})

#Stacked Regressor

stack_reg = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=50, max_depth=5, random_state=42))
    ],
    final_estimator=Ridge()
)
stack_reg.fit(X_train_reg, y_train_reg)
y_pred_stack = stack_reg.predict(X_val_reg)

rmse = np.sqrt(mean_squared_error(y_val_reg, y_pred_stack))
r2 = r2_score(y_val_reg, y_pred_stack)
reg_results.append({"Model": "Stacked Regressor", "RMSE": rmse, "MAE": mean_absolute_error(y_val_reg, y_pred_stack), "R2 Score": r2, "Time (s)": 0})
print(f"--> Stacked RMSE: {rmse:.4f}")


final_reg_data_fram = pd.DataFrame(reg_results).sort_values(by="RMSE", ascending=True)
print("\n=== Final Regression Results ===")
print(final_reg_data_fram)

plt.figure(figsize=(10, 5))
sns.barplot(x="RMSE", y="Model", data=final_reg_data_fram, palette="magma")
plt.title("Regression Models Comparison (RMSE - Lower is Better)")
plt.show()

best_model_name = final_reg_data_fram.iloc[0]['Model']
print(f"\nPlotting Residuals for best model: {best_model_name}")

plt.figure(figsize=(8, 8))
plt.scatter(y_val_reg, stack_reg.predict(X_val_reg), alpha=0.3, s=10)
plt.plot([0, 30], [0, 30], 'r--') 
plt.xlabel("Actual Days")
plt.ylabel("Predicted Days")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.show()

#Hyperparameter

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, f1_score

tscv = TimeSeriesSplit(n_splits=3)
print("Cross-Validation Strategy: TimeSeriesSplit (3 splits)")

f1_scorer = make_scorer(f1_score)

#XGBoost

print("\nRunning RandomizedSearchCV for XGBoost (Boosting) ")

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'scale_pos_weight': [1, 10] 
}

xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)

xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=10, 
    scoring=f1_scorer,
    cv=tscv, 
    verbose=1,
    random_state=42,
    n_jobs=-1
)


xgb_search.fit(X_train.iloc[:50000], y_train.iloc[:50000])

print(f"Best XGBoost Params: {xgb_search.best_params_}")
print(f"Best XGBoost F1 Score (CV): {xgb_search.best_score_:.4f}")


#Random Forest

rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5]
}

rf_model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

rf_grid = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_params,
    scoring=f1_scorer,
    cv=tscv, 
    verbose=1,
    n_jobs=-1
)

rf_grid.fit(X_train.iloc[:20000], y_train.iloc[:20000]) 

print(f"Best Random Forest Params: {rf_grid.best_params_}")
print(f"Best RF F1 Score (CV): {rf_grid.best_score_:.4f}")



#Evaluation

best_xgb = xgb_search.best_estimator_
best_xgb.fit(X_train, y_train) 
y_pred_xgb = best_xgb.predict(X_val)
print(f"Tuned XGBoost F1 on Validation: {f1_score(y_val, y_pred_xgb):.4f}")


best_rf = rf_grid.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_val)
print(f"Tuned Random Forest F1 on Validation: {f1_score(y_val, y_pred_rf):.4f}")



#Decision Boundaries

from sklearn.decomposition import PCA

# 1. Reduce to 2D for visualization
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_val.iloc[:500]) # Take a sample
y_vis = y_val.iloc[:500].values

# 2. Train a simple model on 2D data for visualization purpose
clf_vis = DecisionTreeClassifier(max_depth=5)
clf_vis.fit(X_vis, y_vis)

# 3. Create a meshgrid
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 4. Predict and Plot
Z = clf_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, s=20, edgecolor='k', cmap='viridis')
plt.title("Decision Boundary (PCA Projection - Decision Tree)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()






#Stacking

print("\nBuilding Final Stacking Ensemble ")
stack_model = StackingClassifier(
    estimators=[
        ('tuned_xgb', best_xgb),
        ('tuned_rf', best_rf)
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=tscv
)

stack_model.fit(X_train, y_train)
y_pred_stack = stack_model.predict(X_val)
print(f"Final Stacking F1 Score: {f1_score(y_val, y_pred_stack):.4f}")



























###############################################################################





from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

print("Plotting evaluation metrics")

best_model = stack_model  
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)[:, 1]

# 1. ROC Curve and PR Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))



fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, label=f'Stacking Model (AUC = {roc_auc:.2f})', color='darkorange')
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_title('ROC Curve')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend()


# Precision-Recall
precision, recall, _ = precision_recall_curve(y_val, y_prob)
ap = average_precision_score(y_val, y_prob)
ax2.plot(recall, precision, label=f'Stacking Model (AP = {ap:.2f})', color='purple')
ax2.set_title('Precision-Recall Curve (Crucial for Imbalanced)')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend()
plt.show()

# 2. Confusion Matrix and Calibration
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred, normalize='true')
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax3)
ax3.set_title('Normalized Confusion Matrix')

# Calibration Curve
prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10)
ax4.plot(prob_pred, prob_true, marker='o', label='Stacking Classifier')
ax4.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
ax4.set_title('Calibration Curve (Reliability Diagram)')
ax4.set_xlabel('Mean Predicted Probability')
ax4.set_ylabel('Fraction of Positives')
ax4.legend()
plt.show()

#Interactive Plot

print("=== Generating Interactive Dashboard ===")
import ipywidgets as widgets
from ipywidgets import interact

def plot_interactive_metrics(threshold=0.5):
 
    y_pred_adj = (y_prob >= threshold).astype(int)
    
  
    cm = confusion_matrix(y_val, y_pred_adj)
    
 
    acc = accuracy_score(y_val, y_pred_adj)
    f1 = f1_score(y_val, y_pred_adj)
    
   
    plt.figure(figsize=(10, 4))
    
 
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix (Threshold: {threshold:.2f})")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
   
    plt.subplot(1, 2, 2)
    plt.text(0.1, 0.6, f"Accuracy: {acc:.4f}", fontsize=15)
    plt.text(0.1, 0.4, f"F1 Score: {f1:.4f}", fontsize=15)
    plt.axis('off')
    plt.title("Model Metrics")
    
    plt.tight_layout()
    plt.show()


print("Use the slider below to change the classification threshold:")
interact(plot_interactive_metrics, threshold=(0.0, 1.0, 0.05));


#Explainability with SHAP

import shap

print("=== Generating SHAP Explanations ===")

explainer = shap.TreeExplainer(best_xgb)

X_shap_sample = X_val.sample(n=500, random_state=42)
shap_values = explainer.shap_values(X_shap_sample)

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot")
shap.summary_plot(shap_values, X_shap_sample, plot_type="bar")
plt.show()

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap_sample) # Dot plot
plt.show()

#Robustness and Stress Test

print("=== Robustness Test: Adding Noise ===")

X_val_noisy = X_val.copy()
noise = np.random.normal(0, 5, size=len(X_val_noisy)) 
X_val_noisy['days_since_prior_order'] = X_val_noisy['days_since_prior_order'] + noise

print("Evaluating model on Noisy Data ")
y_pred_noisy = best_model.predict(X_val_noisy)
f1_noisy = f1_score(y_val, y_pred_noisy)

print(f"Original F1 Score: {f1_score(y_val, y_pred):.4f}")
print(f"Noisy F1 Score: {f1_noisy:.4f}")
print(f"Performance Drop: {(f1_score(y_val, y_pred) - f1_noisy):.4f}")




#bouns


!pip install onnx onnxmltools skl2onnx h2o -q

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


#Neural-based approach


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


model_nn = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3), 
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') 
])

model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


print("Training Neural Network ")
history = model_nn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, 
    batch_size=1024, 
    callbacks=[early_stop],
    verbose=1
)


plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve (Check for Overfitting)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


loss_nn, acc_nn = model_nn.evaluate(X_val, y_val, verbose=0)
print(f"Neural Network Accuracy: {acc_nn:.4f}")


#Model Compression , Inference Speed


from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt


if 'best_rf' not in locals():
    print("Warning: 'best_rf' not found, training a small RF for demo ")
    best_rf = RandomForestClassifier(n_estimators=10, max_depth=5).fit(X_train, y_train)


initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(best_rf, initial_types=initial_type)

with open("rf_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("Model saved as 'rf_model.onnx'")


sample_input = X_val.iloc[0:1].values.astype(np.float32)


start = time.time()
for _ in range(1000):
    best_rf.predict(sample_input)
sklearn_time = (time.time() - start) / 1000
print(f"Sklearn Latency: {sklearn_time * 1000:.4f} ms per sample")


sess = rt.InferenceSession("rf_model.onnx", providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

start = time.time()
for _ in range(1000):
    sess.run([label_name], {input_name: sample_input})
onnx_time = (time.time() - start) / 1000
print(f"ONNX Latency:    {onnx_time * 1000:.4f} ms per sample")

print(f"Speedup: {sklearn_time / onnx_time:.2f}x faster with ONNX")



#AutoML

import h2o
from h2o.automl import H2OAutoML

# 1.H2O Cluster
h2o.init(max_mem_size='2G')

train_h2o = h2o.H2OFrame(pd.concat([X_train.iloc[:10000], y_train.iloc[:10000]], axis=1))
test_h2o = h2o.H2OFrame(pd.concat([X_val.iloc[:2000], y_val.iloc[:2000]], axis=1))

target_col_h2o = 'reordered'
feature_cols_h2o = [c for c in train_h2o.columns if c != target_col_h2o]


train_h2o[target_col_h2o] = train_h2o[target_col_h2o].asfactor()
test_h2o[target_col_h2o] = test_h2o[target_col_h2o].asfactor()

print("Running H2O AutoML (Limit: 5 models or 60 seconds) ")
aml = H2OAutoML(max_models=5, max_runtime_secs=120, seed=42, project_name="Instacart_Bonus")
aml.train(x=feature_cols_h2o, y=target_col_h2o, training_frame=train_h2o)


lb = aml.leaderboard
print("\nAutoML Leaderboard:")
print(lb.head(rows=10))


print(f"\nBest AutoML Model: {aml.leader.model_id}")
perf = aml.leader.model_performance(test_h2o)
print("AutoML Performance on Test Set:")
print(perf)



