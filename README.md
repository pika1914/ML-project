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

# 6. Advanced Feature Engineering

if 'prod_popularity' in df.columns and 'user_total_orders' in df.columns:
    df['interaction_user_prod_rank'] = df['prod_popularity'] * df['user_total_orders']

df['log_days_since_prior'] = np.log1p(df['days_since_prior_order'])


df['user_order_frequency'] = df['user_total_orders'] / (df['user_avg_days_between_orders'] + 1)

print("Advanced features created: ['interaction_user_prod_rank', 'log_days_since_prior', 'user_order_frequency']")

# 7. Dimensionality & Multicollinearity (VIF)

print("\n=== Step 7: Multicollinearity Check (VIF) ===")
from statsmodels.stats.outliers_influence import variance_inflation_factor


numeric_feats = ['user_total_orders', 'user_avg_days_between_orders', 
                 'avg_basket_size', 'days_since_prior_order', 
                 'prod_reorder_rate', 'user_reorder_ratio']


X_vif = df[numeric_feats].fillna(df[numeric_feats].mean())

X_vif_sample = X_vif.sample(n=10000, random_state=42)


vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_feats
vif_data["VIF"] = [variance_inflation_factor(X_vif_sample.values, i) 
                   for i in range(len(numeric_feats))]

print(vif_data.sort_values(by="VIF", ascending=False))
print("\nNote: VIF > 5 or 10 indicates high multicollinearity. Consider removing these features.")



# 8. Imbalanced Data Handling


target_count = df['reordered'].value_counts()
print("Class Distribution:\n", target_count)
print("Ratio (0:1):", target_count[0] / target_count[1])

plt.figure(figsize=(6, 4))
sns.countplot(x='reordered', data=df, palette='pastel')
plt.title('Target Class Distribution (Imbalanced)')
plt.show()



from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', 
                                     classes=np.unique(df['reordered']), 
                                     y=df['reordered'])
class_weight_dict = dict(zip(np.unique(df['reordered']), class_weights))
print("Computed Class Weights:", class_weight_dict)


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
print("SMOTE object initialized (Apply strictly within training loop/pipeline to avoid leakage).")



# 9. Time-Aware Splitting


unique_users = df['user_id'].unique()
train_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

print(f"Total Users: {len(unique_users)}")
print(f"Train Users: {len(train_users)}")
print(f"Validation Users: {len(val_users)}")

X_train_full = df[df['user_id'].isin(train_users)]


X_val_full = df[df['user_id'].isin(val_users)]

print(f"Train Set Shape: {X_train_full.shape}")
print(f"Validation Set Shape: {X_val_full.shape}")


features_to_drop = ['order_id', 'user_id', 'product_id', 'eval_set', 
                    'product_name', 'department', 'aisle'] # وأي عمود نصي آخر
target_col = 'reordered'



available_features = [c for c in X_train_full.columns if c not in features_to_drop and c != target_col]

print(f"\nFinal Features List ({len(available_features)}):")
print(available_features[:10], "...")





X_train = X_train_full[available_features].fillna(0) # تعويض بسيط لما تبقى
y_train = X_train_full[target_col]

X_val = X_val_full[available_features].fillna(0)
y_val = X_val_full[target_col]

print("\nData Splits Ready for Modeling.")


# Task A: Classification

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
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l2', C=1.0, class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=42)
    ),
    
    
    "KNN (k=5)": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    ),
    
    
    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
    
    
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
    
    
    "XGBoost": XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        eval_metric='logloss',
        scale_pos_weight=10, 
        random_state=42, 
        n_jobs=-1
    )
}



svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42)
)


results_df = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()
    
  
    model.fit(X_train, y_train)
    
   
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.steps[-1][1].predict_proba(X_val)[:, 1]
    
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    elapsed = time.time() - start_time
    
    print(f"--> Done. Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f} ({elapsed:.2f}s)")
    
    results_df.append({
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "Time (s)": elapsed
    })

X_train_sub = X_train.iloc[:20000]
y_train_sub = y_train.iloc[:20000]

svm_configs = [
    ("SVM (Linear)", SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)),
    ("SVM (RBF)", SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
]


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
    
    print(f"--> {name} Done (Subset Train). Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    results_df.append({"Model": name + " (Subset)", "Accuracy": acc, "F1-Score": f1, "AUC-ROC": auc, "Time (s)": elapsed})


estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=50, max_depth=5, eval_metric='logloss', random_state=42))
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=3 
)

start_time = time.time()
stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_val)
y_prob_stack = stacking_clf.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred_stack)
f1 = f1_score(y_val, y_pred_stack)
auc = roc_auc_score(y_val, y_prob_stack)
elapsed = time.time() - start_time

print(f"--> Stacking Done. Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
results_df.append({"Model": "Stacking Classifier", "Accuracy": acc, "F1-Score": f1, "AUC-ROC": auc, "Time (s)": elapsed})


print("\n=== Final Model Comparison ===")
final_results = pd.DataFrame(results_df).sort_values(by="AUC-ROC", ascending=False)
print(final_results)


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

df_reg = df[df['days_since_prior_order'] >= 0].copy()

target_reg = 'days_since_prior_order'


drop_cols_reg = [target_reg, 'log_days_since_prior', 'order_id', 'user_id', 
                 'product_id', 'eval_set', 'product_name', 'department', 'aisle']

features_reg = [c for c in df_reg.columns if c not in drop_cols_reg]
print(f"Regression Features ({len(features_reg)}): {features_reg}")

train_users_reg, val_users_reg = train_test_split(df_reg['user_id'].unique(), test_size=0.2, random_state=42)

X_train_reg = df_reg[df_reg['user_id'].isin(train_users_reg)][features_reg].fillna(0)
y_train_reg = df_reg[df_reg['user_id'].isin(train_users_reg)][target_reg]

X_val_reg = df_reg[df_reg['user_id'].isin(val_users_reg)][features_reg].fillna(0)
y_val_reg = df_reg[df_reg['user_id'].isin(val_users_reg)][target_reg]


scaler_reg = StandardScaler()
X_train_reg_s = scaler_reg.fit_transform(X_train_reg)
X_val_reg_s = scaler_reg.transform(X_val_reg)

reg_models = {
    # Ordinary Least Squares & Regularized Variants
    "Linear Regression": LinearRegression(),
    "Lasso (L1)": Lasso(alpha=0.1, random_state=42),
    "Ridge (L2)": Ridge(alpha=1.0, random_state=42),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    
    # KNN Regressor
    "KNN Regressor (k=5)": KNeighborsRegressor(n_neighbors=5),
    
    # Trees & Ensembles (No Scaling needed usually, but passing scaled is fine)
    "Decision Tree Reg": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest Reg": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
    "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
}

# 3 (Training Loop)

reg_results = []

for name, model in reg_models.items():
    print(f"Training {name}...")
    start_time = time.time()
    
 
    if name in ["Linear Regression", "Lasso (L1)", "Ridge (L2)", "Elastic Net", "KNN Regressor (k=5)"]:
        model.fit(X_train_reg_s, y_train_reg)
        y_pred = model.predict(X_val_reg_s)
    else:
        model.fit(X_train_reg, y_train_reg) # الشجريات لا تحتاج تحجيم
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


final_reg_df = pd.DataFrame(reg_results).sort_values(by="RMSE", ascending=True)
print("\n=== Final Regression Results ===")
print(final_reg_df)

plt.figure(figsize=(10, 5))
sns.barplot(x="RMSE", y="Model", data=final_reg_df, palette="magma")
plt.title("Regression Models Comparison (RMSE - Lower is Better)")
plt.show()

best_model_name = final_reg_df.iloc[0]['Model']
print(f"\nPlotting Residuals for best model: {best_model_name}")

plt.figure(figsize=(8, 8))
plt.scatter(y_val_reg, stack_reg.predict(X_val_reg), alpha=0.3, s=10)
plt.plot([0, 30], [0, 30], 'r--') # خط المثالية
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

print("\nRunning RandomizedSearchCV for XGBoost (Boosting)...")

xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'scale_pos_weight': [1, 10] # للتعامل مع عدم التوازن
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

rf_grid.fit(X_train.iloc[:20000], y_train.iloc[:20000]) # عينة أصغر للسرعة

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


#Stacking

print("\nBuilding Final Stacking Ensemble...")
stacking_final = StackingClassifier(
    estimators=[
        ('tuned_xgb', best_xgb),
        ('tuned_rf', best_rf)
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=tscv
)

stacking_final.fit(X_train, y_train)
y_pred_stack = stacking_final.predict(X_val)
print(f"Final Stacking F1 Score: {f1_score(y_val, y_pred_stack):.4f}")


























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


###############################################################################









