#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV


# In[2]:


## This method is for determening outliers
def count_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    upper_ouliers = Q3 + 1.5 * IQR
    lower_outliers = Q1 - 1.5 * IQR
    outliers = series[(series < lower_outliers) | (series > upper_ouliers)]
    return len(outliers)


# ## Content of the tables
# ### Chemicals table - all of the concentration of the specific chemical, code for the sensor, amount of water(all the data is from 2023)
# ### Merged_oxygen table - the code of the sensor(it is either 01/07, but it is the same sensor), amount of oxygen, start time, end time(data from 2021)
# 

# In[3]:


ammonium = pd.read_parquet('data/Chemical measurements influent 2023_2024/ammonium_2024.parquet')
ammonium['datumBeginMeting'] = pd.to_datetime(ammonium['datumBeginMeting'])
ammonium = ammonium.rename(columns={"hstWaarde": "hstWaarde_ammonium_2023", "historianTagnummer": "historianTagnummer_ammonium"}).reset_index(drop=True)
ammonium['hstWaarde_ammonium_2023'] = ammonium['hstWaarde_ammonium_2023'].apply(pd.to_numeric, errors='coerce')


# In[4]:


outliers_count = count_outliers_iqr(ammonium['hstWaarde_ammonium_2023'])
print(f"Number of outliers:  {outliers_count}")
plt.boxplot(ammonium['hstWaarde_ammonium_2023'])


# In[5]:


nitrate = pd.read_parquet('data/Chemical measurements influent 2023_2024/nitrate_2024.parquet')
nitrate['datumBeginMeting'] = pd.to_datetime(nitrate['datumBeginMeting'])
nitrate = nitrate.rename(columns={"hstWaarde": "hstWaarde_nitrate", "historianTagnummer": "historianTagnummer_nitrate"}).reset_index(drop=True)
nitrate['hstWaarde_nitrate'] = nitrate['hstWaarde_nitrate'].apply(pd.to_numeric, errors='coerce')


# In[6]:


outliers_count = count_outliers_iqr(nitrate['hstWaarde_nitrate'])
print(f"Number of outliers:  {outliers_count}")
plt.boxplot(nitrate['hstWaarde_nitrate'])


# In[7]:


phosphate = pd.read_parquet('data/Chemical measurements influent 2023_2024/phosphate_2024.parquet')
phosphate['datumBeginMeting'] = pd.to_datetime(phosphate['datumBeginMeting'])
phosphate = phosphate.rename(columns={"hstWaarde": "hstWaarde_phosphate", "historianTagnummer": "historianTagnummer_phosphate"}).reset_index(drop=True)
phosphate['hstWaarde_phosphate'] = phosphate['hstWaarde_phosphate'].apply(pd.to_numeric, errors='coerce')


# In[8]:


outliers_count = count_outliers_iqr(phosphate['hstWaarde_phosphate'])
print(f"Number of outliers:  {outliers_count}")
plt.boxplot(phosphate['hstWaarde_phosphate'])


# In[13]:


merged = pd.merge(ammonium, nitrate, on="datumBeginMeting")
chemicals = pd.merge(merged, phosphate, on="datumBeginMeting")
chemicals_hourly = pd.DataFrame({
    'Hour': chemicals['datumBeginMeting'].dt.floor('h'),  # Hour column for grouping
    'ammonium': chemicals['hstWaarde_ammonium_2023'],
    'nitrate': chemicals['hstWaarde_nitrate'],
    'phosphate': chemicals['hstWaarde_phosphate']
})

hourly_means = chemicals_hourly.groupby('Hour')[['ammonium', 'nitrate', 'phosphate']].mean().reset_index()

hourly_means


# In[14]:


weather = pd.read_csv('weather.csv')
weather = weather.rename(columns={'Timestamp': 'Hour'})
weather['Hour'] = pd.to_datetime(weather['Hour'])
weather


# In[16]:


combined = pd.merge(hourly_means, weather, on='Hour', how='inner')
combined


# In[17]:


numeric_data = combined.select_dtypes(include='number')
spearman_corr = numeric_data.corr(method='spearman')

# Plot the heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(
    spearman_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Spearman Correlation"}
)
plt.title("Spearman Correlation Heatmap")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[18]:


from sklearn.feature_selection import mutual_info_regression

# Example: measure mutual information with respect to a target
target = 'nitrate'
X = combined.drop(columns=[target])
X = X.select_dtypes(include=['int64', 'float64']) 
y = combined[target]

mi = mutual_info_regression(X, y, discrete_features=False)
mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=mi_scores.values, y=mi_scores.index, palette="viridis")
plt.title("Mutual Information with Target (hstWaarde_nitrate)")
plt.xlabel("Mutual Information Score")
plt.tight_layout()
plt.show()


# In[23]:


target = 'nitrate'
features = ['DewPointTemp', 'AirPressure', 'Temperature']
X = combined[features]
y = combined[target]


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# In[72]:


model = Pipeline([
    ("imputing", SimpleImputer(strategy="median")),
    ("scaling", RobustScaler()),
    ("modeling", RandomForestRegressor(
        random_state=42,
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1 
    ))
])


# In[75]:


cv_score = cross_val_score(estimator=model, X=X_train, y=y_train)


# In[76]:


print(cv_score)


# In[66]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(full_pipeline, X_train, y_train, cv=kf)


# In[67]:


full_pipeline.fit(X_train, y_train)
y_test_pred = full_pipeline.predict(X_test)


# In[68]:


print("Train CV MSE :", mean_squared_error(y_train, y_train_pred))
print("Train CV RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))

print("Test MSE     :", mean_squared_error(y_test, y_test_pred))
print("Test RMSE    :", np.sqrt(mean_squared_error(y_test, y_test_pred)))


# In[ ]:





# In[ ]:




