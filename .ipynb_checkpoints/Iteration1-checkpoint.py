#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


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
ammonium = ammonium.rename(columns={"hstWaarde": "hstWaarde_ammonium_2023", "historianTagnummer": "historianTagnummer_ammonium", "datumBeginMeting": "datumBeginMeting_ammonium", "datumEindeMeting": "datumEindeMeting_ammonium"}).reset_index(drop=True)
ammonium['hstWaarde_ammonium_2023'] = ammonium['hstWaarde_ammonium_2023'].apply(pd.to_numeric, errors='coerce')


# In[4]:


outliers_count = count_outliers_iqr(ammonium['hstWaarde_ammonium_2023'])
print(f"Number of outliers:  {outliers_count}")
plt.boxplot(ammonium['hstWaarde_ammonium_2023'])


# In[5]:


nitrate = pd.read_parquet('data/Chemical measurements influent 2023_2024/nitrate_2024.parquet')
nitrate['datumBeginMeting'] = pd.to_datetime(nitrate['datumBeginMeting'])
nitrate = nitrate.rename(columns={"hstWaarde": "hstWaarde_nitrate", "historianTagnummer": "historianTagnummer_nitrate", "datumBeginMeting": "datumBeginMeting_nitrate", "datumEndeMeting": "datumEndeMeting_nitrate"}).reset_index(drop=True)
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


# In[9]:


ammonium_desc = ammonium.describe()
nitrate_desc = nitrate.describe()
phosphate_desc = phosphate.describe()

ammonium_desc.columns = [f'Ammonium_{col}' for col in ammonium_desc.columns]
nitrate_desc.columns = [f'Nitrate_{col}' for col in nitrate_desc.columns]
phosphate_desc.columns = [f'Phosphate_{col}' for col in phosphate_desc.columns]

summary_df = pd.concat([ammonium_desc, nitrate_desc, phosphate_desc], axis=1)
summary_df


# In[10]:


chemicals = pd.concat([ammonium, nitrate, phosphate], axis=1)
chemicals = chemicals.drop(columns=['waardebewerkingsmethodeCode', 'datumEindeMeting', 'datumBeginMeting_nitrate',
                                    'historianTagnummer_ammonium', 'datumBeginMeting_ammonium', 'datumEindeMeting_ammonium',
                                    'historianTagnummer_nitrate', 'historianTagnummer_phosphate'])
chemicals['hstWaarde_ammonium_2023'] = chemicals['hstWaarde_ammonium_2023'].fillna(chemicals['hstWaarde_ammonium_2023'].median())
chemicals


# In[11]:


chemicalsForCorr = chemicals[['hstWaarde_ammonium_2023', 'hstWaarde_nitrate', 'hstWaarde_phosphate']]
corr = chemicalsForCorr.corr()
plt.figure(figsize=(10, 8))
plt.title("Correlation Heatmap")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.heatmap(
    corr,
    annot=True,
    annot_kws={"size": 24},       # Font size for numbers
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}  # Optional
)
plt.show()


# In[12]:


sensor_13 = pd.read_parquet('data/HistoricalWWTPData/DTWINTERNALWWTPDATA/Data/EDE_B121069913_K600.MTW.parquet')
sensor_13['hstWaarde'] = sensor_13['hstWaarde'].apply(pd.to_numeric, errors='coerce')
sensor_13 = sensor_13.rename(columns={"hstWaarde": "hstWaarde_ammonium_2021"}).reset_index(drop=True)
sensor_13.hist(bins=70, figsize=(10, 8))
plt.title(f'Distribution of ammonium in 2021')


# In[15]:


plt.figure(figsize=(10, 8))
plt.scatter(chemicals['hstWaarde_ammonium_2023'].sample(2000), chemicals['hstWaarde_phosphate'].sample(2000))
plt.xlabel('Ammonium')
plt.ylabel('Phosphate')
plt.title('Scatter Plot of Ammonium vs Nitrate')
plt.grid(True)


# In[16]:


oxygen_a = pd.read_parquet('data/OxygenData2024/oxygen_a_2024.parquet')
oxygen_a = oxygen_a.drop(columns=['waardebewerkingsmethodeCode'])
legacy_oxygen_a = pd.read_parquet('data/HistoricalWWTPData/DTWINTERNALWWTPDATA/Oxygen Data/zuurstofA_EDE_B121069901_K600.MTW.parquet')
oxygen_a = oxygen_a.rename(columns={"hstWaarde": "hstWaarde_oxygen_a", "historianTagnummer": "historianTagnummer_oxygen_a"}).reset_index(drop=True)
oxygen_a['hstWaarde_oxygen_a'] = oxygen_a['hstWaarde_oxygen_a'].apply(pd.to_numeric, errors='coerce')

oxygen_b = pd.read_parquet('data/OxygenData2024/oxygen_b_2024.parquet')
oxygen_b = oxygen_b.drop(columns=['waardebewerkingsmethodeCode'])
legacy_oxygen_b = pd.read_parquet('data/HistoricalWWTPData/DTWINTERNALWWTPDATA/Oxygen Data/zuurstofB_EDE_B121069907_K600.MTW.parquet')
oxygen_b = oxygen_b.rename(columns={"hstWaarde": "hstWaarde_oxygen_b", "historianTagnummer": "historianTagnummer_oxygen_b"}).reset_index(drop=True)
oxygen_b['hstWaarde_oxygen_b'] = oxygen_b['hstWaarde_oxygen_b'].apply(pd.to_numeric, errors='coerce')


# In[17]:


combines_oxygen = pd.concat([oxygen_a, oxygen_b], axis=1)
combines_oxygen['Average_value'] = (combines_oxygen['hstWaarde_oxygen_a'] + combines_oxygen['hstWaarde_oxygen_b']) / 2
combines_oxygen


# In[18]:


chemicals['Average_oxygen_level'] = combines_oxygen['Average_value']
chemicals.dropna()
ObjectsForCorr = chemicals[['hstWaarde_ammonium_2023', 'hstWaarde_nitrate', 'hstWaarde_phosphate', 'Average_oxygen_level']]
corr = ObjectsForCorr.corr()
plt.figure(figsize=(10, 8))
plt.title("Correlation Heatm<ap")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.heatmap(
    corr,
    annot=True,
    annot_kws={"size": 24},       # Font size for numbers
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}  # Optional
)
plt.show()


# In[19]:


weather = pd.read_csv('weather.csv')
weather


# In[20]:


chemicals['datumBeginMeting'] = pd.to_datetime(chemicals['datumBeginMeting'])
chemicals['Hour'] = chemicals['datumBeginMeting'].dt.floor('h')
hourly_chemicals = chemicals.groupby('Hour')[[
    'hstWaarde_ammonium_2023',
    'hstWaarde_nitrate',
    'hstWaarde_phosphate'
]].mean().reset_index()

hourly_chemicals


# In[21]:


chemicals['Average_oxygen_level'] = combines_oxygen['Average_value']
chemicals.dropna()
combined = pd.concat([hourly_chemicals[['hstWaarde_ammonium_2023', 'hstWaarde_nitrate', 
                                 'hstWaarde_phosphate']], weather], axis=1)
combined.dropna()
numeric_cols = combined.select_dtypes(include='number')
corr = numeric_cols.corr()
plt.figure(figsize=(10, 8))
plt.title("Correlation Heatmap")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.heatmap(
    corr,
    annot=True,
    annot_kws={"size": 6},       # Font size for numbers
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}  # Optional
)
plt.show()


# # Modelling with one-hot encoding 

# In[22]:


combined['hstWaarde_ammonium_2023'] = combined['hstWaarde_ammonium_2023'].fillna(combined['hstWaarde_ammonium_2023'].median())
combined['hstWaarde_nitrate'] = combined['hstWaarde_nitrate'].fillna(combined['hstWaarde_nitrate'].median())
combined['hstWaarde_phosphate'] = combined['hstWaarde_phosphate'].fillna(combined['hstWaarde_phosphate'].median())
combined


# In[23]:


target = 'hstWaarde_nitrate'
X = combined.drop(columns=[target])
y = combined[target]


# In[24]:


numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()


# In[25]:


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])


# In[26]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[27]:


preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])


# In[28]:


kf = KFold(n_splits=3, shuffle=True, random_state=42)


# In[29]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])


# In[30]:


cv_scores = cross_val_score(
    model,
    X,   # already-numeric feature matrix
    y,
    cv=kf,
    scoring='r2'
)


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[35]:


print("Fold-by-fold R²:", cv_scores)
print("Average R²      :", np.mean(cv_scores))
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error MSE: ", mse)


# In[ ]:




