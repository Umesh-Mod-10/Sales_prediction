# %% Import the important libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xg
from sklearn.metrics import r2_score

# %% Importing the dataset:

data_full = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\Umesh\Data Analysis\Predict Future Sales\sales_train.csv")
shop = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\Umesh\Data Analysis\Predict Future Sales\shops.csv")
item = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\Umesh\Data Analysis\Predict Future Sales\items.csv")
item_cat = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\Umesh\Data Analysis\Predict Future Sales\item_categories.csv")

test = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\Umesh\Data Analysis\Predict Future Sales\test.csv")

# %% Forming the dataset for training from multiple datasets:

data_month = pd.DataFrame(data_full.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'].sum())
data_month = data_month.rename({'item_cnt_day': 'item_cnt_month'}, axis=1)
data_price = pd.DataFrame(data_full.groupby(['date_block_num', 'shop_id', 'item_id'])["item_price"].min())
data = data_month.join(data_price, on=["date_block_num","shop_id","item_id"])
data = data.reset_index(level=[0,1,2])

# %% Getting the rest of the data from other datasets:

# Item Dataset:

data = data.join(item, on='item_id', rsuffix='_item')
data = data.drop(columns='item_id_item', axis=1)

test = test.join(item, on='item_id', rsuffix='_item')
test = test.drop(columns='item_id_item', axis=1)

# %% Merging the remaining dataset to form the best training set:

# Shop Dataset:

shop['location'] = shop['shop_name'].apply(lambda x: x.split()[0])
shop['location'] = shop['location'].replace({'!Якутск': 'Якутск'})

# %% Getting the shop details to join:

data = data.join(shop, on=['shop_id'], rsuffix="_ITEMS")
data = data.drop(columns="shop_id_ITEMS", axis=1)
test = test.join(shop, on=['shop_id'], rsuffix="_ITEMS")
test = test.drop(columns="shop_id_ITEMS", axis=1)

# %% Getting the price value:

df_1 = pd.DataFrame(data.groupby("item_category_id").item_price.mean())
df_2 = pd.DataFrame(data.groupby("shop_id").item_price.mean())

df_2 = df_2.rename(columns={"item_price":"price_by_shop"})
test = test.join(df_1, on="item_category_id")
test = test.join(df_2, on="shop_id")
data = data.join(df_2, on="shop_id")

# %% Getting the basic details of the dataset:

# Train Dataset:

print(data.info())
print(data.describe())
print(data.shape)
print(data.isna().sum()[data.isna().sum() != 0])
print(data.duplicated().sum())

# %% Test Dataset:

print(test.info())
print(test.describe())
print(test.shape)
print(test.isna().sum()[test.isna().sum() != 0])
print(test.duplicated().sum())

# %% Checking the data spread of each column:

sn.boxplot(data=data['item_cnt_month'], color='lightgreen')
plt.show()

sn.boxplot(data=data['item_price'], color='lightgreen')
plt.show()

sn.boxplot(data=data['price_by_shop'], color='lightgreen')
plt.show()

# %% Getting the Correlation between the Numerical data:

corr_data = data.select_dtypes(exclude='object')
corr_data_analysis = corr_data.corr()

# Visualize this:

plt.figure(figsize=(13,8))
sn.color_palette("mako", as_cmap=True)
sn.heatmap(data = corr_data_analysis, edgecolor='black')
plt.xticks(text='bold')
plt.show()

# %% Getting the hidden insights of the dataset:

# Price per each shop name

data_price_shop = (data.groupby('shop_name')['price_by_shop'].sum()).sort_values(ascending=False)
data_price_shop = pd.DataFrame(data_price_shop)
data_price_shop = data_price_shop.reset_index(level=[0])

# Visualize by graphs:

plt.figure(figsize=(18,17))
sn.barplot(data = data_price_shop, y = 'price_by_shop', x = 'shop_name', color='purple', alpha=0.5, edgecolor='black')
plt.xticks(rotation=90, text='bold')
plt.title("Price of Items depending on shops")
plt.show()

# %% Count of shops in each location:

data_item_location = (data.groupby('location')['item_cnt_month'].sum()).sort_values(ascending=False)
data_item_location = pd.DataFrame(data_item_location)
data_item_location = data_item_location.reset_index(level=[0])

# Visualize by graphs:

plt.figure(figsize=(18,17))
sn.barplot(data = data_item_location, y = 'item_cnt_month', x = 'location', color='purple', alpha=0.5, edgecolor='black')
plt.xticks(rotation=90, text='bold')
plt.title("Price of Items depending on location")
plt.show()

# %% Count of shops in each location:

data_item_shop = (data.groupby('shop_name')['item_cnt_month'].sum()).sort_values(ascending=False)
data_item_shop = pd.DataFrame(data_item_shop)
data_item_shop = data_item_shop.reset_index(level=[0])

# Visualize by graphs:

plt.figure(figsize=(18,17))
sn.barplot(data = data_item_shop, y = 'item_cnt_month', x = 'shop_name', color='purple', alpha=0.5, edgecolor='black')
plt.xticks(rotation=90, text='bold')
plt.title("Price of Items depending on shop")
plt.show()

# %% Item Quantity per Month:

data_item_month = pd.DataFrame((data.groupby('date_block_num')['item_cnt_month'].sum()).sort_values(ascending=False)).reset_index(level=[0])

# Visual Representation:

sn.lineplot(data = data_item_month, y = 'item_cnt_month', x = 'date_block_num', color='purple')
plt.title("Quantity of Items depending on day")
plt.show()

# %% Item Price per Month:

data_price_month = pd.DataFrame((data.groupby('date_block_num')['item_price'].sum()).sort_values(ascending=False)).reset_index(level=[0])

# Visual Representation:

sn.lineplot(data = data_price_month, x = 'date_block_num', y = 'item_price', color='red')
plt.title("Price of Items depending on day")
plt.show()

# %% Getting the Trend of item based:
'''
data_item = pd.DataFrame((data.groupby('item_name')['item_name'].count()).sort_values(ascending=False))
data_item = data_item.rename({'item_name': 'count_items'}, axis=1)
data_item = data_item.reset_index(level=[0])

# Visualize it:

sn.barplot(data=data_item, x = 'item_name', y = 'count_items', color='red')
plt.xticks(rotation=90)
plt.show()
'''

# %% Feature Engineering:

# Creating a column with positive outbreak or negative outbreak:

#data['outbreak'] = data['item_cnt_month'].apply(lambda x: True if x>0 else False)
#test['outbreak'] = test['item_cnt_month'].apply(lambda x: True if x>0 else False)

#data['below_avg'] = data['item_cnt_month'].apply(lambda x: True if x >= data['item_cnt_month'].mean() else False)

# %% Splitting the data:

X_train = data.drop(columns=['item_cnt_month', 'item_name', 'date_block_num'], axis=1)
y_train = pd.DataFrame(data['item_cnt_month'])
X_test = test.drop(columns=['item_name'], axis=1)

# %%

print(X_train.info())
print(X_test.info())

X_test = X_test.drop(columns='ID', axis=1)


# %% MinMax Scalling of Continous data:

X_num = ['shop_id', 'item_id', 'item_category_id', 'item_price', 'price_by_shop']
mx = MinMaxScaler()

y_train = mx.fit_transform(y_train[['item_cnt_month']])

for i in X_num:
    X_train[i] = mx.fit_transform(X_train[[i]])
    X_test[i] = mx.fit_transform(X_test[[i]])

# %% Label Encoding for categorical data:

'''
X_con = X.select_dtypes(include='object')
oh = OneHotEncoder()

#print(X_con['item_name'].nunique())
print(X_con['shop_name'].nunique())
print(X_con['location'].nunique())
'''

combined_data = pd.concat([X_train, X_test], axis=0)

# Perform one-hot encoding
combined_data_encoded = pd.get_dummies(combined_data)

# Separate the datasets back
X_train = combined_data_encoded.iloc[:len(X_train)]
X_test = combined_data_encoded.iloc[len(X_train):]

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

#X_train = X_train.drop(columns=['outbreak'], axis=1)
#X_test = X_test.drop(columns=['outbreak'], axis=1)

# %%

print(X_train.isna().sum())

# %% Splitting into Training and Testing Dataset:

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''
X_train = X_train.astype('int64')
X_test = X_test.astype('int64')
y_train = y_train.astype('int64')
'''
 # %%

from sklearn.metrics import r2_score

# Model Training and Prediction
xgb_r = xg.XGBRegressor(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb_r.fit(X_train, y_train)

# %%

Y_predict = xgb_r.predict(X_test)
#xbr1 = r2_score(X_train, y_train)
#print(xbr1)

# %%

Y_predict = Y_predict.round()
print(Y_predict.max())

# %%

final = pd.DataFrame({'ID':test['ID'], 'item_cnt_month':Y_predict})
final.to_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Sub.csv", index_label=True)



