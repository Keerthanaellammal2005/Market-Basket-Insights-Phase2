import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv('/kaggle/input/market-basket-analysis/Assignment-1_Data.csv',delimiter=';')
df.head()

df.info()

df.isnull().sum()

df.loc[df['Quantity']<=0][:5]

df=df.loc[df['Quantity']>0]


df.loc[df['Price']<='0'][:5
                        ]

df=df.loc[df['Price']>'0']

df.loc[(df['Itemname']=='POSTAGE')|(df['Itemname']=='DOTCOM POSTAGE')|(df['Itemname']=='Adjust bad debt')|(df['Itemname']=='Manual')].head()

df.isnull().sum()

df=df.fillna('-')
df.isnull().sum()

df['Year']=df['Date'].apply(lambda x:x.split('.')[2])
df['Year']=df['Year'].apply(lambda x:x.split(' ')[0])
df['Month']=df['Date'].apply(lambda x:x.split('.')[1])
df.head()

df['Price']=df['Price'].str.replace(',','.').astype('float64')
df['Total price']=df.Quantity*df.Price
df.head()

df.groupby(['Year','Month'])['Total price'].sum()

df=df.loc[df['Year']!='2010']

sales=df.groupby(['Year','Month'])['Total price','Quantity'].sum()
sales.to_csv('sales.csv')
sales=pd.read_csv('sales.csv')
sales=sales.pivot_table(sales,index=['Year','Month'],aggfunc=np.sum,fill_value=0)
sales.plot(kind='bar',cmap='Set1')
plt.show()

sales_country=df.groupby(['Year','Month','Country'])['Total price'].sum()
sales_country.to_csv('sales_country.csv')
sales_country=pd.read_csv('sales_country.csv')

fig=px.bar(sales_country,x=['Month'],y='Total price',color='Country',title='Monthly sales amount in each country in 2021')
fig.update_layout(xaxis_title='Month',yaxis_title='Sales amount')
fig.show()

country=df.groupby('Country')['Total price'].sum()
country.to_csv('country.csv')
country=pd.read_csv('country.csv')

fig=px.bar(country,x='Country',y='Total price',title='Sales amount in each country in 2021')
fig.update_layout(xaxis={'categoryorder':'total descending'},yaxis_title='Sales amount')
fig.show()

cm=sns.light_palette("green",as_cmap=True)

item_sales=df.groupby('Itemname')['Price'].sum().sort_values(ascending=False)[:10]
item_sales.to_csv('item_sales.csv')
item_sales=pd.read_csv('item_sales.csv')
item_sales.style.background_gradient(cmap=cm).set_precision(2)

df[['Itemname','Quantity']].sort_values(by='Quantity',ascending=False)[:10].style.background_gradient(cmap=cm).set_precision(2)

color=plt.cm.rainbow(np.linspace(0,1,30))
df['Itemname'].value_counts().head(10).plot.bar(color=color,figsize=(6,3))
plt.xticks(rotation=90,fontsize=8)
plt.grid()
plt.show()

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df['Itemname']=df['Itemname'].str.strip()
df['BillNo']=df['BillNo'].astype('str')

basket=(df[df['Country']=='United Kingdom']
        .groupby(['BillNo','Itemname'])['Quantity']
        .sum().unstack().reset_index().fillna(0)
        .set_index('BillNo'))

basket.head(3)

def encode_units(x):
  if x<=0:
    return 0
  if x>=1:
    return 1

basket_sets=basket.applymap(encode_units)

frequent_itemsets=apriori(basket_sets,min_support=0.03,use_colnames=True)

rules=round(association_rules(frequent_itemsets,metric='lift',min_threshold=1),2)
rules.head(5)

plt.figure(figsize=(6,6))
plt.subplot(221)
sns.scatterplot(x="support",y="confidence",data=rules,hue="lift",palette="viridis")
plt.subplot(222)
sns.scatterplot(x="support",y="lift",data=rules,hue="confidence",palette="viridis")
plt.subplot(223)
sns.scatterplot(x="confidence",y="lift",data=rules,hue='support',palette="viridis")
plt.subplot(224)
sns.scatterplot(x="antecedent support",y="consequent support",data=rules,hue='confidence',palette="viridis")
plt.tight_layout()
plt.show()

rules[['antecedents','consequents','support']].sort_values('support',ascending=False)[:5].style.background_gradient(cmap=cm).set_precision(2)

rules[['antecedents','consequents','confidence']].sort_values('confidence',ascending=False)[:5].style.background_gradient(cmap=cm).set_precision(2)

rules[['antecedents','consequents','lift']].sort_values('lift',ascending=False)[:5].style.background_gradient(cmap=cm).set_precision(2)


rules[(rules['lift']>=13)&(rules['confidence']>=0.7)].sort_values('lift',ascending=False).style.background_gradient(cmap=cm).set_precision(2)

