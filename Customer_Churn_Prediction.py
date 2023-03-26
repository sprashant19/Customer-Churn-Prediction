#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#importing the data
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head()


# In[3]:


# Get information about the dataset
data.info()


# In[4]:


#Total Charges is in String format, we need to convert it into float format.
data['TotalCharges'] =pd.to_numeric(data['TotalCharges'],errors='coerce')


# In[5]:


data.info()


# In[6]:


# Check for missing values
data.isnull().sum()


# The data has no null values as the above results shows.

# In[7]:


#Drop  the unnecessary columns
data = data.drop(columns=['customerID'])
data


# # Outlier detection

# In[8]:


min_threshold,max_threshold = data.tenure.quantile([0.001,0.999])
min_threshold,max_threshold


# In[9]:


data = data[(data.tenure<68)&(data.tenure>min_threshold)]
data


# In[10]:


min_threshold,max_threshold = data.MonthlyCharges.quantile([0.001,0.999])
min_threshold,max_threshold


# # Exploratory Data Analysis

# In[11]:


#plotting Gender VS Churn
plt.figure(figsize=(10,5))
sns.countplot(x='gender',hue='Churn',data=data)
plt.show()


# The above figure depicts that number of Churn is less for males compared to females. But the number of cutomers using the connection is more than those who churn.

# In[12]:


#plotting SeniorCitizen  VS Churn
plt.figure(figsize=(10,5))
sns.countplot(x='SeniorCitizen',hue='Churn',data=data)
plt.show()


# Senior Citizen's are more likely Churn, than the young adults as the above figure depicts.

# In[13]:


#plotting Partner  VS Churn
sns.set_palette('colorblind')
plt.figure(figsize=(10,5))
sns.countplot(x='Partner',hue='Churn',data=data)
plt.show()


# Cusotmers who is married or has a partner are less likely to churn.Whereas those who don't have partners are more likely to Churn as depicted above.

# In[14]:


#plotting Dependents  VS Churn
sns.set_palette('dark')
plt.figure(figsize=(10,5))
sns.countplot(x='Dependents',hue='Churn',data=data)
plt.show()


# Cusotmers who has dependents  are less likely to churn.Whereas those who don't have dependents are more likely to Churn as depicted above.

# In[15]:


#Boxplot for tenure VS Churn
data.boxplot(column='tenure',by='Churn',figsize=(5,6))


# More than 50% of the cutomers Churn who have used the sevice for more than 30 years, whereas the cutomers who have used the service for more than 50 years are less likely to churn as depicted in the above box plot.

# In[16]:


data.boxplot(column='MonthlyCharges',by='Churn',figsize=(5,6))


# Customers whose monthly charge is more than 50, has been churning as shown above. Also those who pay a monthly charge wihtin 20- 80 are less likely churn.

# In[17]:


data.boxplot(column='TotalCharges',by='Churn',figsize=(5,6)) #Boxplot for TotalCharges


# In[18]:


#plotting PhoneService  VS Churn
sns.set_palette('deep')
plt.figure(figsize=(10,5))
sns.countplot(x='PhoneService',hue='Churn',data=data)
plt.show()


# As the above plot shows, the data has more number of people who have phone service and within this, it is shown they are less likely to churn.

# In[19]:


data.columns


# In[20]:


data.OnlineSecurity.unique()


# In[21]:


sns.set()
#Set the figure size and adjust the padding between and around the subplots.
plt.rcParams["figure.figsize"] = [15, 10] 
plt.rcParams["figure.autolayout"] = True

#Subplot with 2 rows and 3 columns
fig,axes = plt.subplots(3,4)
                                   
#Creating the subplots

sns.countplot(x='MultipleLines',hue='Churn',data=data,ax=axes[0,0])
sns.countplot(x='InternetService',hue='Churn',data=data,ax=axes[0,1])
sns.countplot(x='OnlineSecurity',hue='Churn',data=data,ax=axes[0,2])
sns.countplot(x='OnlineBackup',hue='Churn',data=data,ax=axes[0,3])
sns.countplot(x='DeviceProtection',hue='Churn',data=data,ax=axes[1,0])
sns.countplot(x='TechSupport',hue='Churn',data=data,ax=axes[1,1])
sns.countplot(x='StreamingTV',hue='Churn',data=data,ax=axes[1,2])
sns.countplot(x='StreamingMovies',hue='Churn',data=data,ax=axes[1,3])
sns.countplot(x='Contract',hue='Churn',data=data,ax=axes[2,0])
sns.countplot(x='PaperlessBilling',hue='Churn',data=data,ax=axes[2,1])
sns.countplot(x='PaymentMethod',hue='Churn',data=data,ax=axes[2,2])
sns.countplot(x='PhoneService',hue='Churn',data=data,ax=axes[2,3])


plt.show()


# # Encoding the Columns

# In[22]:


#Importing the necessary modules for encoding

from sklearn import preprocessing   
le=preprocessing.LabelEncoder()

Columns =['gender','Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

data[Columns]=data[Columns].apply(le.fit_transform)
    


# In[23]:


data['Churn'] = data['Churn'].map({'Yes':1,'No':0})


# In[24]:


data


# # Preparing the Data

# In[25]:


data.columns


# In[26]:


x = data[['gender', 'Partner', 'Dependents', 'tenure', 'PhoneService',
       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
       'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
       'SeniorCitizen', 'TotalCharges']].values
y = data['Churn'].values


# # Splitting the Data

# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# # Scaling the Data

# In[28]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train,y_train)
x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)


# # Model Building

# In[29]:


from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(x_train_scaled,y_train)


# In[30]:


#Prediction

log_reg.predict(x_test_scaled)


# In[31]:


df_pred=pd.DataFrame({'Actual':y_test,'Predicted':log_reg.predict(x_test_scaled)})
df_pred
     


# # Evaluation Metrics

# In[32]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,log_reg.predict(x_test_scaled))

