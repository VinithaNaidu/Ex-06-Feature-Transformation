# Ex-06-Feature-Transformation
# AIM :
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION :
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM :
# STEP 1:
Read the given Data

# STEP 2:
Clean the Data Set using Data Cleaning Process

# STEP 3:
Apply Feature Transformation techniques to all the features of the data set

# STEP 4:
Print the transformed features

# PPROGRAM:
```
NAME : D.VINITHA NAIDU
REG NO. 212222230175
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```

# OUTPUT:
![image](https://user-images.githubusercontent.com/121166004/233825869-970c641e-7b7f-45d3-975d-c77295edc919.png)
![image](https://user-images.githubusercontent.com/121166004/233825880-bc6416a5-c39d-4dc5-bc0e-00be7db6a4a1.png)
![image](https://user-images.githubusercontent.com/121166004/233825887-b1df10d5-40c6-4c54-8a4a-679da64a85c6.png)
![image](https://user-images.githubusercontent.com/121166004/233826335-4dc57552-ba17-4b21-bc95-fb0d67990eda.png)
![image](https://user-images.githubusercontent.com/121166004/233826394-a54ac321-e550-468a-ad8b-b933e43f7460.png)
![image](https://user-images.githubusercontent.com/121166004/233826424-7f2e69b3-f1e3-40c5-8b2e-9309407c1f08.png)
![image](https://user-images.githubusercontent.com/121166004/233826435-bda9402e-c8b4-4059-bef9-46a6d24e50ad.png)
![image](https://user-images.githubusercontent.com/121166004/233826468-b73692bd-9806-4bcf-b975-653b90d91e28.png)
![image](https://user-images.githubusercontent.com/121166004/233826476-4fd7f84f-456a-4ebe-bb6e-8fae60b9cc3c.png)
![image](https://user-images.githubusercontent.com/121166004/233826486-e14bcc77-767b-4a03-8206-02d5c676df96.png)
RESULT:
Thus feature transformation is done for the given dataset.









