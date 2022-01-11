import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

"""Population and Sample"""
# Create a Population DataFrame with 10 data 

data = pd.DataFrame()
data['Population'] = [47, 48, 85, 20, 19, 13, 72, 16, 50, 60]

# Draw sample with replacement, size=5 from Population
a_sample_with_replacement = data['Population'].sample(5, replace=True)
print(a_sample_with_replacement)

# Draw sample without replacement, size=5 from Population
a_sample_without_replacement = data['Population'].sample(5, replace=False)
print(a_sample_without_replacement)

# Calculate mean and variance
population_mean = data['Population'].mean()
population_var = data['Population'].var()
print('Population mean is ', population_mean)
print('Population variance is', population_var)

# Calculate sample mean and sample standard deviation, size =10
# You will get different mean and varince every time when you excecute the below code
a_sample = data['Population'].sample(10, replace=True)
sample_mean = a_sample.mean()
sample_var = a_sample.var()
print('Sample mean is ', sample_mean)
print('Sample variance is', sample_var)

sample_length = 500
sample_variance_collection=[data['Population'].sample(10, replace=True).var(ddof=1) for i in range(sample_length)]
print(sample_variance_collection)

"""Variation of Sample"""
# Sample mean and SD keep changing, but always within a certain range
Fstsample = pd.DataFrame(np.random.normal(10, 5, size=30))
print('sample mean is ', Fstsample[0].mean())
print('sample SD is ', Fstsample[0].std(ddof=1))

meanlist = []
for t in range(10000):
    sample = pd.DataFrame(np.random.normal(10, 5, size=30))
    meanlist.append(sample[0].mean())

collection = pd.DataFrame()
collection['meanlist'] = meanlist

collection['meanlist'].hist(bins=100, normed=1,figsize=(15,8))

# See what central limit theorem tells you...the sample size is larger enough, 
# the distribution of sample mean is approximately normal
# apop is not normal, but try to change the sample size from 100 to a larger number. The distribution of sample mean of apop 
# becomes normal.
sample_size = 100
samplemeanlist = []
apop =  pd.DataFrame([1, 0, 1, 0, 1])
for t in range(10000):
    sample = apop[0].sample(sample_size, replace=True)  # small sample size
    samplemeanlist.append(sample.mean())

acollec = pd.DataFrame()
acollec['meanlist'] = samplemeanlist
acollec.hist(bins=100, normed=1,figsize=(15,8))

"""Confidence Interval"""
ms = pd.read_csv('microsoft.csv')
ms.head()

# we will use log return for average stock return of Microsoft
ms['logReturn'] = np.log(ms['Close'].shift(-1)) - np.log(ms['Close'])

# Lets build 90% confidence interval for log return
sample_size = ms['logReturn'].shape[0]
sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof=1) / sample_size**0.5

# left and right quantile
z_left = norm.ppf(0.05)
z_right = norm.ppf(0.95)

# upper and lower bound
interval_left = sample_mean + z_left*sample_std
interval_right = sample_mean + z_right*sample_std

# 90% confidence interval tells you that there will be 90% chance that the average stock return lies between "interval_left"
# and "interval_right".
print('90% confidence interval is ', (interval_left, interval_right))


"""Hypothesis Testing"""
# Log return goes up and down during the period
ms['logReturn'].plot(figsize=(20, 8))
plt.axhline(0, color='red')
plt.show()

sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof=1)
n = ms['logReturn'].shape[0]

# if sample size n is large enough, we can use z-distribution, instead of t-distribtuion
# mu = 0 under the null hypothesis
zhat = (sample_mean - 0)/(sample_std/n**0.5)
print(zhat)

# confidence level
alpha = 0.05

zleft = norm.ppf(alpha/2, 0, 1)
zright = -zleft  # z-distribution is symmetric 
print(zleft, zright)

print('At significant level of {}, shall we reject: {}'.format(alpha, zhat>zright or zhat<zleft))

# step 2
sample_mean = ms['logReturn'].mean()
sample_std = ms['logReturn'].std(ddof=1)
n = ms['logReturn'].shape[0]

# if sample size n is large enough, we can use z-distribution, instead of t-distribtuion
# mu = 0 under the null hypothesis
zhat = (sample_mean-0)/(sample_std/(n**0.5))
print(zhat)

# step 3
alpha = 0.05

zright = norm.ppf(1-alpha, 0, 1)
print(zright)

# step 4
print('At significant level of {}, shall we reject: {}'.format(alpha, zhat>zright))

# step 3 (p-value)
p = 1 - norm.cdf(zhat, 0, 1)
print(p)

# step 4
print('At significant level of {}, shall we reject: {}'.format(alpha, p < alpha))
