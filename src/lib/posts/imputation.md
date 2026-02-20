---
title: "Imputation Using Random Sampling and K-Nearest Neighbors"
date: "2026-02-13"
updated: "2026-02-20"
categories:
  - "missing data"
  - "imputation"
coverImage: "/images/RMS_Titanic_3.jpg"
excerpt: We explore two methods replacing missing data.
---
In machine learning, imputation refers to the creation of synthetic data from existing data for the purpose of filling missing data. Missing data are any NaN or null cells in the dataframe. Missing data is to be avoided as it can be problematic for training machine learning models.

The first method of imputation described in this post is designed for categorical data. If the feature you want to impute is continuous, then you'll want to use the imputation functions built into scikit-learn, as detailed later in this post.

Because it contains categorical features, we'll be using the Titanic dataset hosted on OpenML to demonstrate.


```python
from sklearn.datasets import fetch_openml
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
import random
```


```python
random.seed(1)

# Fetch the Titanic dataset
data = fetch_openml(data_id=40945, as_frame=True)
df = data.frame
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>survived</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>boat</th>
      <th>body</th>
      <th>home.dest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Allen, Miss. Elisabeth Walton</td>
      <td>female</td>
      <td>29.0000</td>
      <td>0</td>
      <td>0</td>
      <td>24160</td>
      <td>211.3375</td>
      <td>B5</td>
      <td>S</td>
      <td>2</td>
      <td>NaN</td>
      <td>St Louis, MO</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Allison, Master. Hudson Trevor</td>
      <td>male</td>
      <td>0.9167</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>11</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>Allison, Miss. Helen Loraine</td>
      <td>female</td>
      <td>2.0000</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>Allison, Mr. Hudson Joshua Creighton</td>
      <td>male</td>
      <td>30.0000</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>NaN</td>
      <td>135.0</td>
      <td>Montreal, PQ / Chesterville, ON</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td>
      <td>female</td>
      <td>25.0000</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Montreal, PQ / Chesterville, ON</td>
    </tr>
  </tbody>
</table>
</div>



We shuffle the DataFrame for the purpose of randomly deleting values.

The call to `reset_index` is necessary because, otherwise, our later slicing of the DataFrame will take all elements up to the *nth* index instead of subsetting up to the *nth* row.


```python
# shuffled dataframe
sdf = df.sample(frac=1).reset_index(drop=True)
```


```python
sdf['sex'].value_counts()
```




    sex
    male      843
    female    466
    Name: count, dtype: int64



Because there are a large number of values in each of the categories of these columns, it will be impossible for the below code to delete all of any category.


```python
NUM_DELETE = 100
sdf.loc[:NUM_DELETE - 1, ['sex']] = None
sdf['sex'].value_counts()
```




    sex
    male      791
    female    418
    Name: count, dtype: int64



In a previous post, we talked about the Mind-Reading Machine, which uses a Markov Chain. A Markov Chain depends only on the previous state.

This method of imputation does not depend on the previous state, and therefore not capable of being considered a Markov Chain. It is, however, similar to the Mind-Reading Machine in that we will choose at random from an array, thus having the probability of choosing each unique element in proportion to how frequently it shows up in the array.


```python
sexes = sdf[~sdf['sex'].isna()]['sex']
choices = random.choices(sexes.array, k=NUM_DELETE)
sdf.loc[:NUM_DELETE - 1, ['sex']] = choices
sdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>survived</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>boat</th>
      <th>body</th>
      <th>home.dest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>Youseff, Mr. Gerious</td>
      <td>male</td>
      <td>45.5</td>
      <td>0</td>
      <td>0</td>
      <td>2628</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
      <td>NaN</td>
      <td>312.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>Candee, Mrs. Edward (Helen Churchill Hungerford)</td>
      <td>male</td>
      <td>53.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17606</td>
      <td>27.4458</td>
      <td>NaN</td>
      <td>C</td>
      <td>6</td>
      <td>NaN</td>
      <td>Washington, DC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>Olsson, Mr. Oscar Wilhelm</td>
      <td>female</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>347079</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>A</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>Theobald, Mr. Thomas Leonard</td>
      <td>female</td>
      <td>34.0</td>
      <td>0</td>
      <td>0</td>
      <td>363294</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
      <td>176.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>Svensson, Mr. Johan</td>
      <td>female</td>
      <td>74.0</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



In the original dataset, some of the age values are missing. Fortunately, scikit-learn contains convenient means of imputing data, including numerical data.

First, we encode sex as elements of the set $\{0, 1\}$, because

1. Although it is unlikely to me that sex will predict the age, I want to demonstrate encoding.

2. This feature is currently categorical.

3. The K-Nearest Neighbors imputer (AKA `KNNImputer`) requires that the input be numerical.

(Scikit-Learn)


```python
# Encode the categorical labels
sdf['male'] = pd.get_dummies(sdf['sex'])['male']
```

As mentioned, KNNImputer only wants numeric types. We will therefore provide ourselves with a means of selecting only the numeric columns from the dataframe.


```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
selected_columns = sdf.select_dtypes(include=numerics).columns
selected_columns
```




    Index(['pclass', 'age', 'sibsp', 'parch', 'fare', 'body'], dtype='object')



A high-level of interpretation of the `fit_transform` method of `KNNImputer` is as follows:

- For each row, for each cell, if that cell is missing, do nothing. Otherwise, proceed to the next step.
- Create a value for that cell using the K-Nearest Neighbor algorithm. This uses the other cells in that row and in the neighbors to predict this one.


```python
imp = KNNImputer().set_output(transform='pandas')
transformed = imp.fit_transform(sdf[selected_columns])
```


```python
sdf[selected_columns] = transformed
sdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pclass</th>
      <th>survived</th>
      <th>name</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>ticket</th>
      <th>fare</th>
      <th>cabin</th>
      <th>embarked</th>
      <th>boat</th>
      <th>body</th>
      <th>home.dest</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>0</td>
      <td>Youseff, Mr. Gerious</td>
      <td>male</td>
      <td>45.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2628</td>
      <td>7.2250</td>
      <td>NaN</td>
      <td>C</td>
      <td>NaN</td>
      <td>312.0</td>
      <td>NaN</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1</td>
      <td>Candee, Mrs. Edward (Helen Churchill Hungerford)</td>
      <td>male</td>
      <td>53.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>PC 17606</td>
      <td>27.4458</td>
      <td>NaN</td>
      <td>C</td>
      <td>6</td>
      <td>177.2</td>
      <td>Washington, DC</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1</td>
      <td>Olsson, Mr. Oscar Wilhelm</td>
      <td>female</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>347079</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>A</td>
      <td>117.4</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0</td>
      <td>Theobald, Mr. Thomas Leonard</td>
      <td>female</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>363294</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
      <td>176.0</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>0</td>
      <td>Svensson, Mr. Johan</td>
      <td>female</td>
      <td>74.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>347060</td>
      <td>7.7750</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
      <td>167.6</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



The below tells us that the only columns that have missing values are the ones that we didn't intend to impute.


```python
sdf.isna().sum()
```




    pclass          0
    survived        0
    name            0
    sex             0
    age             0
    sibsp           0
    parch           0
    ticket          0
    fare            0
    cabin        1014
    embarked        2
    boat          823
    body            0
    home.dest     564
    male            0
    dtype: int64



There are scenarios in which deleting rows or columns containing missing values is acceptable. Whether and how we do so depends on a number of factors, including the number of missing values and why those values are missing (if we can know the reason).

An extended discussion of those facets are outside the scope in this post. Instead, I'll leave the reader with the following takeaways.

### To Sum Up

- We showed how to replace missing values solely by randomly selecting from the distribution of those existing values.

- We showed how to impute missing numerical values using a built-in scikit-learn method.

### Future Work

In a future post, we'll explore a method of training a machine learning model without imputing missing data and compare it with imputation.

## Bibliography

“7.4. Imputation of Missing Values.” Scikit-Learn, https://scikit-learn/stable/modules/impute.html. Accessed 13 Feb. 2026.

OpenML. https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=Titanic&id=40945. Accessed 13 Feb. 2026.
