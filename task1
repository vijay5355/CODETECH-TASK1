/* It starts with a dataset of your choice and perform EDA using libraries like pandas, numpy,
and matplotlib or seaborn. Explore the data's characteristics, distributions, correlations,
and outliers. Visualize your findings with histograms, scatter plots, and heatmaps to
gain insights into the data */

import pandas as pd
url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
titanic_df = pd.read_csv(url)
titanic_df.head()

titanic_df.describe(include='all')

missing_values = titanic_df.isnull().sum()
missing_values

import seaborn as sns
import matplotlib.pyplot as plt
numerical_features = ['Age', 'Fare', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']
plt.figure(figsize=(10, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(titanic_df[feature].dropna(), kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()
categorical_features = ['Pclass', 'Sex', 'Survived']
plt.figure(figsize=(10, 6))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=feature, data=titanic_df)
    plt.title(f'Count of {feature}')
plt.tight_layout()
plt.show()

numerical_df = titanic_df[['Age', 'Fare', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']]
correlation_matrix = numerical_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=titanic_df, palette='coolwarm')
plt.title('Age vs. Fare by Survival Status')
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', hue='Survived', data=titanic_df)
plt.title('Age Distribution by Pclass and Survival Status')
plt.show()
