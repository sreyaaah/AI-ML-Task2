import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv('Titanic-Dataset.csv')

print("First 5 rows of the dataset:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())
print("\nDataset Information:")
print(df.info())

plt.figure(figsize=(6, 4))
sns.histplot(df['Age'].dropna(), kde=True, bins=30, color='royalblue')
plt.title('Age Distribution', fontweight='bold')
plt.xlabel('Age', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Fare'], color='darkorange')
plt.title('Fare Boxplot', fontweight='bold')
plt.xlabel('Fare', fontweight='bold')
plt.show()

numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"weight": "bold"})
plt.title('Correlation Matrix (Numeric Features Only)', fontweight='bold')
plt.show()

sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']].dropna(), hue='Survived', palette='husl')
plt.suptitle('Pairplot of Age, Fare, Pclass vs Survived', y=1.02)
plt.show()

sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Gender', fontweight='bold')
plt.xlabel('Sex', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Pastel1')
plt.title('Survival by Passenger Class', fontweight='bold')
plt.xlabel('Pclass', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.legend(title='Survived', title_fontsize='13', fontsize='11')
plt.show()

fig = px.histogram(df, x='Age', color='Survived', nbins=30,
                   color_discrete_map={0: 'goldenrod', 1: 'royalblue'},
                   title="Age vs Survived")
fig.show()
