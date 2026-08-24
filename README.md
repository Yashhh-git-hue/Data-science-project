
# project is about titanic people survived or not

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv(r"C:\Users\user\Desktop\Final project\Titanic-Dataset.csv")
df.head()
df.shape


df.info()
# Statistical summary

df.describe()
# Missing values count
df.isnull().sum()
df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop("Cabin", axis=1, inplace=True)
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Gender Survival")
plt.show()
sns.histplot(df["Age"], bins=30)
plt.title("Age Distribution")
plt.show()
df.drop(columns=["PassengerId", "Name", "Ticket"], inplace=True, errors="ignore")
df.head()
encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])
df["Embarked"] = encoder.fit_transform(df["Embarked"])
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
cm = confusion_matrix(y_test,y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(classification_report(y_test,y_pred))

new_passenger = [[3,1,25,0,0,7.25,2]]
new_passenger = scaler.transform(new_passenger)
prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")
