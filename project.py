import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv("New York Airbnb_4 dec 2021.csv")

print(data.shape)
print(data.head(3))
print(data.columns.values)
print(data.isnull().sum())

data = data.drop(["id","name","host_id","host_name","license"],axis=1)
print(data.dtypes)

# data cleaning in "reviews_per_month"
data["reviews_per_month"].fillna(data["reviews_per_month"].mean(), inplace=True)
print(data.isnull().sum())

# data cleaning in "last_review"
data["last_review"] = pd.to_datetime(data["last_review"]) 
data["last_day_review"] = data["last_review"].dt.day
data["last_month_review"] = data["last_review"].dt.month
data["last_year_review"] = data["last_review"].dt.year
data = data.drop("last_review",axis=1)
data["last_day_review"].fillna(0, inplace=True)
data["last_month_review"].fillna(0, inplace=True)
data["last_year_review"].fillna(data["last_year_review"].mean(), inplace=True)
print(data.dtypes)
print(data.isnull().sum())

# LabelEncoder
La = LabelEncoder()

# data cleaning in "neighbourhood_group"
print(data["neighbourhood_group"].value_counts())
data["neighbourhood_group"] = La.fit_transform(data["neighbourhood_group"])
print(data["neighbourhood_group"].value_counts())

# data cleaning in "neighbourhood"
print(data["neighbourhood"].value_counts())
print(data["neighbourhood"].nunique())
data["neighbourhood"] = La.fit_transform(data["neighbourhood"])
print(data["neighbourhood"].value_counts())

# data cleaning in "room_type"
print(data["room_type"].value_counts())
data["room_type"] = data["room_type"].map({"Entire home/apt":1,"Private room":2,"Shared room":3,
                                           "Hotel room":4})


# data heatmap
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(),annot=True)
plt.show()

# pairplot to data
# sns.pairplot(data)
# plt.show()

# distplot to price
plt.figure(figsize=(14,7))
sns.distplot(data["price"])
plt.show()

# countplot "last_day_review"
plt.figure(figsize=(14,7))
sns.countplot(data["last_day_review"])
plt.show()

# countplot "last_month_review"
plt.figure(figsize=(14,7))
sns.countplot(data["last_month_review"])
plt.show()

# countplot "room_type"
plt.figure(figsize=(14,7))
sns.countplot(data["room_type"])
plt.show()

# regplot between "number_of_reviews" & "price"
plt.figure(figsize=(14,7))
sns.regplot(data["number_of_reviews"],data["price"], marker="+")
plt.show()

# chak data again
print(data.dtypes)
print(data.isnull().sum())

# x & y
x = data.drop("price", axis=1)
y = data["price"]
print(x.shape)
print(y.shape)

# StandardScaler to x 
ss = StandardScaler()
x = ss.fit_transform(x)
print(x[:5])

# data spliting
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,shuffle =True)
print(X_train.shape)
print(y_train.shape)

Lo = LinearRegression()
Lo.fit(X_train, y_train)

print("*"*100)
print(Lo.score(X_train, y_train))
print(Lo.score(X_test, y_test))
print("*"*100)


print("_"*150)
for x in range(2,20):
    Dt = DecisionTreeRegressor(max_depth=x,random_state=33)
    Dt.fit(X_train, y_train)

    print("x = ", x)
    print(Dt.score(X_train, y_train))
    print(Dt.score(X_test, y_test))
    print("*"*100)

for x in range(3,100):
    KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors = x, weights='uniform')
    KNeighborsRegressorModel.fit(X_train, y_train)

    KNeighborsRegressorModel.fit(X_train, y_train)

    print("* "*50,x," *"*50)
    print(KNeighborsRegressorModel.score(X_train, y_train))
    print(KNeighborsRegressorModel.score(X_test, y_test))

# In these data,
# it is difficult to predict the price because
# the relationship between
# the input and the price is not strong,
# and therefore it is difficult to use any model with it.