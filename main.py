import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('/home/adimu/Downloads/bank+marketing/bank/bank.csv', sep=";")

## Label-encoding 

label_encoder = preprocessing.LabelEncoder()
df['job'] = label_encoder.fit_transform(df['job'])
df['marital'] = label_encoder.fit_transform(df['marital'])
df['education'] = label_encoder.fit_transform(df['education'])
df['default'] = label_encoder.fit_transform(df['default'])
df['housing'] = label_encoder.fit_transform(df['housing'])
df['loan'] = label_encoder.fit_transform(df['loan'])
df['contact'] = label_encoder.fit_transform(df['contact'])
df['month'] = label_encoder.fit_transform(df['month'])
df['poutcome'] = label_encoder.fit_transform(df['poutcome'])
df['y'] = label_encoder.fit_transform(df['y'])

X = df.drop("y", axis=1)
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Accuracy using label encoding: ")
print(classification_report(y_test, model.predict(X_test)))


## One hot encoding
one_hot_encoded_data = preprocessing.OneHotEncoder(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])

X = one_hot_encoded_data.drop("y", axis=1)
y = one_hot_encoded_data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Accuracy using one hot encoding: ")
print(classification_report(y_test, model.predict(X_test)))

## Test-cases

# Test-case 1
# Label encoding: 0.8888888888888888
# One-hot encoding: 0.8911000552791598

# Test-case 2
# Label encoding: 0.8855721393034826
# One-hot encoding: 0.892758430071863

# Test-case 3
# Label encoding: 0.8872305140961857
# One-hot encoding: 0.8999447208402432

# Conclusion
# According to this dataset one-hot encoding is giving more accurate result rather than label encoding.
