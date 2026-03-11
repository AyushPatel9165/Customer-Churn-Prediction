import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# load dataset
train_data = pd.read_csv(r"C:\Data Science\Project Assignments and Project Topic list\Customer-Churn-Prediction\customer_churn_dataset-training-master.csv")
test_data = pd.read_csv(r"C:\Data Science\Project Assignments and Project Topic list\Customer-Churn-Prediction\customer_churn_dataset-testing-master.csv")

# remove missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# encode categorical columns
le = LabelEncoder()

for col in ['Gender','Subscription Type','Contract Length']:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.fit_transform(test_data[col])

# training features
X_train = train_data.drop(['CustomerID','Churn'],axis=1)
y_train = train_data['Churn']

# testing features
X_test = test_data.drop(['CustomerID','Churn'],axis=1)
y_test = test_data['Churn']

# model
model = RandomForestClassifier(n_estimators=200)

# train model
model.fit(X_train,y_train)

# prediction
pred = model.predict(X_test)

# accuracy
print("Accuracy:",accuracy_score(y_test,pred))

# confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test,pred))

# save model
pickle.dump(model,open("model.pkl","wb"))

print("Model Saved Successfully")