import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score


train_x = []
normal = pd.read_csv("normal_point.csv")
fall = pd.read_csv("fall_point.csv")
print(normal.values.shape)
print(fall.values.shape)
res = pd.concat([normal, fall])
print(res)
data = res.iloc[:, 1:]
target = res.iloc[:, 0]
print(data.values)
print(target.values)
X_train, X_test, y_train, y_test = train_test_split(data.values, target.values, test_size=0.3, random_state=0)
print(X_train.shape)
print(y_train.shape)
pose = KNN(n_neighbors=3, algorithm='auto')
pose.fit(X_train, y_train)
predict_result = pose.predict(X_test)
accuracy = accuracy_score(y_test, predict_result)
print("Test set accuracy: {:.2f}".format(accuracy))

# print(predict_result)
# print(y_test)
joblib.dump(pose, "Model/PoseKeypoint.joblib")
