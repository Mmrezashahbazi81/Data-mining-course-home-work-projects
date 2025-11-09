import pandas as pd
import numpy as np

df = pd.read_csv("Covid Data.csv")
df.PREGNANT = df.PREGNANT.replace(97,2)
df = df[(df.PREGNANT == 1) | (df.PREGNANT == 2)]

df['CLASIFFICATION_FINAL'] = df['CLASIFFICATION_FINAL'].replace([1, 2, 3], 2).replace([4, 5, 6, 7], 1)
df = df.drop(columns=["DATE_DIED"])
df = df.drop(columns=["INTUBED"])
df = df.drop(columns=["ICU"])
df = df[(df.PNEUMONIA == 1) | (df.PNEUMONIA == 2)]
df = df[(df.DIABETES == 1) | (df.DIABETES == 2)]
df = df[(df.COPD == 1) | (df.COPD == 2)]
df = df[(df.ASTHMA == 1) | (df.ASTHMA == 2)]
df = df[(df.INMSUPR == 1) | (df.INMSUPR == 2)]
df = df[(df.HIPERTENSION == 1) | (df.HIPERTENSION == 2)]
df = df[(df.OTHER_DISEASE == 1) | (df.OTHER_DISEASE == 2)]
df = df[(df.CARDIOVASCULAR == 1) | (df.CARDIOVASCULAR == 2)]
df = df[(df.OBESITY == 1) | (df.OBESITY == 2)]
df = df[(df.RENAL_CHRONIC == 1) | (df.RENAL_CHRONIC == 2)]
df = df[(df.TOBACCO == 1) | (df.TOBACCO == 2)]

df.to_csv('newfile.csv', index=False)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Calculate Gini Impurity for a single node
def gini_impurity_node(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / counts.sum()
    return 1 - np.sum(prob ** 2)

# Calculate Gini Gain for a split
def gini_gain(y, y_left, y_right):
    parent_impurity = gini_impurity_node(y)
    left_impurity = gini_impurity_node(y_left)
    right_impurity = gini_impurity_node(y_right)
    # TODO
    proportions = [len(y_left) / len(y), len(y_right) / len(y)]
    impurities = [left_impurity, right_impurity]
    weighted_impurity = sum(p * i for p, i in zip(proportions, impurities))
    return parent_impurity - weighted_impurity

# Split dataset based on a given attribute value
def split(X, y, feature_index, threshold):
    left = X[:, feature_index] <= threshold
    right = X[:, feature_index] > threshold
    return X[left], y[left], X[right], y[right]

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.leaf = True

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=gini_impurity_node(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )
        if depth < self.max_depth and len(set(y)) > 1:
            best_gini_gain = 0
            best_index, best_threshold = None, None
            for feature_index in range(X.shape[1]):
                #TODO: thresholds =  Done!
                thresholds = np.array(list(set(X[:, feature_index])))
                for threshold in thresholds:
                    X_left, y_left, X_right, y_right = split(X, y, feature_index, threshold)
                    if len(y_left) > 0 and len(y_right) > 0:
                        gain = gini_gain(y, y_left, y_right)
                        if gain > best_gini_gain:
                            # TODO Done!
                            best_gini_gain = gain
                            best_index = feature_index
                            best_threshold = threshold
                            
            if best_gini_gain > 0:
                node.feature_index = best_index
                node.threshold = best_threshold
                X_left, y_left, X_right, y_right = split(X, y, best_index, best_threshold)
                node.leaf = False
                node.left = self.build_tree(X_left, y_left, depth + 1)
                node.right = self.build_tree(X_right, y_right, depth + 1)
        return node

    def predict(self, X):
        return [self.predict_single_input(inputs) for inputs in X]

    def predict_single_input(self, inputs):
        node = self.tree
        while not node.leaf:
            # TODO
            threshold_check = inputs[node.feature_index] <= node.threshold
            node = node.left if threshold_check else node.right
        return node.predicted_class



#X = np.array([[2, 3], [1, 1], [2, 1], [5, 3], [5, 6], [4, 4], [6, 6]])
#y = np.array([0, 0, 0, 1, 1, 1, 1])
#test_points = np.array([[2, 2], [5, 5]])


data = pd.read_csv('newfile.csv')
X = data.drop(columns=['CLASIFFICATION_FINAL'])
y = data['CLASIFFICATION_FINAL']

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
print("F1 Score:" , f1_score(y_test, predictions))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report

data = pd.read_csv("newfile.csv")

data['TARGET'] = (data['CLASIFFICATION_FINAL'] == 1).astype(int)


columns_to_drop = ['CLASIFFICATION_FINAL']
data.drop(columns_to_drop, axis=1, inplace=True)

X = data.drop('TARGET', axis=1)
y = data['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
