# PATP-Experiment-3-Linear-Discriminant-analysis

### Aim

To implement Linear Discriminant Analysis (LDA) on the Iris dataset using Python to classify flower species, evaluate accuracy, and visualize the results.

### Algorithm

1. Import NumPy, Pandas, Matplotlib, and Scikit-learn libraries.
2. Load the Iris dataset using load_iris().
3. Separate input features (X) and target labels (Y).
4. Split the dataset into 70% training and 30% testing data.
5. Create an LDA model using LinearDiscriminantAnalysis().
6. Train the model and transform the data into two discriminant components.
7. Predict the flower species using the test data.
8. Calculate model accuracy using accuracy_score().
9. Visualize the three flower species using a scatter plot.
10. Display the results and stop.

## Input
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris=load_iris()
X=iris.data
Y=iris.target
names = iris.target_names

# Split the data into training and testing sets
X_train, X_test, Y_train,Y_test=train_test_split(
    X,Y,
    test_size=0.3,
    random_state=42,
    stratify=Y
)
#Fit LDA
lda = LinearDiscriminantAnalysis(n_components=2)

X_train_lda = lda.fit_transform(X_train, Y_train)
X_test_lda = lda.transform(X_test)

#Predict on test data
Y_pred = lda.predict(X_test)

#Evaluate accuracy
acc= accuracy_score(Y_test, Y_pred)
print("LDA Test Accuracy:", acc)

#LDA projection for visualisation
X_lda = lda.fit_transform(X, Y)

# plot
plt.figure(figsize=(10,5))

for label, marker, color in zip(
    [0,1,2],
    ('o','^','s'),
    ('blue','red','green')):

  plt.scatter(
      X_lda[Y==label,0],
      X_lda[Y==label,1],
      marker=marker,
      color=color,
      label=names[label]
  )

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower right')

plt(":\n LDA Projection")
```
## Output
```
LDA Test Accuracy: 0.9777777777777777
Text(0.5, 1.0, 'LDA Projection')
```

<img width="844" height="470" alt="image" src="https://github.com/user-attachments/assets/51ff70a8-8948-459b-ae01-565d36fd99fb" />


## Result
Thus, Linear Discriminant Analysis was successfully implemented on the Iris dataset. The classification accuracy was calculated, and the three flower species were visualized using two linear discriminant components (LD1 and LD2).
