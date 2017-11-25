#%%
###---Binary Classification---###
import os
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#%%
# number 1 to 10 data
strDataSetsPath = "D:/Git/Hands-on-Machine-Learning-with-Scikit-Learn-and-TensorFlow/datasets/"
strSavePath = os.path.join(os.path.dirname((os.path.dirname(strDataSetsPath))), "03-Classification")
mnist = input_data.read_data_sets(os.path.join(strDataSetsPath, "MNIST_data"))
#%%
X_train, Y_train = mnist.train.images, mnist.train.labels
X_test, Y_test = mnist.test.images, mnist.test.labels
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#%%
# prepare the label that we want
Y_train_5 = (Y_train == 5)  # size of number 5 is 4987
Y_test_5 = (Y_test == 5)

# Model
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, Y_train_5)

# cross val score
cvs_train_scores = cross_val_score(
    sgd_clf, X_train, Y_train_5, cv=3, scoring="accuracy")
print("cvs_train_scores", cvs_train_scores)

# cross val predict
Y_train_pred = cross_val_predict(sgd_clf, X_train, Y_train_5, cv=3)

# confusion matrix
conf_mx_train = confusion_matrix(Y_train_5, Y_train_pred)
print("conf_mx_train", conf_mx_train)

#%%
# precision and recall
precision_train = precision_score(Y_train_5, Y_train_pred)
recall_train = recall_score(Y_train_5, Y_train_pred)
print("precision_train", precision_train)
print("recall_train", recall_train)

#%%
# calculate the score for each instance
Y_train_scores = cross_val_predict(
    sgd_clf, X_train, Y_train_5, cv=3, method="decision_function")
print("Y_train_scores", Y_train_scores)
#%%
# calculate precision and recall for each threshold
precisions, recalls, thresholds = precision_recall_curve(
    Y_train_5, Y_train_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend()

# plot precision and recall vs threshold
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.savefig(os.path.join(strSavePath, "Image6.jpg"))
plt.show()

# plot precision vs recall
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

plot_precision_vs_recall(precisions, recalls)
plt.axhline(0.8, c="red")
plt.savefig(os.path.join(strSavePath, "Image7.jpg"))
plt.show()

#%%
# Model 
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, Y_train_5, cv=3, method="predict_proba")

# calculate fpr, tpr for sgd_clf
fpr, tpr, thresholds = roc_curve(Y_train_5,  Y_train_scores)
# calculate fpr, tpr for forest_clf
fpr_forest, tpr_forest, thresholds_forest = roc_curve(Y_train_5,  y_probas_forest[:, 1])

# plot roc curve
def plot_roc_curve(fpr, tpr, label=None):
 plt.plot(fpr, tpr, linewidth=2, label=label)
 plt.plot([0, 1], [0, 1], "k--")
 plt.xlabel("False Positive Rate")
 plt.ylabel("True Positive Rate")

plot_roc_curve(fpr, tpr, label="SGD")
plt.plot(fpr_forest, tpr_forest, "b:", label="Random Forest")
plt.legend()
plt.savefig(os.path.join(strSavePath, "Image8.jpg"))
plt.show()


###---Multiclass Classifiation---###
