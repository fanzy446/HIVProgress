# load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
dataset = pandas.read_csv('training_data_mod.csv')

# shape
#print(dataset.shape)
#
## head
#print(dataset.head(20))
#
## descriptions
#print(dataset.describe())
#
## class distribution
#print(dataset.groupby('Resp').size())

# dataset.plot(kind='box', sharex=False, sharey=False)
# plt.show()

# histograms
# dataset.hist()
# plt.show()

# scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

# split-out validation dataset
array = dataset.values

X = array[:, 2:4]
Y = array[:, 4]
validation_size = 0.20
seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# test options and evaluation metric
seed = 7
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
print '***************************** human data *****************************'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
#    print predictions
    msg = "%s: %f (%f) %f" % (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions))
    print(msg)

dic = {'-':0, 'A':1, 'T':2, 'C':3, 'G':4, 'Y':5, 'R':6, 'W':7, 'M':8, 'K':9, 'N':10, 'S':11, 'B':12, 'H':13, 'D':14, 'V':15}
# two sequences
X2 = array[:, 0:2]
X2_mod = []

ratio = len(X2[0][1]) / len(X2[0][0])
for sequences in X2:
    temp = [ratio * dic[letter] for letter in sequences[0]]
    temp += [dic[letter] for letter in sequences[1]]
    X2_mod.append(temp)

seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X2_mod, Y, test_size=validation_size, random_state=seed)

print '***************************** virus DNA *****************************'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    
    model.fit(X_train, Y_train)

    predictions = model.predict(X_validation)
#    print predictions
    msg = "%s: %f (%f) %f" % (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions))
    print(msg)

# four attributes
X3 = array[:, 0:4]
X3_mod = []

ratio = len(X3[0][1]) / len(X3[0][0])
ratio2 = len(X3[0][1])
for sequences in X3:
    temp = [ratio * dic[letter] for letter in sequences[0]]
    temp += [dic[letter] for letter in sequences[1]]
    temp += [ratio2 * sequences[2]]
    temp += [ratio2 * sequences[3]]
    X3_mod.append(temp)

seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X3_mod, Y, test_size=validation_size, random_state=seed)

print '***************************** four attributes *****************************'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_validation)
    #    print predictions
    msg = "%s: %f (%f) %f" % (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions))
    print(msg)




