# load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from find_markers import find_markers
from sklearn.metrics import confusion_matrix

# load dataset
dataset = pandas.read_csv('training_data_mod.csv')

# shape
# print(dataset.shape)
#
## head
# print(dataset.head(20))
#
## descriptions
# print(dataset.describe())
#
## class distribution
# print(dataset.groupby('Resp').size())

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
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# test options and evaluation metric
# seed = 7
scoring = 'accuracy'

models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
models.append(('RF', RandomForestClassifier()))

# evaluate each model in turn
results = []
names = []
# print '***************************** human data *****************************'
# for name, model in models:
#    kfold = model_selection.KFold(n_splits=10, random_state=seed)
#    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#    results.append(cv_results)
#    names.append(name)
#
#    model.fit(X_train, Y_train)
#    predictions = model.predict(X_validation)
##    print predictions
#    msg = "%s: %f (%f) %f" % (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions))
#    print(msg)
#
dic = {'-': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4, 'Y': 5, 'R': 6, 'W': 7, 'M': 8, 'K': 9, 'N': 10, 'S': 11, 'B': 12,
       'H': 13, 'D': 14, 'V': 15}
## two sequences
# X2 = array[:, 0:2]
# X2_mod = []
#
# ratio = len(X2[0][1]) / len(X2[0][0])
# for sequences in X2:
#    temp = [ratio * dic[letter] for letter in sequences[0]]
#    temp += [dic[letter] for letter in sequences[1]]
#    X2_mod.append(temp)
#
# seed = 0
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X2_mod, Y, test_size=validation_size, random_state=seed)
#
# print '***************************** virus DNA *****************************'
# for name, model in models:
#    kfold = model_selection.KFold(n_splits=10, random_state=seed)
#    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#    results.append(cv_results)
#    names.append(name)
#
#    model.fit(X_train, Y_train)
#
#    predictions = model.predict(X_validation)
##    print predictions
#    msg = "%s: %f (%f) %f" % (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions))
#    print(msg)

# four attributes
X3 = array[:, 0:4]
X3_mod = []

# encoding sequence!!!

# ratio = len(X3[0][1]) / len(X3[0][0])
# ratio2 = len(X3[0][1])
ratio = 1
ratio2 = 1

# for sequences in X3:
#    temp = [ratio * dic[letter] for letter in sequences[0]]
#    temp += [dic[letter] for letter in sequences[1]]
#    temp += [ratio2 * sequences[2]]
#    temp += [ratio2 * sequences[3]]
#    X3_mod.append(temp)
len0 = len(X3[0][0])
len1 = len(X3[0][1])

enc0 = preprocessing.OneHotEncoder(n_values=[len(dic) for i in range(len0)])
enc1 = preprocessing.OneHotEncoder(n_values=[len(dic) for i in range(len1)])

# seq0 = []
# seq1 = []
# for sequences in X3:
#     temp = [dic[letter] for letter in sequences[0]]
#     seq0.append(temp)
#     temp = [dic[letter] for letter in sequences[1]]
#     seq1.append(temp)
#
# enc0.fit(seq0)
# enc1.fit(seq1)
#
# for sequences in X3:
#     temp = [dic[letter] for letter in sequences[0]]
#     a0 = enc0.transform([temp]).toarray()
#     temp = [dic[letter] for letter in sequences[1]]
#     a1 = enc1.transform([temp]).toarray()
#     tmp = np.append(a0[0], a1[0])
#     tmp = np.append(tmp, sequences[2:])
#     #    tmp = list(a0[0]) + list(a1[0]) + list(sequences[2:])
#     X3_mod.append(tmp)

seed = 0
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X3, Y, test_size=validation_size,
                                                                                random_state=seed)

seq0 = []
seq1 = []
for sequences in X_train:
    temp = [dic[letter] for letter in sequences[0]]
    seq0.append(temp)
    temp = [dic[letter] for letter in sequences[1]]
    seq1.append(temp)

enc0.fit(seq0)
enc1.fit(seq1)

for sequences in X_train:
    temp = [dic[letter] for letter in sequences[0]]
    a0 = enc0.transform([temp]).toarray()
    temp = [dic[letter] for letter in sequences[1]]
    a1 = enc1.transform([temp]).toarray()
    tmp = np.append(a0[0], a1[0])
    tmp = np.append(tmp, sequences[2:])
    #    tmp = list(a0[0]) + list(a1[0]) + list(sequences[2:])
    X3_mod.append(tmp)

print X_validation[0]
print '***************************** four attributes *****************************'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X3_mod, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

    #    print model.get_params(True)
    weights_history = []
    for i in range(20):
        model = RandomForestClassifier()
        model.fit(X3_mod, Y_train)
        weights_history.append(model.feature_importances_)

    weight = []
    for index in range(len(weights_history[0])):
        max = 0
        min = 1
        sum = 0
        for i in range(20):
            if max < weights_history[i][index]:
                max = weights_history[i][index]
            if min > weights_history[i][index]:
                min = weights_history[i][index]
            sum += weights_history[i][index]
        sum = sum - max - min
        average = sum / 18
        weight.append(average)

    #    if name == 'RF':
    #        for weight in model.feature_importances_:
    #            print weight

    (PR_pos, PR_neg, RT_pos, RT_neg) = find_markers(X3, Y, weight, len0, len1, len(dic))
    # predictions = model.predict(X_validation)
    #    print predictions
    # c_matrix = confusion_matrix(Y_validation, predictions)
    print "fengexian"
    # print c_matrix
    # msg = "%s: %f (%f) %f" % (name, cv_results.mean(), cv_results.std(), accuracy_score(Y_validation, predictions))
    # print(msg)

    result = 0
    for idx, test in enumerate(X_validation):
        sum_pos = 0
        sum_neg = 0
        for pr_pos in PR_pos:
            if test[0][pr_pos[1]: pr_pos[2] + 1] == pr_pos[0]:
                sum_pos += 1
        for pr_neg in PR_neg:
            if test[0][pr_neg[1]: pr_neg[2] + 1] == pr_neg[0]:
                sum_neg += 1
        for rt_pos in RT_pos:
            if test[1][rt_pos[1]: rt_pos[2] + 1] == rt_pos[0]:
                sum_pos += 1
        for rt_neg in RT_neg:
            if test[1][rt_neg[1]: rt_neg[2] + 1] == rt_neg[0]:
                sum_neg += 1
        if sum_neg >= sum_pos:
            if Y_validation[idx] == 'worse':
                result += 1
        else:
            if Y_validation[idx] == 'better':
                result += 1
        print (sum_pos, sum_neg, Y_validation[idx])

    print float(result) / len(X_validation)





