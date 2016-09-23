import codecs
import os

from sklearn import cross_validation
from sklearn.datasets import get_data_home
from pandas import DataFrame, np
from ClassifierFactory import ClassifierFactory, ClassifierSettings, ClassifierType
from classifier.data import WixDataAdapter, WixInstance

predict_mode = True
category = 'flag'
categories = ['0', '1']
dataAdapter = WixDataAdapter(category)

# 1. Generate training set by splitting the input files multiple files
# dataAdapter.create_data()

# 2. Load train data from files or cache
trainData = dataAdapter.get_train_data(categories=categories)

# 3. Train classifier
classifierBuilder = ClassifierFactory()
classifierSettings = ClassifierSettings()
classifierSettings.train_data = trainData
classifierSettings.classifier_type = ClassifierType.RandomForest

clf = classifierBuilder.buildClassifier(classifierSettings)

if not predict_mode:
    # Evaluation with cross validation test
    print 'performing cross validation c=5 on train data'
    scores = cross_validation.cross_val_score(clf.classifier,
                                              trainData.data,
                                              trainData.target,
                                              cv=5,
                                              scoring='precision_weighted')
    scores_mean = scores.mean()
    print 'cross validation done'
    print 'scores: ' + str(scores)
    print 'scores_mean: ' + str(scores_mean)

else:
    # make prediction
    testData = dataAdapter.get_unclassified_data(categories=categories)
    # predicted = clf.classifier.predict(testData.data)
    predicted_prob = clf.classifier.predict_proba(testData.data)
    print ('predict done')

    #for i in range(len(testData.data)):
    #    probabilities = predicted_prob[i]
    #    zero_prob = probabilities[0]
    #    one_prob = probabilities[1]
    #
    #    if one_prob > 0.1:
    #        print one_prob
    #
    #    #testData.data[i].append(str(clf.labels[predicted_prob[i]]))

    file_dir = os.path.join(get_data_home(), 'output')

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    # np.savetxt(os.path.join(file_dir, "predicted.csv"), predicted, delimiter=",")
    np.savetxt(os.path.join(file_dir, "predicted_prob.csv"), predicted_prob, delimiter=",")

    print("done")

print('done!')