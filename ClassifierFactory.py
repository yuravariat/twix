# -*- coding: utf-8 -*-
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from enum import Enum
from sklearn import tree, cross_validation
from sklearn.ensemble import AdaBoostClassifier

from Transformers.transformers import GlobalTransformer


class ClassifierType(Enum):
    MultinomialNB = 1
    SVM = 2
    DecisionTree = 3
    RandomForest = 4
    LogisticRegression = 5
    AdaBoost = 6


class Classifier:

    def __init__(self):
        pass

    classifier = None
    labels = None


class ClassifierSettings:

    def __init__(self):
        pass

    '''
    enable_text_length_transformer = False
    enable_url_transformer = False
    enable_pos_transformer = False
    enable_ngrams_transformer = False
    enable_emoticons_transformer = False
    enable_username_transformer = False
    enable_part_of_day_transformer = False
    enable_punctuation_transformer = False
    '''

    train_data = None
    classifier_type = None
    categories = None


class ClassifierFactory:

    def __init__(self):
        pass

    annotated_data = None
    classifier_type = None
    classifierSettings = None

    def get_transformers(self):
        transformers_list = []
        transformers_list.append(('f1', GlobalTransformer()))
        print 'feature: ' + str([x[0] for x in transformers_list])
        return transformers_list

    def get_classifier(self):
        __classifier = None
        if self.classifier_type is ClassifierType.MultinomialNB:
            __classifier = MultinomialNB()
        if self.classifier_type is ClassifierType.SVM:
            __classifier = SVC(kernel='linear',
                               C=0.5,
                               probability=True,
                               #class_weight={1:155},
                               random_state=0)
        if self.classifier_type is ClassifierType.DecisionTree:
            __classifier = tree.DecisionTreeClassifier()
        if self.classifier_type is ClassifierType.RandomForest:
            __classifier = RandomForestClassifier()
        if self.classifier_type is ClassifierType.LogisticRegression:
            __classifier = LogisticRegression(class_weight={1:155, 0:1/155})
        if self.classifier_type is ClassifierType.AdaBoost:
            __classifier = AdaBoostClassifier()
        return __classifier

    def buildClassifier(self, classifierSettings):

        print('start training classifier')
        self.classifierSettings = classifierSettings

        if self.classifierSettings.classifier_type is None and self.classifier_type is None:
            self.classifier_type = ClassifierType.MultinomialNB

        if self.classifierSettings.classifier_type is not None and self.classifier_type is None:
            self.classifier_type = classifierSettings.classifier_type

        # Annotated data
        self.annotated_data = classifierSettings.train_data

        cats_with_counts = {}
        for target in classifierSettings.train_data.target.tolist():
            if target in cats_with_counts.keys():
                cats_with_counts[target] += 1
            else:
                cats_with_counts[target] = 1

        #for t_name in classifierSettings.train_data.target_names:
        #    cats_with_counts.append((t_name, len([x for x in classifierSettings.train_data.target.tolist() if
        #                                          x == classifierSettings.train_data.target_names.index(t_name)])))

        print('train contains ' + str(len(classifierSettings.train_data.data)) + ' tweets and ' +
                str(len(classifierSettings.train_data.target_names)) +
              ' categories: ' + str(cats_with_counts) )

        # Postprocessing (urls, numbers and user references replacement)
        #preproccessor = PreProccessor()
        #preproccessor.perform(self.annotated_data.data)

        print('pre-proccess done')

        print('classifier algorithm: ' + str(self.classifier_type) )

        transformers_list = self.get_transformers()
        features = FeatureUnion(transformer_list=transformers_list)
        __classifier = self.get_classifier()

        pipeline = Pipeline([
          #('features', features),
          ('classifier', __classifier)
        ])

        # Actually builds the classifier
        classifier = pipeline.fit(self.annotated_data.data, self.annotated_data.target)
        print 'classifier ready to use'

        classifier_obj = Classifier()
        classifier_obj.classifier = classifier
        classifier_obj.labels = self.annotated_data.target_names

        return classifier_obj
