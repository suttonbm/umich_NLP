import A
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# You might change the window size
window_size = 15

# Constants for an instance tuple
INST_ID = 0
L_CON = 1
HEAD = 2
R_CON = 3
SENSE_ID = 4

# Default options for feature generator
DEFAULT_OPS = {"STOPWORDS":    False, \
               "PUNCTUATION":  False, \
               "BAGOFWORDS":   True, \
               "COLLOCATION":  False, \
               "PARTOFSPEECH": False \
               }

def countElts(l):
    result = defaultdict(int)
    for i in l:
        result[i] += 1
    return result
# END countElts

def bagOfWordsFeatures(lContext, rContext, window):
    '''
    :param lContext: a list of words appearing to the LEFT of the target
    :param rContext: a list of words appearing to the RIGHT of the target
    :param window: an integer number specifying a maximum distance from
        the target from which to select the context

    :return: dict where the key is a word and the value is its count
        (the number of appearances within the context)
    '''
    lCon = lContext[-window:]
    if len(lCon) < window:
        lCon = ['<B>'] * (window - len(lCon)) + lCon
    # END if

    rCon = rContext[:window]
    if len(rCon) < window:
        rCon = rCon + ['<E>'] * (window - len(rCon))
    # END if

    context = [elt + ".BOW" for elt in lCon + rCon]
    bagCounts = countElts(context)

    return bagCounts
# END bagOfWordsFeatures

def collocationFeatures(lContext, rContext, window, fn=lambda x: x):
    '''
    :param lContext: a list of words appearing to the LEFT of the target
    :param rContext: a list of words appearing to the RIGHT of the target
    :param window: an integer number specifying a maximum distance from
        the target from which to select the context
    :param fn: an optional scaling function to calculate a value for each
        collocation.

    :return: dict where the key is a word and the value is fn(distance)
        from the target word.  In cases where there are multiple appearances
        the closest distance is selected.
    '''
    lCon = lContext[-window:]
    if len(lCon) < window:
        lCon = ['<B>'] * (window - len(lCon)) + lCon
    # END if

    rCon = rContext[:window]
    if len(rCon) < window:
        rCon = rCon + ['<E>'] * (window - len(rCon))
    # END if

    result = {}
    for k in range(0, window):
        dist = fn(k)
        rFeat = rCon[k] + '.COLLOC'
        lFeat = lCon[-(k+1)] + '.COLLOC'
        if rFeat not in result.keys():
            result[rFeat] = dist
        if lFeat not in result.keys():
            result[lFeat] = -dist
    # END for

    return result
# END collocationFeatures

def rmStopWords(l):
    stops = set(stopwords.words('english'))
    return [w for w in l if w not in stops]
# END rmStopWords

def rmPunct(s):
    tab = {ord(c): None for c in string.punctuation}
    return s.translate(tab)
# END rmPunct

def posCollocationFeatures(target, lContext, rContext, window):
    '''
    :param target: the target word to be disambiguated
    :param lContext: a list of words appearing to the LEFT of the target
    :param rContext: a list of words appearing to the RIGHT of the target
    :param window: an integer number specifying a maximum distance from
        the target from which to select the context

    :return: dict where the key is a part of speech and the value is the distance
        from the target word.
    '''
    lCon = lContext[-window:]
    if len(lCon) < window:
        lCon = ['<B>'] * (window - len(lCon)) + lCon
    # END if

    rCon = rContext[:window]
    if len(rCon) < window:
        rCon = rCon + ['<E>'] * (window - len(rCon))
    # END if

    context = lCon + [target] + rCon
    context_Tagged = nltk.pos_tag(context)

    result = {}
    for k in range(window-1, -1, -1):
        rFeat = context_Tagged[-(k+1)][1] + '.POS'
        lFeat = context_Tagged[k][1] + '.POS'
        if rFeat not in result.keys():
            result[rFeat] = float(window - k)/window
        if lFeat not in result.keys():
            result[lFeat] = -float(window - k)/window
    # END for
    result['HEAD.POS'] = context_Tagged[window][1]

    return result
# END posCollocationFeatures

def getFeatures(inst, ops):
    result = {}

    lContext = inst[L_CON].lower()
    rContext = inst[R_CON].lower()

    if ops['PUNCTUATION']:
        lContext = rmPunct(lContext)
        rContext = rmPunct(rContext)

    lContext = lContext.split()
    rContext = rContext.split()

    if ops['STOPWORDS']:
        lContext = rmStopWords(lContext)
        rContext = rmStopWords(rContext)
    if ops['BAGOFWORDS']:
        result.update(bagOfWordsFeatures(lContext, rContext, window_size))
    if ops['COLLOCATION']:
        result.update(collocationFeatures(lContext, rContext, window_size, \
                        lambda x: float(x)/window_size))
    if ops['PARTOFSPEECH']:
        result.update(posCollocationFeatures(inst[HEAD], lContext, rContext, \
                        window_size))

    return result

# B.1.a,b,c,d
def extract_features(data, ops=DEFAULT_OPS):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param data: dict of items defining which features to select

    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    for instance in data:
        # Add label to the dict
        labels[instance[INST_ID]] = instance[SENSE_ID]

        # Get features dict
        features[instance[INST_ID]] = getFeatures(instance, ops)

    return features, labels

# implemented for you
def vectorize(train_features, test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train, X_test, y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []

    trainVectors, _, trainOutcomes = A.getFeatureVectors(X_train, y_train)
    testVectors, testKeys = A.getFeatureVectors(X_test)

    # Select Features
    svm_clf = svm.LinearSVC()
    selector = RFE(svm_clf, verbose=0, step=10)
    selector = selector.fit(trainVectors, trainOutcomes)
    featMask = selector.get_support()

    # Mask Features
    nItems = testVectors.shape[0]
    testVectorsNew = np.zeros((nItems, np.sum(featMask)))
    for k in range(nItems):
        testVectorsNew[k, :] = testVectors[k, :][featMask]

    model = selector.estimator_
    svm_predict = model.predict(testVectorsNew)
    #svm_clf.fit(trainVectorsNew, trainOutcomes)
    #svm_predict = svm_clf.predict(testVectors)

    results = [(testKeys[k], svm_predict[k]) for k in range(len(testKeys))]

    return results

# run part B
def run(train, test, language, answer, ops=DEFAULT_OPS):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt], ops)

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)