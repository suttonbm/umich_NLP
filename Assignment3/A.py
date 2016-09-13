#from main import replace_accented
from sklearn import svm
from sklearn import neighbors
from collections import defaultdict
import numpy as np
import unicodedata

# don't change the window size
window_size = 10

# Constants for an instance tuple
INST_ID = 0
L_CON = 1
HEAD = 2
R_CON = 3
SENSE_ID = 4

def replace_accented(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def getContext(inst):
    '''
    :param inst: tuple with the following structure:
        (instance_id, left_context, head, right_context, sense_id)
    :return: list of words within window_size distance of the head
    '''
    lCon = inst[L_CON].split()[-window_size:]
    if len(lCon) < window_size:
        lCon = ['<B>'] * (window_size - len(lCon)) + lCon

    rCon = inst[R_CON].split()[:window_size]
    if len(rCon) < window_size:
        rCon = rCon + ['<E>'] * (window_size - len(rCon))

    return [s.lower() for s in lCon + rCon]

def getWordCounts(S):
    '''
    :param S: a list of words which have appeared in context with a lexical elt
    :return: dict where the key is a word and the key is its count
    '''
    result = defaultdict(int)
    for word in S:
        result[word] += 1
    # END for
    return result

def getFeatureVectors(X, y=None):
    '''
    :param X: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
    :return: an array of size (m, n) where:
        m = number of samples
        n = number of features
    '''
    keys = X.keys()
    n = len(X[keys[0]]) # all items should have same number of features
    m = len(keys)

    result_X = np.zeros((m, n))
    result_y = np.zeros(m, dtype=(np.unicode_, 16))

    for k in range(len(keys)):
        result_X[k,:] = X[keys[k]]
        if y is not None:
            result_y[k] = y[keys[k]]
    # END for

    if y is not None:
        return (result_X, keys, result_y)
    else:
        return (result_X, keys)

def sortByInstance(data):
    '''
    :param data: a list of tuples of form [(instance, result), ...]
    :return: a list of tuples, sorted by instance
    '''
    return sorted(data, key=lambda item: item[0])

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''

    s = defaultdict(list)
    for lexelt, senses in data.iteritems():
        result = set()
        for sense in senses:
            context = getContext(sense)
            result.update(set(context))
        # END for
        s[lexelt] = sorted(list(result))
    # END for

    return s

# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}
    for instance in data:
        context = getContext(instance)
        contextCounts = getWordCounts(context)
        vectors[instance[INST_ID]] = [contextCounts[word] for word in s]
        labels[instance[INST_ID]] = instance[SENSE_ID]
    # END for

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

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

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    # Translate training data into feature vectors
    trainVectors, _, trainOutcomes = getFeatureVectors(X_train, y_train)
    testVectors, testKeys = getFeatureVectors(X_test)

    svm_clf = svm.LinearSVC()
    svm_clf.fit(trainVectors, trainOutcomes)
    svm_predict = svm_clf.predict(testVectors)
    svm_results = [(testKeys[k], svm_predict[k]) for k in range(len(testKeys))]

    knn_clf = neighbors.KNeighborsClassifier()
    knn_clf.fit(trainVectors, trainOutcomes)
    knn_predict = knn_clf.predict(testVectors)
    knn_results = [(testKeys[k], knn_predict[k]) for k in range(len(testKeys))]

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results, output_file):
    '''
    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''
    with open(output_file, 'w') as ofile:
        sorted_Lexelts = sorted(results.keys())
        for lexelt in sorted_Lexelts:
            sortedValues = sortByInstance(results[lexelt])
            for tup in sortedValues:
                ofile.write("{0} {1} {2}\n".format(replace_accented(lexelt), \
                                                    replace_accented(tup[0]), \
                                                    replace_accented(tup[1])))
            # END for
        # END for
    pass

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s.keys():
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)
