import random
from providedcode import dataset
from providedcode.transitionparser import TransitionParser
from providedcode.evaluate import DependencyEvaluator
from featureextractor import MyFeatureExtractor
from transition import Transition

if __name__ == '__main__':
    tests = {'swedish': dataset.get_swedish_train_corpus,
             'danish': dataset.get_danish_train_corpus,
             'english': dataset.get_english_train_corpus}
    scoreWeight = {'swedish': 25.,
                   'danish': 25.,
                   'english': 50.}
    totalPoints = 0
    for testName in tests.keys():
        data = tests[testName]().parsed_sents()
        data_1h = data[0:(len(data)/2)]
        data_2h = data[(len(data)/2):-1]

        random.seed(99999)
        traindata = random.sample(data_1h, 200)
        testdata = random.sample(data_2h, 800)

        try:
            print "Training {0} model...".format(testName)
            tp = TransitionParser(Transition, MyFeatureExtractor)
            tp.train(traindata)
            tp.save(testName + ".model")

            print "Testing {0} model...".format(testName)
            parsed = tp.parse(testdata)

#            with open('test.conll', 'w') as f:
#                for p in parsed:
#                    f.write(p.to_conll(10).encode('utf-8'))
#                    f.write('\n')

            ev = DependencyEvaluator(testdata, parsed)
            print "Test Results For: {0}".format(testName)
            (uas, las) = ev.eval()
            points = scoreWeight[testName] * (min(0.7, las)/0.7)**2
            totalPoints += points
            print "UAS: {0} \nLAS: {1}".format(uas, las)
            print "Points Scored: {0}".format(points)

            # parsing arbitrary sentences (english):
            # sentence = DependencyGraph.from_sentence('Hi, this is a test')

            # tp = TransitionParser.load('english.model')
            # parsed = tp.parse([sentence])
            # print parsed[0].to_conll(10).encode('utf-8')
        except NotImplementedError:
            print """
            This file is currently broken! We removed the implementation of Transition
            (in transition.py), which tells the transitionparser how to go from one
            Configuration to another Configuration. This is an essential part of the
            arc-eager dependency parsing algorithm, so you should probably fix that :)

            The algorithm is described in great detail here:
                http://aclweb.org/anthology//C/C12/C12-1059.pdf

            We also haven't actually implemented most of the features for for the
            support vector machine (in featureextractor.py), so as you might expect the
            evaluator is going to give you somewhat bad results...

            Your output should look something like this:

                LAS: 0.23023302131
                UAS: 0.125273849831

            Not this:

                Traceback (most recent call last):
                    File "test.py", line 41, in <module>
                        ...
                        NotImplementedError: Please implement shift!


            """
        # END try
    # END for
    print "Total Points: {0}".format(totalPoints)