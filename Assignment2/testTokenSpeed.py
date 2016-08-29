import timeit
import nltk

testCorpus = 'data/brown_small.txt'

setupSplit = """
with open(testCorpus, 'r') as ifile:
    corpus = ifile.readlines()
"""

def testSplitting(corpus):
    pass
# END testSplitting

def testTokenizing(splitCorpus):
    pass
# END testTokenizing

def testSummarizing(words, bigrams, trigrams):
    pass
# END testSummarizing

def main():
    with open(testCorpus, 'r') as ifile:
        corpus = ifile.readlines()
    # END with
    tokens = testSplitting(corpus)
    (words, bigrams, trigrams) = testTokenizing(tokens)
    (words, bigrams, trigrams) = testSummarizing(words, bigrams, trigrams)
# END main

if __name__ == "__main__":
    main()