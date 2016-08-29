import time
import math
import numpy as np
from collections import defaultdict

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Given an array of tokenized sentences, will generate words (as a single elt tuple)
def words(sents):
    return [[(s,) for s in sent] for sent in sents]
# END words

# Given an array of tokenized sentences (see fancySplit), will generate bigrams
# for each sentence.
# sents: an list of tokenized sentences
# output: a list of lists of tuples
def bigrams(sents):
    return [[(sent[i], sent[i+1]) for i in range(len(sent)-1)] for sent in sents]
# END bigramTokens

# Given an array of tokenized sentences (see fancySplit), will generate trigrams
# for each sentence.
# sents: an list of tokenized sentences
# output: a list of lists of tuples
def trigrams(sents):
    return [[(sent[i], sent[i+1], sent[i+2]) for i in range(len(sent)-2)] for sent in sents]
# END trigramTokens

# Given a list of sentences with whitespace between each word, will split into
# individual words.
# sents: a list of strings with whitespace-delimited words
# output: a list of lists
def fancySplit(sents):
    result = [[START_SYMBOL, START_SYMBOL] + s.split() + [STOP_SYMBOL] \
                for s in sents]
    return result
# END fancySplit

# Given an array of sentences, will summarize with unigram, bigram, and trigrams.
# sents: a list of sentences, separated by spaces.
# output: a list of lists of tuples (or strings for unigrams)
def getTokens(sents):
    # Tokenize the input sentences
    print("Tokenizing sentences...")
    toks = fancySplit(sents)

    # Create words, bigrams, and trigrams
    print("Creating unigrams, bigrams, and trigrams...")
    w = words(toks)
    b = bigrams(toks)
    t = trigrams(toks)

    return (w, b, t)
# END getTokens

# Given a list of tokens, will count the number of times each unique token
# appears in the list.
def getCounts(toks):
    d = defaultdict(int)
    for t in toks:
        d[t] += 1
    return d
# END getCounts

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    nLines = len(training_corpus)
    (words, bigrams, trigrams) = getTokens(training_corpus)

    print("Getting unigram counts")
    flatWords = [w for sublist in words for w in sublist]
    unigram_counts = getCounts(flatWords)
    totalWords = len(flatWords)
    print("Found {0} words".format(totalWords))

    # FUDGE FACTORY
    totalWords = totalWords - nLines*2

    # Direct calculation of word log probabilities.
    print("Calculating Unigram Log-Probabilities")
    unigram_p = {k : math.log(float(v) / totalWords,2) \
                    for k, v in unigram_counts.iteritems()}
    # Clean up unigram for START_SUMBOL
    unigram_p.pop((START_SYMBOL,), None)

    print("Getting bigram counts")
    flatBigrams = [b for sublist in bigrams for b in sublist]
    bigram_counts = getCounts(flatBigrams)
    # Calculate bigram MLE log probabilities.
    print("Calculating Bigram Log-Probabilities")
    bigram_p = {}
    for k, v in bigram_counts.iteritems():
        u_Count = unigram_counts[k[0:1]]
        if k[0:1] == (START_SYMBOL,):
            u_Count = u_Count/2.
        bigram_p[k] = math.log(float(v) / u_Count,2)
    # END for
    bigram_p.pop((START_SYMBOL,START_SYMBOL), None)

    print("Getting trigram counts")
    flatTrigrams = [t for sublist in trigrams for t in sublist]
    trigram_counts = getCounts(flatTrigrams)
    # Calculate trigram MLE log probabilities
    print("Calculating Trigram Log-Probabilities")
    trigram_p = {}
    for k, v in trigram_counts.iteritems():
        b_Count = bigram_counts[k[0:2]]
        trigram_p[k] = math.log(float(v) / b_Count, 2)
    # END for

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc.
def score(ngram_p, n, corpus):
    scores = []

    def wordSplit(sents):
        toks = fancySplit(sents)
        toks = [sent[2:] for sent in toks]
        return words(toks)
    def bigramSplit(sents):
        toks = fancySplit(sents)
        toks = [sent[1:] for sent in toks]
        return bigrams(toks)
    def trigramSplit(sents):
        toks = fancySplit(sents)
        return trigrams(toks)

    if n==1:
        tokenizer = wordSplit
    elif n==2:
        tokenizer = bigramSplit
    elif n==3:
        tokenizer = trigramSplit
    else:
        return None
    # END if

    corpusTokens = tokenizer(corpus)
    for sentence in corpusTokens:
        try:
            prob = 0
            for token in sentence:
                prob += ngram_p[token]
            # END for
        except KeyError:
            prob = MINUS_INFINITY_SENTENCE_LOG_PROB
        finally:
            scores.append(prob)
        # END try
    # END for

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(u_p, b_p, t_p, corpus):
    scores = []

    toks = fancySplit(corpus)
    w = words([sent[2:] for sent in toks])
    b = bigrams([sent[1:] for sent in toks])
    t = trigrams(toks)

    for i in range(len(corpus)):
        score = 0
        for k in range(len(w[i])):
            try:
                wordProb = u_p[w[i][k]]
            except KeyError:
                wordProb = MINUS_INFINITY_SENTENCE_LOG_PROB

            try:
                bigramProb = b_p[b[i][k]]
            except KeyError:
                bigramProb = MINUS_INFINITY_SENTENCE_LOG_PROB

            try:
                trigramProb = t_p[t[i][k]]
            except KeyError:
                trigramProb = MINUS_INFINITY_SENTENCE_LOG_PROB

            decimal = (2**wordProb + 2**bigramProb + 2**trigramProb)/3
            score += math.log(decimal, 2)
        # END for

        if score <= MINUS_INFINITY_SENTENCE_LOG_PROB:
            score = -1000
        scores.append(score)
    # END for

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
