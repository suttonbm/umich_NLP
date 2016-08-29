import sys
import nltk
import math
import time
from collections import defaultdict
from itertools import compress
import numpy as np

START_SYMBOL = '*/*'
STOP_SYMBOL = 'STOP/STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


class SparseMatrix:
    def __init__(self):
        self.elements = {}

    def add(self, tuple, value):
        self.elements[tuple] = value

    def get(self, tuple):
        try:
            return self.elements[tuple]
        except KeyError:
            return LOG_PROB_OF_ZERO


# Given a list of sentences with whitespace between each word, will split into
# individual words.
# sents: a list of strings with whitespace-delimited words
# output: a list of lists
def fancySplit(sents):
    result = [[START_SYMBOL, START_SYMBOL] + s.split() + [STOP_SYMBOL] \
                    for s in sents]
    result = [[tok for word in sent for tok in word.rsplit('/',1)] \
                    for sent in result]
    return result
# END fancySplit

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

# Given a list of tokens, will count the number of times each unique token
# appears in the list.
def getCounts(toks):
    d = defaultdict(int)
    for t in toks:
        d[t] += 1
    return d
# END getCounts

# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    toks = fancySplit(brown_train)
    brown_words = [sent[0::2] for sent in toks]
    brown_tags = [sent[1::2] for sent in toks]
    return brown_words, brown_tags

# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    bigram = bigrams(brown_tags)
    trigram = trigrams(brown_tags)

    flatBigrams = [b for sublist in bigram for b in sublist]
    bigram_counts = getCounts(flatBigrams)

    flatTrigrams = [t for sublist in trigram for t in sublist]
    trigram_counts = getCounts(flatTrigrams)
    # Calculate trigram MLE log probabilities
    q_values = {}
    for k, v in trigram_counts.iteritems():
        b_Count = bigram_counts[k[0:2]]
        q_values[k] = math.log(float(v) / b_Count, 2)
    # END for

    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    flatWords = [w for sublist in brown_words for w in sublist]
    wordCounts = getCounts(flatWords)

    known_words = set([k for k, v in wordCounts.iteritems() if v > RARE_WORD_MAX_FREQ])
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = [[s if s in known_words else RARE_SYMBOL for s in sent] \
                            for sent in brown_words]
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    words = [w for sublist in brown_words_rare for w in sublist]
    tags = [t for sublist in brown_tags for t in sublist]

    tag_counts = getCounts(tags)
    taglist = set(tag_counts.keys())

    e_values = {}
    for tag in taglist:
        mask = [1 if t==tag else 0 for t in tags]
        words_masked = compress(words, mask)
        wordCount = getCounts(words_masked)
        tagCount = float(tag_counts[tag])

        for word in wordCount.keys():
            e_values[(word, tag)] = math.log(wordCount[word]/tagCount, 2)

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values, debug=False):
    # Create a list of tags for easy indexing.
    tags = list(taglist)
    words = list(set([k[0] for k in e_values.keys()]))

    # Convert q_values and e_values into sparse matrices
    # This enables easy indexing to get probabilities:
    #
    # P(word | preposition) = e_mat[w, p]
    # P(p3 | p1 p2) = q_mat[p1, p2, p3]
    #
    # This also simplifies and speeds up computation in the recursion phase
    # because tuples do not need to be created dynamically to calculate
    # the probabilities of each item.
    nTags = len(tags)
    tag_Dict = {k: v for k, v in zip(tags, range(nTags))}
    tag_ReverseDict = {v: k for k, v in tag_Dict.iteritems()}
    START_ID = tag_Dict['*']
    STOP_ID = tag_Dict['STOP']
    nWords = len(words)
    word_Dict = {k: v for k, v in zip(words, range(nWords))}

    if (debug):
        print tag_Dict
        print word_Dict

    e_mat = SparseMatrix()
    for t, p in e_values.iteritems():
        w_id = word_Dict[t[0]]
        t_id = tag_Dict[t[1]]
        e_mat.add((w_id, t_id), p)
    # END for
    q_mat = SparseMatrix()
    for t, p in q_values.iteritems():
        t1_id = tag_Dict[t[0]]
        t2_id = tag_Dict[t[1]]
        t3_id = tag_Dict[t[2]]
        q_mat.add((t1_id, t2_id, t3_id), p)
    # END for

    def recurseViterbi(j, n, wordN_id, vMatrix, bMatrix, bOverride = None):
        # Need to have the option to override the backpointed matrix for the
        # second column, where no data has been initialized.
        if bOverride is None:
            q_Probs = [q_mat.get((bMatrix[i, n-2], i, j)) for i in range(nTags)]
        else:
            q_Probs = [q_mat.get((bOverride, i, j)) for i in range(nTags)]
        # END if

        # Word emission probability is constant across all items
        e_Prob = e_mat.get((wordN_id, j))

        # Note that we are working with log probabilities, so we can sum
        # for the intersetion of probabilities.
        return np.add(np.add(q_Probs, vMatrix[:, n-1]), e_Prob)
    # END recurseViterbi

    def myViterbi(sent):
        # Initialize a matrix with N rows and M columns, where:
        #   N = number of POS Tag Classes
        #   M = number of words in the sentence
        vMat = np.zeros((nTags, len(sent)), np.dtype(np.float))
        bMat = np.zeros((nTags, len(sent)-1), np.dtype(np.int))

        # Initizlization step for initial transition probabilities.
        # use first word in the sentence unless it is rare.
        word0 = word_Dict[sent[0] if sent[0] in known_words else RARE_SYMBOL]
        for j in range(nTags):
            # Set the initial transition probabilities.
            vMat[j, 0] = q_mat.get((START_ID, START_ID, j)) + \
                            e_mat.get((word0, j))

            # Note: the backtrace is not used for the first column in this
            #       implementation because the starting tag is known (*).
        # END for
        if debug:
            print "Set initial transition probabilities."
            print vMat

        # Initialization step for the second column of transition probabilities
        # Hard-coded due to trigram model
        word1 = word_Dict[sent[1] if sent[1] in known_words else RARE_SYMBOL]
        for j in range(nTags):
            # Transition probabilities for the second column are recursively
            # defined based on the first column.
            vProbs = recurseViterbi(j, 1, word1, vMat, bMat, START_ID)
            vMat[j, 1] = np.max(vProbs)
            bMat[j, 0] = np.argmax(vProbs)
        # END for
        if debug:
            print "Set second column with recursive definition"
            print vMat
            print bMat

        # Recursion phase
        for n in range(2, len(sent)):
            # Use selected word unless it is a rare word.
            wordN = word_Dict[sent[n] if sent[n] in known_words else RARE_SYMBOL]
            for j in range(nTags):
                vProbs = recurseViterbi(j, n, wordN, vMat, bMat)
                vMat[j, n] = np.max(vProbs)
                bMat[j, n-1] = np.argmax(vProbs)
            # END for
            if debug:
                print "Set column {0} recursively".format(n)
                print vMat
                print bMat
        # END for

        # Terminate the recursion
        vProbs = recurseViterbi(STOP_ID, len(sent), STOP_ID, vMat, bMat)
        p_star = np.max(vProbs)
        q_star = np.argmax(vProbs)

        if debug:
            print "P*: {0}".format(p_star)
            print "Q*: {0}".format(q_star)

        # Iterate through the backtrace pointer
        ptr = q_star
        tagIds = []
        for k in range(len(sent)-2, -1, -1):
            tagIds.insert(0, ptr)
            ptr = bMat[ptr, k]
            if debug:
                print "Info for word {0}".format(k+1)
                print tagIds
                print ptr
        # END for
        tagIds.insert(0, ptr)

        # Get back the tags
        tags = [tag_ReverseDict[int(k)] for k in tagIds]

        return ' '.join(['/'.join((sent[i],tags[i])) for i in range(len(sent))]) + '\n'
    # END myViterbi

    tagged = [myViterbi(sent) for sent in brown_dev_words]
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a
# terminal newline, not a list of tokens.
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')


    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
