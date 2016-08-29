import sys

def main():
    if len(sys.argv) < 3:
	print "Usage: python pos.py <tagger output> <reference file>"
	exit(1)

    infile = open(sys.argv[1], "r")
    user_sentences = infile.readlines()
    infile.close()

    infile = open(sys.argv[2], "r")
    correct_sentences = infile.readlines()
    infile.close()

    num_correct = 0
    total = 0

    print "{0} user sentences".format(len(user_sentences))
    print "{0} golden sentences".format(len(correct_sentences))
    for user_sent, correct_sent in zip(user_sentences, correct_sentences):
        user_tok = user_sent.split()
        correct_tok = correct_sent.split()

        if len(user_tok) != len(correct_tok):
            print "Sentence length mismatch"
            continue

        for u, c in zip(user_tok, correct_tok):
            if u == c:
                num_correct += 1
            total += 1

    score = float(num_correct) / total * 100

    print "Percent correct tags:", score


if __name__ == "__main__": main()
