with open('KNN-English-Result.txt', 'r') as ifile:
    data = ifile.readlines()

with open("KNN-English-Result.csv", 'w') as ofile:
    for k in range(0, len(data)-7, 4):
        item = data[k].split()[2]
        score = data[k].split()[3]
        key = data[k+1].split()[2]
        guess = data[k+2].split()[2]
        ofile.write("{0},{1},{2},{3}\n".format(item, score, key, guess))
