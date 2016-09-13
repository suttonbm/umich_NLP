from main import parse_data
import A
import os
import subprocess
import sys

def runPart(train_file, test_file, knn_answer, svm_answer, language):
    train_set = parse_data(train_file)
    test_set = parse_data(test_file)

    A.run(train_set, test_set, language, knn_answer, svm_answer)
# END runPart

print "Running English"
runPart("data/English-train.xml", "data/English-dev.xml", "KNN-English.answer",
        "SVM-English.answer", "English")

#print "Running Catalan"
#runPart("data/Catalan-train.xml", "data/Catalan-dev.xml", "KNN-Catalan.answer",
#        "SVM-Catalan.answer", "Catalan")

#print "Running Spanish"
#runPart("data/Spanish-train.xml", "data/Spanish-dev.xml", "KNN-Spanish.answer",
#        "SVM-Spanish.answer", "Spanish")

def evaluate_part(partIdx):

  if partIdx == 1:
    files = ['KNN-English.answer','KNN-Spanish.answer','KNN-Catalan.answer','SVM-English.answer','SVM-Spanish.answer','SVM-Catalan.answer']
    #test_files = ['data/English-dev.key data/English.sensemap','data/Spanish-dev.key','data/Catalan-dev.key'] * 2
    test_files = ['data/English-dev.key','data/Spanish-dev.key','data/Catalan-dev.key'] * 2
    baselines = [0.535,0.684,0.678] * 2
    references = [0.550,0.690,0.705,0.605,0.785,0.805]
    scores = [10] * 6
    raw_score = evaluate(files,test_files,baselines,references,scores)
    return raw_score / 60.0
  elif partIdx == 2:
    files = ['Best-English.answer','Best-Spanish.answer','Best-Catalan.answer']
    #test_files = ['data/English-dev.key data/English.sensemap','data/Spanish-dev.key','data/Catalan-dev.key']
    test_files = ['data/English-dev.key','data/Spanish-dev.key','data/Catalan-dev.key']
    baselines = [0.605,0.785,0.805]
    references = [0.650,0.810,0.820]
    scores = [20,10,10]
    raw_score = evaluate(files,test_files,baselines,references,scores)
    return raw_score / 40.0

def evaluate(files,test_files,baselines,references,scores):

  score_total = 0
  for i in range(len(files)):
    f = files[i]
    baseline = baselines[i]
    reference = references[i]
    test_file = test_files[i]
    score = scores[i]
    if not os.path.exists(f):
      print 'Please save your output file', f, 'under Assignment3 directory.'
      continue

    command = "scorer.exe " + f + " " + test_file
    print command

    #res = subprocess.check_output(command,shell = True)
    try:
      res = subprocess.check_output(command,shell = True)
    except Exception, e:
      res = None
      print 'scorer2 failed for',f
      sys.exit()

    #print res

    acc = 0
    if res:
      try:
        acc = float(res.split('\n')[2].split(' ')[2])
      except Exception, e:
        print 'scorer2 failed for',f
        sys.exit()

    print 'accuracy',acc,
    if acc < baseline:
      score_i = 0
    elif acc >= reference:
      score_i = score
    else:
      score_i = (score - score*(reference - acc)/(reference - baseline))

    score_total += score_i
    print 'score',score_i

  return score_total

evaluate_part(1)
