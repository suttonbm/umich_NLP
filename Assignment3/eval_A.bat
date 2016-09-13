python main.py "data/English-train.xml" "data/English-dev.xml" "KNN-English.answer" "SVM-English.answer" "_" "English"
scorer.exe "KNN-English.answer" "data/English-dev.key" "data/English.sensemap"
scorer.exe "SVM-English.answer" "data/English-dev.key" "data/English.sensemap"
pause

python main.py "data/Catalan-train.xml" "data/Catalan-dev.xml" "KNN-Catalan.answer" "SVM-Catalan.answer" "_" "Catalan"
scorer.exe "KNN-Catalan.answer" "data/Catalan-dev.key"
scorer.exe "SVM-Catalan.answer" "data/Catalan-dev.key"
pause

python main.py "data/Spanish-train.xml" "data/Spanish-dev.xml" "KNN-Spanish.answer" "SVM-Spanish.answer" "_" "Spanish"
scorer.exe "KNN-Spanish.answer" "data/Spanish-dev.key"
scorer.exe "SVM-Spanish.answer" "data/Spanish-dev.key"
pause