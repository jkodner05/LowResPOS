#!/bin/bash

header="#$ -cwd
\n#$ -V
\n#$ -l mem=64G
\n#$ -l h_cpu=372800
\n#$ -pe parallel-onenode 4
\n#$ -S /bin/bash
\n#$ -M jkodner@seas.upenn.edu
\n#$ -m eas
\n#$ -j y -o /home1/j/jkodner/"

#k=400; FernAguado; no tags
#python parkesclustering.py --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/FernAguado /mnt/nlpgridio2/nlp/users/jkodner/lorelei/spanish_notags.pickle -k 400
#python parkesclustering.py --loadmats /mnt/nlpgridio2/nlp/users/jkodner/lorelei/spanish_notags.pickle

#korean; k=400; no tags
if false; then
echo -e $header > korean_notags.sh
echo >> korean_notags.sh
printf "python parkesclustering.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ko/ /mnt/nlpgridio2/nlp/users/jkodner/lorelei/korean_notags.pickle -k 400 --both
#python parkesclustering.py --loadmats /mnt/nlpgridio2/nlp/users/jkodner/lorelei/korean_notags.pickle --both" >> korean_notags.sh
qsub korean_notags.sh
fi

#Tamil; k=400; no tags
if false; then
echo -e $header > tamil_notags.sh
echo >> tamil_notags.sh
printf "python parkesclustering.py --corpus lctl /mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Tamil_v1.0/POS_Tagged_Text/ltf /mnt/nlpgridio2/nlp/users/jkodner/lorelei/tamil_notags.pickle -k 400 --both
#python parkesclustering.py --loadmats /mnt/nlpgridio2/nlp/users/jkodner/lorelei/tamil_notags.pickle --both" >> tamil_notags.sh
qsub tamil_notags.sh
fi

#Kurdish; k=400; no tags
if false; then
echo -e $header > kurdish_notags.sh
echo >> kurdish_notags.sh
printf "python parkesclustering.py --corpus lctl /mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Kurdish_v1.0/POS_Tagged_Text/ltf /mnt/nlpgridio2/nlp/users/jkodner/lorelei/kurdish_notags.pickle -k 400 --both
#python parkesclustering.py --loadmats /mnt/nlpgridio2/nlp/users/jkodner/lorelei/kurdish_notags.pickle --both" >> kurdish_notags.sh
qsub kurdish_notags.sh
fi

#Tagalog; k=400; no tags
if false; then
echo -e $header > tagalog_notags.sh
echo >> tagalog_notags.sh
printf "python parkesclustering.py --corpus lctl /mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Tagalog_v1.0/POS_Tagged_Text/ltf /mnt/nlpgridio2/nlp/users/jkodner/lorelei/tagalog_notags.pickle -k 400 --both
#python parkesclustering.py --loadmats /mnt/nlpgridio2/nlp/users/jkodner/lorelei/tagalog_notags.pickle --both" >> tagalog_notags.sh
qsub tagalog_notags.sh
fi





#Turkish TS; k=1000
if false; then
echo "Turkish TURKISH k1000"
echo -e $header > turkish_k1000_turkishtags.sh
echo >> turkish_k1000_turkishtags.sh
printf "#python parkesclustering.py --corpus turkishts /mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS/ /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/turkish_k1000_turkishtags.pickle -k 1000 --both
echo \"TURKISH Tagset\"
python parkesclustering.py --loadmats  --corpus turkishts /mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/turkish_k1000_turkishtags.pickle -s 3 --both -k 50,100,500,900,1000 -c 0.9 --guessremainder" >> turkish_k1000_turkishtags.sh
bash turkish_k1000_turkishtags.sh
fi


#Turkish TS; k=10000
if false; then
echo "Turkish TURKISH k10000"
echo -e $header > turkish_k10000_turkishtags.sh
echo >> turkish_k10000_turkishtags.sh
printf "python parkesclustering.py --corpus turkishts /mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS/ /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/turkish_k10000_turkishtags.pickle -k 10000 --both
echo \"TURKISH Tagset\"
#python parkesclustering.py --loadmats  --corpus turkishts /mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/turkish_k10000_turkishtags.pickle -s 3 --both -k 50,100,500,900,10000 -c 0.9" >> turkish_k10000_turkishtags.sh
qsub turkish_k10000_turkishtags.sh
fi





#English Brown; k=1000
if false; then
echo "English Brown k1000"
echo -e $header > englishbrown_k1000_browntags.sh
echo >> englishbrown_k1000_browntags.sh
printf "#python parkesclustering_static.py --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k1000_browntags.pickle -k 1000 -s 3 --both
echo \"Brown Tagset -> Reduced Tagset\"
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k1000_browntags.pickle --evalmap brown_to_reduced.txt -s 3 --both -k 100,500,900,1000 -c 0.5 --guessremainder
echo \"***Brown Tagset\"
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k1000_browntags.pickle -s 3 --both -k 100,500,900,1000 -c 0.5 
echo \"***Reduced Tagset S=Carlson\"
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k1000_browntags.pickle --distmap brown_to_reduced.txt -s 0 --both -k 100,500,900,1000 -c 0.9 --guessremainder
echo \"***Reduced Tagset S=3\"
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k1000_browntags.pickle --distmap brown_to_reduced.txt -s 3 --both -k 100,500,900,1000 -c 0.9 --guessremainder
echo \"***Reduced Tagset S=11\"
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k1000_browntags.pickle --distmap brown_to_reduced.txt -s 11 --both -k 100,500,900,1000 -c 0.9 --guessremainder" >> englishbrown_k1000_browntags.sh
qsub englishbrown_k1000_browntags.sh
fi


#English Brown; k=8307
if false; then
echo "English Brown k8307"
echo -e $header > englishbrown_k8307_brownreducedtags.sh
echo >> englishbrown_k8307_brownreducedtags.sh
printf "#python parkesclustering_static.py --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k8307_notags.pickle -k 8307 -s 3 --both
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k8307_notags.pickle --evalmap brown_to_reduced.txt -s 3 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.5
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k8307_notags.pickle -s 3 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.5" >> englishbrown_k8307_brownreducedtags.sh
qsub englishbrown_k8307_brownreducedtags.sh #> english-childes_k8307_brown-reduced2.txt

echo -e $header > englishbrown_k8307_reducedtags.sh
echo >> englishbrown_k8307_reducedtags.sh
printf "echo \"***Reduced Tagset S=Carlson\"
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k8307_notags.pickle --distmap brown_to_reduced.txt -s 0 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.9
echo \"***Reduced Tagset S=3\"
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k8307_notags.pickle --distmap brown_to_reduced.txt -s 11 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.9
echo \"***Reduced Tagset S=11\"
python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k8307_notags.pickle --distmap brown_to_reduced.txt -s 3 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.9" >> englishbrown_k8307_reducedtags.sh
qsub englishbrown_k8307_reducedtags.sh #> english-childes_k8307_reduced2.txt

echo -e $header > englishbrown_k8307_browntags.sh
echo >> englishbrown_k8307_browntags.sh
printf "python parkesclustering.py --loadmats  --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbrown_k8307_notags.pickle -s 3 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.5" >> englishbrown_k8307_browntags.sh
qsub englishbrown_k8307_browntags.sh #> english-childes_k8307_brown2.txt
fi


#English BLIIP; k=10000
if false; then
echo "English BLIIP k10000"
echo -e $header > englishbliip_k10000_wsjtags.sh
echo >> englishbliip_k10000_wsjtags.sh
printf "python parkesclustering.py --corpus wsj /mnt/nlpgridio3/data/corpora/lorelei/bliip_87_89_wsj/data/ /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbliip_k10000_wsjtags.pickle -k 10000 --both
python parkesclustering.py --loadmats  --corpus wsj /mnt/nlpgridio3/data/corpora/lorelei/bliip_87_89_wsj/data/ /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbliip_k10000_wsjtags.pickle -s 3 --both -k 100,500,900,10000 -c 0.7 --guessremainder
echo \"WSJ Tagset\"
#python parkesclustering.py --loadmats  --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/ /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbliip_k10000_wsjtags.pickle -s 3 --both -k 100,500,900,10000 -c 0.7" >> englishbliip_k10000_wsjtags.sh
bash englishbliip_k10000_wsjtags.sh
fi




#English BLIIP; k=1000
if false; then
echo "English BLIIP k1000"
echo -e $header > englishbliip_k1000_wsjtags.sh
echo >> englishbliip_k1000_wsjtags.sh
printf "#python parkesclustering.py --corpus wsj /mnt/nlpgridio3/data/corpora/lorelei/bliip_87_89_wsj/data/ /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbliip_k1000_wsjtags.pickle -k 1000 --both
#python parkesclustering.py --loadmats  --corpus wsj /mnt/nlpgridio3/data/corpora/lorelei/bliip_87_89_wsj/data/ /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbliip_k1000_wsjtags.pickle -s 3 --both -k 100,500,900,1000 -c 0.7 --guessremainder
echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/ /mnt/nlpgridio2/nlp/users/jkodner/lorelei/englishbliip_k1000_wsjtags.pickle -s 3 --both -k 100,500,900,1000 -c 0.7" >> englishbliip_k1000_wsjtags.sh
bash englishbliip_k1000_wsjtags.sh
fi



#English WSJ; k=1000
if true; then
echo "English WSJ k1000"
echo -e $header > englishwsj_k1000_wsjtags.sh
echo >> englishwsj_k1000_wsjtags.sh
printf "#python parkesclustering.py --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/ /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/englishwsj_k1000_wsjtags3.pickle -k 1000 --both
echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/ /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/englishwsj_k1000_wsjtags3.pickle -s 3 --both -k 100,500,900,1000 -c 0.7" >> englishwsj_k1000_wsjtags.sh
bash englishwsj_k1000_wsjtags.sh
fi


#English WSJ; k=10000
if false; then
echo "English WSJ k10000"
echo -e $header > englishwsj_k10000_wsjtags.sh
echo >> englishwsj_k10000_wsjtags.sh
printf "#python parkesclustering_static.py --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/englishwsj_k10000_wsjtags2.pickle -k 10000 --both
echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/englishwsj_k10000_wsjtags2.pickle -s 3 --both -k 900,1000,2000,5000,10000 -c 0.9 --guessremainder" >> englishwsj_k10000_wsjtags.sh
qsub englishwsj_k10000_wsjtags.sh
fi
exit

#English WSJ-00; k=1000
if false; then
echo "English WSJ-00 k1000"
echo -e $header > englishwsj00_k1000_wsjtags.sh
echo >> englishwsj00_k1000_wsjtags.sh
printf "#python parkesclustering.py --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/00 /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/englishwsj00_k1000_wsjtags.pickle -k 1000 --both
echo \"WSJ-00 Tagset\"
python parkesclustering.py --loadmats  --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/00 /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/englishwsj00_k1000_wsjtags.pickle -s 3 --both -k 50,100,300,500,700,900,1000 -c 0.3 --guessremainder" >> englishwsj00_k1000_wsjtags.sh
bash englishwsj00_k1000_wsjtags.sh
fi

#English WSJ-00; k=6777
if false; then
echo "English WSJ-00 k6777"
echo -e $header > englishwsj00_k6777_wsjtags.sh
echo >> englishwsj00_k6777_wsjtags.sh
printf "python parkesclustering_static.py --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/00 /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/englishwsj00_k6777_wsjtags.pickle -k 6777 --both
echo \"WSJ-00 Tagset\"
#python parkesclustering.py --loadmats  --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/00 /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/englishwsj00_k6777_wsjtags.pickle -s 3 --both -k 900,1000,6777 -c 0.9 --guessremainder" >> englishwsj00_k6777_wsjtags.sh
qsub englishwsj00_k6777_wsjtags.sh
fi



#Chinese TB; k=1000
if false; then
echo "Chinese TB k1000"
echo -e $header > chinesetb_k1000_wsjtags.sh
echo >> chinesetb_k1000_wsjtags.sh
printf "#python parkesclustering.py --corpus ctb /mnt/pollux-new/cis/nlp/data/corpora/ctb_ectb_5/ctb5/data/postagged/ /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/chinesetb_k1000_wsjtags.pickle -k 1000 --both
echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus ctb /mnt/pollux-new/cis/nlp/data/corpora/ctb_ectb_5/ctb5/data/postagged/ /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/chinesetb_k1000_wsjtags.pickle -s 3 --both -k 100,500,900,1000 -c 0.7" >> chinesetb_k1000_wsjtags.sh
bash chinesetb_k1000_wsjtags.sh
fi

#Chinese TB; k=10000
if false; then
echo "Chinese TB k10000"
echo -e $header > chinesetb_k10000_wsjtags.sh
echo >> chinesetb_k10000_wsjtags.sh
printf "#python parkesclustering.py --corpus ctb /mnt/pollux-new/cis/nlp/data/corpora/ctb_ectb_5/ctb5/data/postagged /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/chinesetb_k10000_wsjtags.pickle -k 10000 --both
echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus ctb /mnt/pollux-new/cis/nlp/data/corpora/ctb_ectb_5/ctb5/data/postagged /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/chinesetb_k10000_wsjtags.pickle -s 3 --both -k 100,500,900,1000,2000,5000,8000,10000 -c 0.7 --guessremainder" >> chinesetb_k10000_wsjtags.sh
bash chinesetb_k10000_wsjtags.sh
fi













#german; k=10000; no tags
if false; then
echo -e $header > german_k10000_conlltags.sh
echo >> german_k10000_conlltags.sh
printf "#python parkesclustering_static.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/de /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/germanconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/de /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/germanconll_k10000_conlltags.pickle -s 3 --both -k 100,500,1000,2000,5000,10000 -c 1.1" >> german_k10000_conlltags.sh
qsub german_k10000_conlltags.sh
fi

#german; k=1000; no tags
if false; then
echo -e $header > german_k1000_conlltags.sh
echo >> german_k1000_conlltags.sh
printf "#python parkesclustering.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/de /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/germanconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/de /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/germanconll_k1000_conlltags.pickle -s 3 --both -k 100,500,1000 -c 1.1 --guessremainder" >> german_k1000_conlltags.sh
bash german_k1000_conlltags.sh
fi


#japanese; k=10000; no tags
if false; then
echo -e $header > japanese_k10000_conlltags.sh
echo >> japanese_k10000_conlltags.sh
printf "python parkesclustering_static.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ja /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/japaneseconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Conll Tagset\"
#python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ja /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/japaneseconll_k10000_conlltags.pickle -s 3 --both -k 100,500,1000,2000,5000,10000 -c 0.5" >> japanese_k10000_conlltags.sh
qsub japanese_k10000_conlltags.sh
fi

#japanese; k=1000; no tags
if false; then
echo -e $header > japanese_k1000_conlltags.sh
echo >> japanese_k1000_conlltags.sh
printf "python parkesclustering.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ja /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/japaneseconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Japanese Conll Tagset\"
#python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ja /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/japaneseconll_k1000_conlltags.pickle -s 3 --both -k 100,500,900,1000 -c 0.5" >> japanese_k1000_conlltags.sh
qsub japanese_k1000_conlltags.sh
fi


#indonesian; k=10000; no tags
if false; then
echo -e $header > indonesian_k10000_conlltags.sh
echo >> indonesian_k10000_conlltags.sh
printf "#python parkesclustering.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/id /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/indonesianconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Conll Tagset\"
python parkesclustering_static.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/id /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/indonesianconll_k10000_conlltags.pickle -s 3 --both -k 100,500,900,2000,4000,7000,10000 -c 1.0 --guessremainder
#python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/id /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/indonesianconll_k10000_conlltags.pickle -s 3 --both -k 100,500,900,2000,4000,10000 -c 0.5" >> indonesian_k10000_conlltags.sh
bash indonesian_k10000_conlltags.sh
fi

#indonesian; k=1000; no tags
if false; then
echo -e $header > indonesian_k1000_conlltags.sh
echo >> indonesian_k1000_conlltags.sh
printf "#python parkesclustering.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/id /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/indonesianconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Indonesian Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/id /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/indonesianconll_k1000_conlltags.pickle -s 3 --both -k 100,500,900,1000 -c 1.0 --guessremainder" >> indonesian_k1000_conlltags.sh
bash indonesian_k1000_conlltags.sh
fi



#french; k=10000; no tags
if false; then
echo -e $header > french_k10000_conlltags.sh
echo >> french_k10000_conlltags.sh
printf "python parkesclustering_static.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/fr /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/frenchconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"French Conll Tagset\"
#python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/fr /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/frenchconll_k10000_conlltags.pickle -s 3 --both -k 100,500,1000,2000,5000,10000 -c 0.5" >> french_k10000_conlltags.sh
qsub french_k10000_conlltags.sh
fi

#french; k=1000; no tags
if false; then
echo -e $header > french_k1000_conlltags.sh
echo >> french_k1000_conlltags.sh
printf "python parkesclustering.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/fr /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/frenchconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"French Conll Tagset\"
#python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/fr /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/frenchconll_k1000_conlltags.pickle -s 3 --both -k 100,500,900,1000 -c 0.5" >> french_k1000_conlltags.sh
qsub french_k1000_conlltags.sh
fi



#spanish; k=10000; no tags
if false; then
echo -e $header > spanish_k10000_conlltags.sh
echo >> spanish_k10000_conlltags.sh
printf "#python parkesclustering_static.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/es /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/spanishconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Spanish Conll Tagset\"
python parkesclustering_static.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/es /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/spanishconll_k10000_conlltags.pickle -s 3 --both -k 100,500,1000,2000,5000,10000 -c 0.3 --guessremainder" >> spanish_k10000_conlltags.sh
qsub spanish_k10000_conlltags.sh
fi

#spanish; k=1000; no tags
if false; then
echo -e $header > spanish_k1000_conlltags.sh
echo >> spanish_k1000_conlltags.sh
printf "#python parkesclustering.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/es /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/spanishconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Spanish Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/es /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/spanishconll_k1000_conlltags.pickle -s 3 --both -k 100,500,1000 -c 0.3 --guessremainder" >> spanish_k1000_conlltags.sh
bash spanish_k1000_conlltags.sh
fi



#korean; k=10000; no tags
if false; then
echo -e $header > korean_k10000_conlltags.sh
echo >> korean_k10000_conlltags.sh
printf "#python parkesclustering_static.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ko /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/koreanconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Korean Conll Tagset\"
python parkesclustering_static.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ko /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/koreanconll_k10000_conlltags.pickle -s 3 --both -k 100,500,900,1000,2000,5000,10000 -c 0.5" >> korean_k10000_conlltags.sh
qsub korean_k10000_conlltags.sh
fi

#korean; k=1000; no tags
if false; then
echo -e $header > korean_k1000_conlltags.sh
echo >> korean_k1000_conlltags.sh
printf "#python parkesclustering.py --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ko /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/koreanconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Korean Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll /mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ko /mnt/nlpgridio2/nlp/users/lorelei/parkespickles/koreanconll_k1000_conlltags.pickle -s 3 --both -k 100,500,900,1000 -c 0.5 --guessremainder" >> korean_k1000_conlltags.sh
bash korean_k1000_conlltags.sh
fi
