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

#RUN THE FOLLOWING

CREATE_EN_BLIIP_10000=false
CREATE_EN_BROWN_8307=false
CREATE_EN_WSJ_10000=false
CREATE_EN_WSJ00_10000=false

CREATE_CHINESE_10000=false

CREATE_FRENCH_10000=false
CREATE_GERMAN_10000=false
CREATE_INDONESIAN_10000=false
CREATE_JAPANESE_10000=false
CREATE_KOREAN_10000=false
CREATE_SPANISH_10000=false

CREATE_KURDISH_10000=false
CREATE_TAGALOG_10000=false
CREATE_TAMIL_10000=false

CREATE_TURKISH_10000=false


CLUSTER_EN_BLIIP_1000=false
CLUSTER_EN_BLIIP_10000=false
CLUSTER_EN_BROWN_1000=false
CLUSTER_EN_BROWN_8307=false
CLUSTER_EN_WSJ_1000=false
CLUSTER_EN_WSJ_10000=false
CLUSTER_EN_WSJ00_1000=false
CLUSTER_EN_WSJ00_6777=false
CLUSTER_EN_WSJ00_10000=false

CLUSTER_CHINESE_1000=false
CLUSTER_CHINESE_10000=false

CLUSTER_FRENCH_1000=false
CLUSTER_FRENCH_10000=false
CLUSTER_GERMAN_1000=false
CLUSTER_GERMAN_10000=false
CLUSTER_INDONESIAN_1000=false
CLUSTER_INDONESIAN_10000=false
CLUSTER_JAPANESE_1000=false
CLUSTER_JAPANESE_10000=false
CLUSTER_KOREAN_400=false
CLUSTER_KOREAN_1000=false
CLUSTER_KOREAN_10000=false
CLUSTER_SPANISH_1000=false
CLUSTER_SPANISH_10000=false

CLUSTER_KURDISH_400=false
CLUSTER_TAGALOG_400=false
CLUSTER_TAMIL_400=false

CLUSTER_TURKISH_1000=false
CLUSTER_TURKISH_10000=false


#DIRECTORIES

INTERMEDIATE_DIR="/mnt/nlpgridio2/nlp/users/jkodner/lowrespos_data/pickles"

#ENGLISH Corpora
EN_BROWN_DATA="/mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/Brown"
EN_BLIIP_DATA="/mnt/nlpgridio3/data/corpora/lorelei/bliip_87_89_wsj/data/"
EN_WSJ_DATA="/mnt/pollux-new/cis/nlp/data/corpora/wsj/"
EN_WSJ00_DATA="/mnt/pollux-new/cis/nlp/data/corpora/wsj/00"

#CHINESE Treebank Corpus
CHINESE_DATA="/mnt/pollux-new/cis/nlp/data/corpora/ctb_ectb_5/ctb5/data/postagged/"

#UTB Corpora
FRENCH_DATA="/mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/FR"
GERMAN_DATA="/mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/de"
INDONESIAN_DATA="/mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ID"
JAPANESE_DATA="/mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ja"
KOREAN_DATA="/mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/ko"
SPANISH_DATA="/mnt/nlpgridio2/nlp/users/lorelei/uni-dep-tb/universal_treebanks_v2.0/std/es"

#LTF Corpora
KURDISH_DATA="/mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Kurdish_v1.0/POS_Tagged_Text/ltf"
TAGALOG_DATA="/mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Tagalog_v1.0/POS_Tagged_Text/ltf"
TAMIL_DATA="/mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Tamil_v1.0/POS_Tagged_Text/ltf"

#TURKISH TS Corpus
TURKISH_DATA="/mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS"


#k=400; FernAguado; no tags
#python parkesclustering.py --corpus brown /mnt/nlpgridio2/nlp/users/jkodner/lorelei/CHILDES/FernAguado /mnt/nlpgridio2/nlp/users/jkodner/lorelei/spanish_notags.pickle -k 400
#python parkesclustering.py --loadmats /mnt/nlpgridio2/nlp/users/jkodner/lorelei/spanish_notags.pickle

#korean; k=400; no tags
if $CLUSTER_KOREAN_400; then
echo -e $header > $INTERMEDIATE_DIR/korean_notags.sh
echo >> $INTERMEDIATE_DIR/korean_notags.sh
echo "python parkesclustering.py --corpus conll" $KOREAN_DATA $INTERMEDIATE_DIR"/korean_notags.pickle -k 400 --both
#python parkesclustering.py --loadmats" $INTERMEDIATE_DIR"/korean_notags.pickle --both" >> $INTERMEDIATE_DIR/korean_notags.sh
qsub $INTERMEDIATE_DIR/korean_notags.sh
fi

#Tamil; k=400; no tags
if $CLUSTER_TAMIL_400; then
echo -e $header > $INTERMEDIATE_DIR/tamil_notags.sh
echo >> $INTERMEDIATE_DIR/tamil_notags.sh
echo "python parkesclustering.py --corpus lctl /mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Tamil_v1.0/POS_Tagged_Text/ltf" $INTERMEDIATE_DIR"/tamil_notags.pickle -k 400 --both
#python parkesclustering.py --loadmats" $INTERMEDIATE_DIR"/tamil_notags.pickle --both" >> $INTERMEDIATE_DIR/tamil_notags.sh
qsub $INTERMEDIATE_DIR/tamil_notags.sh
fi

#Kurdish; k=400; no tags
if $CLUSTER_KURDISH_400; then
echo -e $header > $INTERMEDIATE_DIR/kurdish_notags.sh
echo >> $INTERMEDIATE_DIR/kurdish_notags.sh
echo "python parkesclustering.py --corpus lctl /mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Kurdish_v1.0/POS_Tagged_Text/ltf" $INTERMEDIATE_DIR"/kurdish_notags.pickle -k 400 --both
#python parkesclustering.py --loadmats" $INTERMEDIATE_DIR"/kurdish_notags.pickle --both" >> $INTERMEDIATE_DIR/kurdish_notags.sh
qsub $INTERMEDIATE_DIR/kurdish_notags.sh
fi

#Tagalog; k=400; no tags
if $CLUSTER_TAGALOG_400; then
echo -e $header > $INTERMEDIATE_DIR/tagalog_notags.sh
echo >> $INTERMEDIATE_DIR/tagalog_notags.sh
echo "python parkesclustering.py --corpus lctl /mnt/nlpgridio3/data/corpora/LDC-language-packs/LCTL_Tagalog_v1.0/POS_Tagged_Text/ltf" $INTERMEDIATE_DIR"/tagalog_notags.pickle -k 400 --both
#python parkesclustering.py --loadmats" $INTERMEDIATE_DIR"/tagalog_notags.pickle --both" >> $INTERMEDIATE_DIR/tagalog_notags.sh
qsub $INTERMEDIATE_DIR/tagalog_notags.sh
fi





#Turkish TS; k=1000
if $CLUSTER_TURKISH_1000; then
echo "Turkish TURKISH k1000"
echo -e $header > $INTERMEDIATE_DIR/turkish_k1000_turkishtags.sh
echo >> $INTERMEDIATE_DIR/turkish_k1000_turkishtags.sh
echo "#python parkesclustering.py --corpus turkishts /mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS/" $INTERMEDIATE_DIR"/turkish_k1000_turkishtags.pickle -k 1000 --both
echo \"TURKISH Tagset\"
python parkesclustering.py --loadmats  --corpus turkishts /mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS" $INTERMEDIATE_DIR"/turkish_k1000_turkishtags.pickle -s 3 --both -k 50,100,500,900,1000 -c 0.9 --guessremainder" >> $INTERMEDIATE_DIR/turkish_k1000_turkishtags.sh
bash $INTERMEDIATE_DIR/turkish_k1000_turkishtags.sh
fi


#Turkish TS; k=10000
if $CLUSTER_TURKISH_10000; then
echo "Turkish TURKISH k10000"
echo -e $header > $INTERMEDIATE_DIR/turkish_k10000_turkishtags.sh
echo >> $INTERMEDIATE_DIR/turkish_k10000_turkishtags.sh
echo "python parkesclustering.py --corpus turkishts /mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS/" $INTERMEDIATE_DIR"/turkish_k10000_turkishtags.pickle -k 10000 --both
echo \"TURKISH Tagset\"
#python parkesclustering.py --loadmats  --corpus turkishts /mnt/nlpgridio2/nlp/users/lorelei/TurkishPOS" $INTERMEDIATE_DIR"/turkish_k10000_turkishtags.pickle -s 3 --both -k 50,100,500,900,10000 -c 0.9" >> $INTERMEDIATE_DIR/turkish_k10000_turkishtags.sh
qsub $INTERMEDIATE_DIR/turkish_k10000_turkishtags.sh
fi





#English Brown; k=1000
if $CLUSTER_EN_BROWN_1000; then
echo "English Brown k1000"
echo -e $header > $INTERMEDIATE_DIR/englishbrown_k1000_browntags.sh
echo >> $INTERMEDIATE_DIR/englishbrown_k1000_browntags.sh
echo "#python parkesclustering_static.py --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k1000_browntags.pickle -k 1000 -s 3 --both
echo \"Brown Tagset -> $INTERMEDIATE_DIR/Reduced Tagset\"
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k1000_browntags.pickle --evalmap brown_to_reduced.txt -s 3 --both -k 100,500,900,1000 -c 0.5 --guessremainder
echo \"***Brown Tagset\"
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k1000_browntags.pickle -s 3 --both -k 100,500,900,1000 -c 0.5 
echo \"***Reduced Tagset S=Carlson\"
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k1000_browntags.pickle --distmap brown_to_reduced.txt -s 0 --both -k 100,500,900,1000 -c 0.9 --guessremainder
echo \"***Reduced Tagset S=3\"
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k1000_browntags.pickle --distmap brown_to_reduced.txt -s 3 --both -k 100,500,900,1000 -c 0.9 --guessremainder
echo \"***Reduced Tagset S=11\"
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k1000_browntags.pickle --distmap brown_to_reduced.txt -s 11 --both -k 100,500,900,1000 -c 0.9 --guessremainder" >> $INTERMEDIATE_DIR/englishbrown_k1000_browntags.sh
qsub $INTERMEDIATE_DIR/englishbrown_k1000_browntags.sh
fi


#English Brown; k=8307
if $CLUSTER_EN_BROWN_8307; then
echo "English Brown k8307"
echo -e $header > $INTERMEDIATE_DIR/englishbrown_k8307_brownreducedtags.sh
echo >> $INTERMEDIATE_DIR/englishbrown_k8307_brownreducedtags.sh
echo "#python parkesclustering_static.py --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k8307_notags.pickle -k 8307 -s 3 --both
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k8307_notags.pickle --evalmap brown_to_reduced.txt -s 3 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.5
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k8307_notags.pickle -s 3 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.5" >> $INTERMEDIATE_DIR/englishbrown_k8307_brownreducedtags.sh
qsub $INTERMEDIATE_DIR/englishbrown_k8307_brownreducedtags.sh #> $INTERMEDIATE_DIR/english-childes_k8307_brown-reduced2.txt

echo -e $header > $INTERMEDIATE_DIR/englishbrown_k8307_reducedtags.sh
echo >> $INTERMEDIATE_DIR/englishbrown_k8307_reducedtags.sh
echo "echo \"***Reduced Tagset S=Carlson\"
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k8307_notags.pickle --distmap brown_to_reduced.txt -s 0 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.9
echo \"***Reduced Tagset S=3\"
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k8307_notags.pickle --distmap brown_to_reduced.txt -s 11 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.9
echo \"***Reduced Tagset S=11\"
python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k8307_notags.pickle --distmap brown_to_reduced.txt -s 3 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.9" >> $INTERMEDIATE_DIR/englishbrown_k8307_reducedtags.sh
qsub $INTERMEDIATE_DIR/englishbrown_k8307_reducedtags.sh #> $INTERMEDIATE_DIR/english-childes_k8307_reduced2.txt

echo -e $header > $INTERMEDIATE_DIR/englishbrown_k8307_browntags.sh
echo >> $INTERMEDIATE_DIR/englishbrown_k8307_browntags.sh
echo "python parkesclustering.py --loadmats  --corpus brown" $EN_BROWN_DATA $INTERMEDIATE_DIR"/englishbrown_k8307_notags.pickle -s 3 --both -k 100,500,900,1000,2000,4000,6000,8307 -c 0.5" >> $INTERMEDIATE_DIR/englishbrown_k8307_browntags.sh
qsub $INTERMEDIATE_DIR/englishbrown_k8307_browntags.sh #> $INTERMEDIATE_DIR/english-childes_k8307_brown2.txt
fi


#English BLIIP; k=10000
if $CREATE_EN_BLIIP_10000; then
echo "English BLIIP k10000"
echo -e $header > $INTERMEDIATE_DIR/englishbliip_k10000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/englishbliip_k10000_wsjtags.sh
echo "python parkesclustering.py --corpus wsj" $EN_BLIIP_DATA $INTERMEDIATE_DIR"/englishbliip_k10000_wsjtags.pickle -k 10000 --both" >> $INTERMEDIATE_DIR/englishbliip_k10000_wsjtags.sh
qsub $INTERMEDIATE_DIR/englishbliip_k10000_wsjtags.sh
fi

if $CLUSTER_EN_BLIIP_10000; then
echo "English BLIIP k10000"
echo -e $header > $INTERMEDIATE_DIR/englishbliip_k10000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/englishbliip_k10000_wsjtags.sh
echo "echo \"WSJ Tagset\"
#python parkesclustering.py --loadmats  --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/" $INTERMEDIATE_DIR"/englishbliip_k10000_wsjtags.pickle -s 3 --both -k 100,500,900,10000 -c 0.7" >> $INTERMEDIATE_DIR/englishbliip_k10000_wsjtags.sh
bash $INTERMEDIATE_DIR/englishbliip_k10000_wsjtags.sh
fi




#English BLIIP; k=1000
if $CLUSTER_EN_BLIIP_1000; then
echo "English BLIIP k1000"
echo -e $header > $INTERMEDIATE_DIR/englishbliip_k1000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/englishbliip_k1000_wsjtags.sh
echo "#python parkesclustering.py --corpus wsj" $EN_BLIIP_DATA $INTERMEDIATE_DIR"/englishbliip_k1000_wsjtags.pickle -k 1000 --both
#python parkesclustering.py --loadmats  --corpus wsj" $EN_BLIIP_DATA $INTERMEDIATE_DIR"/englishbliip_k1000_wsjtags.pickle -s 3 --both -k 100,500,900,1000 -c 0.7 --guessremainder
echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus wsj /mnt/pollux-new/cis/nlp/data/corpora/wsj/" $INTERMEDIATE_DIR"/englishbliip_k1000_wsjtags.pickle -s 3 --both -k 100,500,900,1000 -c 0.7" >> $INTERMEDIATE_DIR/englishbliip_k1000_wsjtags.sh
bash $INTERMEDIATE_DIR/englishbliip_k1000_wsjtags.sh
fi



#WSJ
if $CREATE_EN_WSJ_10000; then
echo "English WSJ k10000"
echo -e $header > $INTERMEDIATE_DIR/englishwsj_k10000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/englishwsj_k10000_wsjtags.sh
echo "python parkesclustering_static.py --corpus wsj" $EN_WSJ_DATA $INTERMEDIATE_DIR"/englishwsj_k10000_wsjtags.pickle -k 10000 --both" >> $INTERMEDIATE_DIR/englishwsj_k10000_wsjtags.sh
qsub $INTERMEDIATE_DIR/englishwsj_k10000_wsjtags.sh
fi


#English WSJ; k=1000
if $CLUSTER_EN_WSJ_1000; then
echo "English WSJ k1000"
echo -e $header > $INTERMEDIATE_DIR/englishwsj_k1000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/englishwsj_k1000_wsjtags.sh
echo "echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus wsj" $EN_WSJ_DATA $INTERMEDIATE_DIR"/englishwsj_k1000_wsjtags3.pickle -s 3 --both -k 100,500,900,1000 -c 0.7" >> $INTERMEDIATE_DIR/englishwsj_k1000_wsjtags.sh
bash $INTERMEDIATE_DIR/englishwsj_k1000_wsjtags.sh
fi


#English WSJ; k=10000
if $CLUSTER_EN_WSJ_10000; then
echo "English WSJ k10000"
echo -e $header > $INTERMEDIATE_DIR/englishwsj_k10000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/englishwsj_k10000_wsjtags.sh
echo "echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus wsj" $EN_WSJ_DATA $INTERMEDIATE_DIR"/englishwsj_k10000_wsjtags2.pickle -s 3 --both -k 900,1000,2000,5000,10000 -c 0.9 --guessremainder" >> $INTERMEDIATE_DIR/englishwsj_k10000_wsjtags.sh
qsub $INTERMEDIATE_DIR/englishwsj_k10000_wsjtags.sh
fi


#English WSJ-00; k=1000
if $CLUSTER_EN_WSJ00_1000; then
echo "English WSJ-00 k1000"
echo -e $header > $INTERMEDIATE_DIR/englishwsj00_k1000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/englishwsj00_k1000_wsjtags.sh
echo "echo \"WSJ-00 Tagset\"
python parkesclustering.py --loadmats  --corpus wsj" $EN_WSJ00_DATA $INTERMEDIATE_DIR"/englishwsj00_k1000_wsjtags.pickle -s 3 --both -k 50,100,300,500,700,900,1000 -c 0.3 --guessremainder" >> $INTERMEDIATE_DIR/englishwsj00_k1000_wsjtags.sh
bash $INTERMEDIATE_DIR/englishwsj00_k1000_wsjtags.sh
fi

#English WSJ-00; k=6777
if $CLUSTER_EN_WSJ00_6777; then
echo "English WSJ-00 k6777"
echo -e $header > $INTERMEDIATE_DIR/englishwsj00_k6777_wsjtags.sh
echo >> $INTERMEDIATE_DIR/englishwsj00_k6777_wsjtags.sh
echo "echo \"WSJ-00 Tagset\"
#python parkesclustering.py --loadmats  --corpus wsj" $EN_WSJ00_DATA $INTERMEDIATE_DIR"/englishwsj00_k6777_wsjtags.pickle -s 3 --both -k 900,1000,6777 -c 0.9 --guessremainder" >> $INTERMEDIATE_DIR/englishwsj00_k6777_wsjtags.sh
qsub $INTERMEDIATE_DIR/englishwsj00_k6777_wsjtags.sh
fi



#Chinese TB; k=1000
if $CLUSTER_CHINESE_1000; then
echo "Chinese TB k1000"
echo -e $header > $INTERMEDIATE_DIR/chinesetb_k1000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/chinesetb_k1000_wsjtags.sh
echo "#python parkesclustering.py --corpus ctb" $CHINESE_DATA $INTERMEDIATE_DIR"/chinesetb_k1000_wsjtags.pickle -k 1000 --both
echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus ctb" $CHINESE_DATA $INTERMEDIATE_DIR"/chinesetb_k1000_wsjtags.pickle -s 3 --both -k 100,500,900,1000 -c 0.7" >> $INTERMEDIATE_DIR/chinesetb_k1000_wsjtags.sh
bash $INTERMEDIATE_DIR/chinesetb_k1000_wsjtags.sh
fi

#Chinese TB; k=10000
if $CLUSTER_CHINESE_10000; then
echo "Chinese TB k10000"
echo -e $header > $INTERMEDIATE_DIR/chinesetb_k10000_wsjtags.sh
echo >> $INTERMEDIATE_DIR/chinesetb_k10000_wsjtags.sh
echo "#python parkesclustering.py --corpus ctb" $CHINESE_DATA $INTERMEDIATE_DIR"/chinesetb_k10000_wsjtags.pickle -k 10000 --both
echo \"WSJ Tagset\"
python parkesclustering.py --loadmats  --corpus ctb" $CHINESE_DATA $INTERMEDIATE_DIR"/chinesetb_k10000_wsjtags.pickle -s 3 --both -k 100,500,900,1000,2000,5000,8000,10000 -c 0.7 --guessremainder" >> $INTERMEDIATE_DIR/chinesetb_k10000_wsjtags.sh
bash $INTERMEDIATE_DIR/chinesetb_k10000_wsjtags.sh
fi













#german; k=10000; no tags
if $CLUSTER_GERMAN_10000; then
echo -e $header > $INTERMEDIATE_DIR/german_k10000_conlltags.sh
echo >> $INTERMEDIATE_DIR/german_k10000_conlltags.sh
echo "#python parkesclustering_static.py --corpus conll" $GERMAN_DATA $INTERMEDIATE_DIR"/germanconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll" $GERMAN_DATA $INTERMEDIATE_DIR"/germanconll_k10000_conlltags.pickle -s 3 --both -k 100,500,1000,2000,5000,10000 -c 1.1" >> $INTERMEDIATE_DIR/german_k10000_conlltags.sh
qsub $INTERMEDIATE_DIR/german_k10000_conlltags.sh
fi

#german; k=1000; no tags
if $CLUSTER_GERMAN_1000; then
echo -e $header > $INTERMEDIATE_DIR/german_k1000_conlltags.sh
echo >> $INTERMEDIATE_DIR/german_k1000_conlltags.sh
echo "#python parkesclustering.py --corpus conll" $GERMAN_DATA $INTERMEDIATE_DIR"/germanconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll" $GERMAN_DATA $INTERMEDIATE_DIR"/germanconll_k1000_conlltags.pickle -s 3 --both -k 100,500,1000 -c 1.1 --guessremainder" >> $INTERMEDIATE_DIR/german_k1000_conlltags.sh
bash $INTERMEDIATE_DIR/german_k1000_conlltags.sh
fi


#japanese; k=10000; no tags
if $CLUSTER_JAPANESE_10000; then
echo -e $header > $INTERMEDIATE_DIR/japanese_k10000_conlltags.sh
echo >> $INTERMEDIATE_DIR/japanese_k10000_conlltags.sh
echo "python parkesclustering_static.py --corpus conll" $JAPANESE_DATA $INTERMEDIATE_DIR"/japaneseconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Conll Tagset\"
#python parkesclustering.py --loadmats  --corpus conll" $JAPANESE_DATA $INTERMEDIATE_DIR"/japaneseconll_k10000_conlltags.pickle -s 3 --both -k 100,500,1000,2000,5000,10000 -c 0.5" >> $INTERMEDIATE_DIR/japanese_k10000_conlltags.sh
qsub $INTERMEDIATE_DIR/japanese_k10000_conlltags.sh
fi

#japanese; k=1000; no tags
if $CLUSTER_JAPANESE_1000; then
echo -e $header > $INTERMEDIATE_DIR/japanese_k1000_conlltags.sh
echo >> $INTERMEDIATE_DIR/japanese_k1000_conlltags.sh
echo "python parkesclustering.py --corpus conll" $JAPANESE_DATA $INTERMEDIATE_DIR"/japaneseconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Japanese Conll Tagset\"
#python parkesclustering.py --loadmats  --corpus conll" $JAPANESE_DATA $INTERMEDIATE_DIR"/japaneseconll_k1000_conlltags.pickle -s 3 --both -k 100,500,900,1000 -c 0.5" >> $INTERMEDIATE_DIR/japanese_k1000_conlltags.sh
qsub $INTERMEDIATE_DIR/japanese_k1000_conlltags.sh
fi


#indonesian; k=10000; no tags
if $CLUSTER_INDONESIAN_10000; then
echo -e $header > $INTERMEDIATE_DIR/indonesian_k10000_conlltags.sh
echo >> $INTERMEDIATE_DIR/indonesian_k10000_conlltags.sh
echo "#python parkesclustering.py --corpus conll" $INDONESIAN_DATA $INTERMEDIATE_DIR"/indonesianconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Conll Tagset\"
python parkesclustering_static.py --loadmats  --corpus conll" $INDONESIAN_DATA $INTERMEDIATE_DIR"/indonesianconll_k10000_conlltags.pickle -s 3 --both -k 100,500,900,2000,4000,7000,10000 -c 1.0 --guessremainder
#python parkesclustering.py --loadmats  --corpus conll" $INDONESIAN_DATA $INTERMEDIATE_DIR"/indonesianconll_k10000_conlltags.pickle -s 3 --both -k 100,500,900,2000,4000,10000 -c 0.5" >> $INTERMEDIATE_DIR/indonesian_k10000_conlltags.sh
bash $INTERMEDIATE_DIR/indonesian_k10000_conlltags.sh
fi

#indonesian; k=1000; no tags
if $CLUSTER_INDONESIAN_1000; then
echo -e $header > $INTERMEDIATE_DIR/indonesian_k1000_conlltags.sh
echo >> $INTERMEDIATE_DIR/indonesian_k1000_conlltags.sh
echo "#python parkesclustering.py --corpus conll" $INDONESIAN_DATA $INTERMEDIATE_DIR"/indonesianconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Indonesian Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll" $INDONESIAN_DATA $INTERMEDIATE_DIR"/indonesianconll_k1000_conlltags.pickle -s 3 --both -k 100,500,900,1000 -c 1.0 --guessremainder" >> $INTERMEDIATE_DIR/indonesian_k1000_conlltags.sh
bash $INTERMEDIATE_DIR/indonesian_k1000_conlltags.sh
fi



#french; k=10000; no tags
if $CLUSTER_FRENCH_10000; then
echo -e $header > $INTERMEDIATE_DIR/french_k10000_conlltags.sh
echo >> $INTERMEDIATE_DIR/french_k10000_conlltags.sh
echo "python parkesclustering_static.py --corpus conll" $FRENCH_DATA $INTERMEDIATE_DIR"/frenchconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"French Conll Tagset\"
#python parkesclustering.py --loadmats  --corpus conll" $FRENCH_DATA $INTERMEDIATE_DIR"/frenchconll_k10000_conlltags.pickle -s 3 --both -k 100,500,1000,2000,5000,10000 -c 0.5" >> $INTERMEDIATE_DIR/french_k10000_conlltags.sh
qsub $INTERMEDIATE_DIR/french_k10000_conlltags.sh
fi

#french; k=1000; no tags
if $CLUSTER_FRENCH_1000; then
echo -e $header > $INTERMEDIATE_DIR/french_k1000_conlltags.sh
echo >> $INTERMEDIATE_DIR/french_k1000_conlltags.sh
echo "python parkesclustering.py --corpus conll" $FRENCH_DATA $INTERMEDIATE_DIR"/frenchconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"French Conll Tagset\"
#python parkesclustering.py --loadmats  --corpus conll" $FRENCH_DATA $INTERMEDIATE_DIR"/frenchconll_k1000_conlltags.pickle -s 3 --both -k 100,500,900,1000 -c 0.5" >> $INTERMEDIATE_DIR/french_k1000_conlltags.sh
qsub $INTERMEDIATE_DIR/french_k1000_conlltags.sh
fi



#spanish; k=10000; no tags
if $CLUSTER_SPANISH_10000; then
echo -e $header > $INTERMEDIATE_DIR/spanish_k10000_conlltags.sh
echo >> $INTERMEDIATE_DIR/spanish_k10000_conlltags.sh
echo "#python parkesclustering_static.py --corpus conll" $SPANISH_DATA $INTERMEDIATE_DIR"/spanishconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Spanish Conll Tagset\"
python parkesclustering_static.py --loadmats  --corpus conll" $SPANISH_DATA $INTERMEDIATE_DIR"/spanishconll_k10000_conlltags.pickle -s 3 --both -k 100,500,1000,2000,5000,10000 -c 0.3 --guessremainder" >> $INTERMEDIATE_DIR/spanish_k10000_conlltags.sh
qsub $INTERMEDIATE_DIR/spanish_k10000_conlltags.sh
fi

#spanish; k=1000; no tags
if $CLUSTER_SPANISH_1000; then
echo -e $header > $INTERMEDIATE_DIR/spanish_k1000_conlltags.sh
echo >> $INTERMEDIATE_DIR/spanish_k1000_conlltags.sh
echo "#python parkesclustering.py --corpus conll" $SPANISH_DATA $INTERMEDIATE_DIR"/spanishconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Spanish Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll" $SPANISH_DATA $INTERMEDIATE_DIR"/spanishconll_k1000_conlltags.pickle -s 3 --both -k 100,500,1000 -c 0.3 --guessremainder" >> $INTERMEDIATE_DIR/spanish_k1000_conlltags.sh
bash $INTERMEDIATE_DIR/spanish_k1000_conlltags.sh
fi



#korean; k=10000; no tags
if $CLUSTER_KOREAN_10000; then
echo -e $header > $INTERMEDIATE_DIR/korean_k10000_conlltags.sh
echo >> $INTERMEDIATE_DIR/korean_k10000_conlltags.sh
echo "#python parkesclustering_static.py --corpus conll" $KOREAN_DATA $INTERMEDIATE_DIR"/koreanconll_k10000_conlltags.pickle -k 10000 -s 3 --both 
echo \"Korean Conll Tagset\"
python parkesclustering_static.py --loadmats  --corpus conll" $KOREAN_DATA $INTERMEDIATE_DIR"/koreanconll_k10000_conlltags.pickle -s 3 --both -k 100,500,900,1000,2000,5000,10000 -c 0.5" >> $INTERMEDIATE_DIR/korean_k10000_conlltags.sh
qsub $INTERMEDIATE_DIR/korean_k10000_conlltags.sh
fi

#korean; k=1000; no tags
if $CLUSTER_KOREAN_1000; then
echo -e $header > $INTERMEDIATE_DIR/korean_k1000_conlltags.sh
echo >> $INTERMEDIATE_DIR/korean_k1000_conlltags.sh
echo "#python parkesclustering.py --corpus conll" $KOREAN_DATA $INTERMEDIATE_DIR"/koreanconll_k1000_conlltags.pickle -k 1000 -s 3 --both
echo \"Korean Conll Tagset\"
python parkesclustering.py --loadmats  --corpus conll" $KOREAN_DATA $INTERMEDIATE_DIR"/koreanconll_k1000_conlltags.pickle -s 3 --both -k 100,500,900,1000 -c 0.5 --guessremainder" >> $INTERMEDIATE_DIR/korean_k1000_conlltags.sh
bash $INTERMEDIATE_DIR/korean_k1000_conlltags.sh
fi
