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

source $1

#set in config file
#MESSAGE:      Printed at top of output
#INTERMED_DIR: Where intermediate outputs should go
#DATA_DIR:     Where to read in from  
#CREATE:       Whether or not to create initial similarity matrices
#CORPUS:       Corpus type
#NUM_SEEDS:    Number of seeds
#THRESHOLD:    Confidence threshold
#K_SEQUENCE:   Sequence of k to classify
#SUBMIT:       Whether to run on NLPGrid or not (treated as true when CREATE == false)
#SIM_FILE:     Location of similarity file within INTERMED_DIR
#SCRIPT_FILE:  Location of output script within INTERMED_DIR
#OTHER_ARGS:   Any other specific arguments

echo $MESSAGE
echo CREATING? $CREATE
echo SIM FILE $INTERMED_DIR/$SIM_FILE
echo SCRIPT FILE $INTERMED_DIR/$SCRIPT_FILE

echo -e $header > $INTERMED_DIR/$SCRIPT_FILE
echo >> $INTERMED_DIR/$SCRIPT_FILE
echo "echo" $MESSAGE >> $INTERMED_DIR/$SCRIPT_FILE

if $CREATE; then
    echo python parkesclustering.py --corpus $CORPUS $DATA_DIR $INTERMED_DIR/$SIM_FILE -k $K_SEQUENCE -s $NUM_SEEDS --both $OTHER_ARGS >> $INTERMED_DIR/$SCRIPT_FILE
    qsub $INTERMED_DIR/$SCRIPT_FILE
else
    echo python parkesclustering.py --loadmats --corpus $DATA_DIR $INTERMED_DIR/$SIM_FILE -k $K_SEQUENCE -s $NUM_SEEDS --both $OTHER_ARGS -c $THRESHOLD >> $INTERMED_DIR/$SCRIPT_FILE
    if $SUBMIT; then
	qsub $INTERMED_DIR/$SCRIPT_FILE
    else
	bash $INTERMED_DIR/$SCRIPT_FILE
    fi
fi



