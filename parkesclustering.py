import argparse
import sys, os, os.path
from threading import Thread
import time
from math import log
from matplotlib import pyplot as plt
import numpy as np
from scipy import spatial
import scipy.sparse as sparse
import scipy.stats as stats
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as hierarchy
import re, copy, random
from collections import defaultdict
import cPickle as pickle
import sklearn.cluster as cluster
import numpy.matlib as matlib

START = "#START"
STOP = "#STOP"

SKIP_TAG = "SKIP"
PUNCT_TAGS = set([])

CONF_THRESHOLD = 0.0

WSJ_FINDWORD = re.compile(r"\(([\w\d\.,\?!]+)\s([A-Za-z\d'-\.,\?!]+?)\)")

carlsonseeds = {"arm":{"NN":1},"ate":{"VB":1},"baby":{"NN":1},"balloon":{"NN":1},"bed":{"NN":1},"big":{"JJ":1},"black":{"JJ":1},"break":{"VB":1},"climb":{"VB":1},"come":{"VB":1},"cookie":{"NN":1},"crying":{"VB":1},"cut":{"VB":1},"dad":{"NN":1},"dirty":{"JJ":1},"do":{"VB":1},"down":{"RB":1},"eat":{"VB":1},"face":{"NN":1},"fall":{"VB":1},"goes":{"VB":1},"happy":{"JJ":1},"have":{"VB":1},"hide":{"VB":1},"him":{"PRP":1},"hold":{"VB":1},"hug":{"NN":1},"hurt":{"VB":1},"in":{"IN":1},"juice":{"NN":1},"make":{"VB":1},"milk":{"NN":1},"mom":{"NN":1},"move":{"VB":1},"not":{"RB":1},"off":{"RB":1},"on":{"RB":1},"one":{"DT":1},"put":{"VB":1},"run":{"VB":1},"sad":{"JJ":1},"say":{"VB":1},"sleep":{"VB":1},"small":{"JJ":1},"spoon":{"NN":1},"this":{"DT":1},"three":{"DT":1},"through":{"RB":1},"throw":{"VB":1},"to":{"IN":1},"today":{"RB":1},"tomorrow":{"RB":1},"train":{"NN":1},"tree":{"NN":1},"turn":{"NN":1},"two":{"DT":1},"up":{"RB":1},"very":{"RB":1},"walk":{"VB":1},"watch":{"VB":1},"water":{"NN":1},"by":{"IN":1},"cake":{"NN":1},"call":{"VB":1},"car":{"NN":1},"carry":{"VB":1},"chair":{"NN":1},"clean":{"JJ":1},"flower":{"NN":1},"fly":{"VB":1},"from":{"IN":1},"girl":{"NN":1},"give":{"VB":1},"kick":{"VB":1},"last":{"JJ":1},"later":{"RB":1},"look":{"VB":1},"out":{"RB":1},"play":{"VB":1},"stand":{"VB":1},"there":{"RB":1},"we":{"PRP":1},"wet":{"JJ":1},"who":{"PRP":1},"work":{"NN":1},"you":{"PRP":1}}


def set_puncttags(corpus):
    global PUNCT_TAGS
    if corpus.lower() == "conll":
        PUNCT_TAGS = set(["_","x","mad","mid","pad"])

def tag_word(wordtup, keeptag):
    if keeptag:
        return wordtup[0] + "_" + wordtup[1]
    else:
        return wordtup[1]

def check_tokenlimit(maxtokens, numtokens):
    if maxtokens and numtokens > maxtokens:
        print "REACHED TOKEN LIMIT"
        print "Num Tokens:\t", numtokens
        return True
    return False

def read_turkishtsfile(filename, tags, freqs, contexts, numtokens, maxtokens):
    findword = re.compile(r"([^\s]+)_([^\s]+)")
    with open(filename, "r") as f:
        prevprev = START
        prev = START
        for line in f:
            if len(freqs) >= 50000:
                break
            words = findword.findall(line.lower())
            for wordtup in words:
                if "//" in wordtup[0]:
                    continue
                word = tag_word((wordtup[1],wordtup[0]),False).decode("utf-8").lower()
                numtokens += 1
                if check_tokenlimit(maxtokens, numtokens):
                    return tags, freqs, contexts, numtokens

#                if word == u"\u0130mparatorlu\u011fu'nu":
#                    print "GOT IT"
                if word not in tags:
                    tags[word] = {}
                if wordtup[1] not in tags[word]:
                    tags[word][wordtup[1]] = 0
                tags[word][wordtup[1]] += 1
                freqs[word] += 1
                contexts[prev].add((prevprev,word))
                #print word, wordtup[1]
                #print prevprev, prev, word
#                print prevprev.encode("utf-8"), prev.encode("utf-8"), word.encode("utf-8"), "\t", prev.encode("utf-8"), contexts[prev]
                prevprev = prev
                prev = word

            contexts[prev].add((prevprev,STOP))
            prevprev = START
            prev = START

#        print set([tag for tagdict in tags.values() for tag in tagdict])

    print len(freqs), sum(freqs.values())
    return tags, freqs, contexts, numtokens


def read_ctbfile(filename, tags, freqs, contexts, numtokens, maxtokens):
    findword = re.compile(r"([^\s]+)_([^\s]+)")
    with open(filename, "r") as f:
        prevprev = START
        prev = START
        for line in f:
            if "</S>" in line or "<S ID" in line or "<P>" in line or "</P>" in line or "HEADER>" in line or "DATE>" in line or "BODY>" in line or "DOCID>" in line or "HEADLINE>" in line or "DOC>" in line or "TEXT>" in line:
                continue

#            print "\n", line.strip()
            words = findword.findall(line.lower())
            for wordtup in words:
                if wordtup[1].lower() == "-none-":
                    continue
                word = tag_word((wordtup[1],wordtup[0]),False).lower()
                numtokens += 1
                if check_tokenlimit(maxtokens, numtokens):
                    return tags, freqs, contexts, numtokens

                if word not in tags:
                    tags[word] = {}
                if wordtup[1] not in tags[word]:
                    tags[word][wordtup[1]] = 0
                tags[word][wordtup[1]] += 1
                freqs[word] += 1
                contexts[prev].add((prevprev,word))
 #               print prevprev, prev, word
#                print prevprev.encode("utf-8"), prev.encode("utf-8"), word.encode("utf-8"), "\t", prev.encode("utf-8"), contexts[prev]
                prevprev = prev
                prev = word

            contexts[prev].add((prevprev,STOP))
            prevprev = START
            prev = START

#        print set([tag for tagdict in tags.values() for tag in tagdict])

    return tags, freqs, contexts, numtokens


def read_ctbfile_eval(filename, numtokens, maxtokens):
    findword = re.compile(r"([^\s]+)_([^\s]+)")
    pospairs = []
    with open(filename, "r") as f:
        for line in f:
            if "</S>" in line or "<S ID" in line or "<P>" in line or "</P>" in line or "HEADER>" in line or "DATE>" in line or "BODY>" in line or "DOCID>" in line or "HEADLINE>" in line or "DOC>" in line or "TEXT>" in line:
                continue
            words = findword.findall(line.lower())
            if not words:
                continue
            pospairs.extend([(tag_word((wordtup[1],wordtup[0]),False),wordtup[1]) for wordtup in words if wordtup[1].lower() != "-none-"])
            numtokens += len(words)
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs, numtokens


def read_turkishtsfile_eval(filename, numtokens, maxtokens):
    findword = re.compile(r"([^\s]+)_([^\s]+)")
    pospairs = []
    types = set([])
    with open(filename, "r") as f:
        for line in f:
            words = findword.findall(line.lower())
            if not words:
                continue
            if len(types) >= 50000:
                break
            for wordtup in words:
                types.add(wordtup[0])
            pospairs.extend([(tag_word((wordtup[1],wordtup[0]),False).decode("utf-8").lower(),wordtup[1]) for wordtup in words if wordtup[1].lower() != "-none-"])
            numtokens += len(words)
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs, numtokens



def read_wsjfile(filename, tags, freqs, contexts, numtokens, maxtokens):
#    findword = re.compile(r"\((\w+)\s([a-z'-]+)\)")
    findword = WSJ_FINDWORD
    with open(filename, "r") as f:
        prevprev = START
        prev = START
        savedend = False
        for line in f:
            savedend = False
            if not line.strip():
                savedend = True
#            if line.strip() == "(. .) ))":
                contexts[prev].add((prevprev,STOP))
#                print prevprev, prev, STOP, "\t", prev, contexts[prev]
                prevprev = START
                prev = START

            words = findword.findall(line.lower())
            for i, wordtup in enumerate(words):
                word = tag_word(wordtup,False).decode("utf-8").lower()
                if ")" in word:
                    print word, line
                numtokens += 1
                if check_tokenlimit(maxtokens, numtokens):
                    return tags, freqs, contexts, numtokens

                if word not in tags:
                    tags[word] = {}
                if wordtup[0] not in tags[word]:
                    tags[word][wordtup[0]] = 0
                tags[word][wordtup[0]] += 1
                freqs[word] += 1
                contexts[prev].add((prevprev,word))
#                print prevprev, prev, word, "\t", prev, contexts[prev]
                prevprev = prev
                prev = word
        if not savedend:
            contexts[prev].add((prevprev,STOP))
            prevprev = START
            prev = START

#        print set([tag for tagdict in tags.values() for tag in tagdict])

    return tags, freqs, contexts, numtokens


def read_wsjfile_eval(filename, numtokens, maxtokens):
    findword = WSJ_FINDWORD
    pospairs = []
    with open(filename, "r") as f:
        for line in f:
            words = findword.findall(line.lower())
            for wordtup in words:
                word = wordtup[1]
                if ")" in word:
                    print word

            if not words:
                continue
            pospairs.extend([(tag_word(wordtup,False).decode("utf-8"),wordtup[0]) for wordtup in words])
            numtokens += len(words)
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs, numtokens


def read_childesbrownfile_eval(filename, numtokens, maxtokens):
    findword = re.compile(r"([A-Za-z:]+)\|([^\s]+)")
    pospairs = []
    with open(filename, "r") as f:
        for line in f:
            if line[0:5] != "%mor:":
                continue

            words = findword.findall(line.lower())
            pospairs.extend([(tag_word(wordtup,False).decode("utf-8"),wordtup[0]) for wordtup in words])
            numtokens += len(words)
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs, numtokens


def read_childesbrownfile(filename, tags, freqs, contexts, numtokens, maxtokens):
    findword = re.compile(r"([A-Za-z:]+)\|([^\s]+)")
    with open(filename, "r") as f:
        prevprev = START
        prev = START
        for line in f:
            if line[0:5] != "%mor:":
                continue

            words = findword.findall(line.lower())
            for wordtup in words:
                word = tag_word(wordtup,False).decode("utf-8")
                numtokens += 1
                if check_tokenlimit(maxtokens, numtokens):
                    return tags, freqs, contexts, numtokens

                if word not in tags:
                    tags[word] = {}
                if wordtup[0] not in tags[word]:
                    tags[word][wordtup[0]] = 0
                tags[word][wordtup[0]] += 1
                #try:
                 #   if CHILDES_TAGMAP[wordtup[0]] == "SKIP":
                  #      prevprev = prev
                   #     prev = word
                #        continue
                #    tags[word][CHILDES_TAGMAP[wordtup[0]]] += 1
                #except KeyError:
                #    prevprev = prev
                #    prev = word
                #    continue
#                    tags[word]["SKIP"] += 1
                freqs[word] += 1
                contexts[prev].add((prevprev,word))
#                print prevprev, prev, word, "\t", prev, contexts[prev]
                prevprev = prev
                prev = word

            contexts[prev].add((prevprev,STOP))

    return tags, freqs, contexts, numtokens


def read_conlluniversalfile_eval(filename, numtokens, maxtokens):
    findword = re.compile(r"^\d+\t(.+?)\t.+?\t(\w+)")
    pospairs = []
    with open(filename, "r") as f:
        for line in f:
            words = findword.findall(line.lower())
            if not words:
                continue
            pospairs.extend([(tag_word((wordtup[1],wordtup[0]),False).decode("utf-8").lower(),wordtup[1]) for wordtup in words])
            numtokens += len(words)
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs, numtokens


def read_conlluniversalfile(filename, tags, freqs, contexts, numtokens, maxtokens):
    findword = re.compile(r"^\d+\t(.+?)\t.+?\t(\w+)")
#    findword = re.compile(r"\d+\t(.+?)\t.+?\t(\w+)\t(\w+)")
    with open(filename, "r") as f:
        prevprev = START
        prev = START
        ended = False
        for line in f:
            if not line.strip():
                ended = True
                contexts[prev].add((prevprev,STOP))
#                print prevprev, prev, STOP, "\t", prev, contexts[prev]
                prevprev = START
                prev = START
                continue

            ended = False
            words = findword.findall(line.lower())
            for wordtup in words:
                word = tag_word((wordtup[1],wordtup[0]),False).decode("utf-8").lower()
                numtokens += 1
                if check_tokenlimit(maxtokens, numtokens):
                    return tags, freqs, contexts, numtokens

                if word not in tags:
                    tags[word] = defaultdict(int)
                tags[word][wordtup[1]] += 1
                freqs[word] += 1
                contexts[prev].add((prevprev,word))
#                print prevprev, prev, word, "\t", prev, contexts[prev]
                prevprev = prev
                prev = word
        contexts[prev].add((prevprev,STOP))
        prevprev = START
        prev = START

    return tags, freqs, contexts, numtokens


def read_lctlfile(filename, tags, freqs, contexts, numtokens, maxtokens):
    findword = re.compile(r"""pos="(\w+)">(.+?)</""")
    with open(filename, "r") as f:
        prevprev = START
        prev = START
        for line in f:
            if "</SEG>" in line:
                contexts[prev].add((prevprev,STOP))
#                print prevprev, prev, STOP, "\t", prev, contexts[prev]
                prevprev = START
                prev = START
                continue
            elif "<TOKEN id=" not in line:
                continue

            words = findword.findall(line.lower())
            for wordtup in words:
                word = tag_word((wordtup[1],wordtup[0]),False).decode("utf-8")
                numtokens += 1
                if check_tokenlimit(maxtokens, numtokens):
                    return tags, freqs, contexts, numtokens

                freqs[word] += 1
                if word not in tags:
                    tags[word] = defaultdict(int)
                tags[word][wordtup[0]] += 1
                contexts[prev].add((prevprev,word))
#                print prevprev, prev, word, "\t", prev, contexts[prev]
                prevprev = prev
                prev = word

    return tags, freqs, contexts, numtokens



def read_corpusdirfiles(basedir, corpustype, maxtokens):
    tags = {}
    freqs = defaultdict(int)
    contexts = defaultdict(lambda : set([]))
    numtokens = 0
    for subdir, dirs, files in os.walk(basedir):
        if check_tokenlimit(maxtokens, numtokens):
            break
        for f in files:
            if check_tokenlimit(maxtokens, numtokens):
                break
            if corpustype == "wsj":
                tags, freqs, contexts, numtokens = read_wsjfile(os.path.join(subdir,f), tags, freqs, contexts, numtokens, maxtokens)
            if corpustype == "ctb":
                tags, freqs, contexts, numtokens = read_ctbfile(os.path.join(subdir,f), tags, freqs, contexts, numtokens, maxtokens)
            if corpustype == "turkishts":
                tags, freqs, contexts, numtokens = read_turkishtsfile(os.path.join(subdir,f), tags, freqs, contexts, numtokens, maxtokens)
            elif corpustype == "brown":
                tags, freqs, contexts, numtokens = read_childesbrownfile(os.path.join(subdir,f), tags, freqs, contexts, numtokens, maxtokens)
            elif corpustype == "conll":
                tags, freqs, contexts, numtokens = read_conlluniversalfile(os.path.join(subdir,f), tags, freqs, contexts, numtokens, maxtokens)
            elif corpustype == "lctl":
                tags, freqs, contexts, numtokens = read_lctlfile(os.path.join(subdir,f), tags, freqs, contexts, numtokens, maxtokens)
#    tags = {k:dict(v) for k, v in tags.iteritems()}
    freqs = dict(freqs)
    contexts = dict(contexts)
    return tags, freqs, contexts


def load_tagmap(tagmapfname):
    tagmap = {}
    with open(tagmapfname, "r") as f:
        for line in f:
            if line:
                comps = line.decode("utf-8").strip().split("\t")
                tagmap[comps[0]] = comps[1]
    return tagmap


def load_ersatz(corpustype, outfile):
    with open(outfile+".remainder.pickle", 'rb') as instream:
        distances = pickle.load(instream)
        seeds = pickle.load(instream)
    return seeds, distances


def evaluate_tokens(basedir, corpustype, assignments, evaltagmap, disttagmap, maxtokens, seeds=None, ngram=False, outfile=None):
    numallpospairs = 0
    numpospairs = 0
    numcorrect = 0

#    def prob_to_type(rangedict, rando):
#        for typerange, type in rangedict.iteritems():
#            if typerange[0] <= rando and rando < typerange[1]:
#                return type
#        raise Exception("You picked a stupid rando")
#        return

#    probranges = {}
#    for word, tags in assignments.iteritems():
#        rangedict = {}
#        lowerbound = 0
#        for tag, prob in tags.iteritems():
#            rangedict[(lowerbound, lowerbound + prob)] = tag
#            lowerbound += prob
#        probranges[word] = rangedict

    if ngram:
        allpospairs, goldallpospairs = crf_tag(basedir, corpustype, assignments, seeds, outfile)
        allpospairs, goldallpospairs = ngram_tag(basedir, corpustype, assignments, seeds, outfile)
        pospairs = [pair for pair in allpospairs if pair[0] in assignments]
        goldpospairs = [pair for pair in goldallpospairs if pair[0] in assignments]
        numallpospairs = len(allpospairs)
        numpospairs = len(pospairs)
        for i, pair in enumerate(allpospairs):
            word = pair[0]
            pos = pair[1]
            goldpos = goldallpospairs[i][1]
            if pos:
                if evaltagmap[disttagmap[pos]] == evaltagmap[goldpos]:
                    numcorrect += 1
    else:
        numtokens = 0
        for subdir, dirs, files in os.walk(basedir):
            if check_tokenlimit(maxtokens, numtokens):
                break
            for f in files:
                if check_tokenlimit(maxtokens, numtokens):
                    break
                if corpustype == "brown":
                    allpospairs, numtokens = read_childesbrownfile_eval(os.path.join(subdir,f), numtokens, maxtokens)
                elif corpustype == "wsj":
                    allpospairs, numtokens = read_wsjfile_eval(os.path.join(subdir,f), numtokens, maxtokens)
                elif corpustype == "ctb":
                    allpospairs, numtokens = read_ctbfile_eval(os.path.join(subdir,f), numtokens, maxtokens)
                elif corpustype == "turkishts":
                    allpospairs, numtokens = read_turkishtsfile_eval(os.path.join(subdir,f), numtokens, maxtokens)
                elif corpustype == "conll":
                    allpospairs, numtokens = read_conlluniversalfile_eval(os.path.join(subdir,f), numtokens, maxtokens)
                
                if check_tokenlimit(maxtokens, numtokens):
                    allpospairs = allpospairs[0:maxtokens]
                pospairs = [pair for pair in allpospairs if pair[0] in assignments]
                numallpospairs += len(allpospairs)
                numpospairs += len(pospairs)

                for pair in pospairs:
                    word = pair[0]
                    pos = pair[1]
                    maxtag = max(assignments[word], key=assignments[word].get)
                    #if len(assgnlist) > 1:
                    #    print word, dict(assignments[word])
                    if maxtag == "UNK":
                        continue
                    if evaltagmap[disttagmap[pos]] == evaltagmap[maxtag]:
                        numcorrect += 1

                    #if assignments[word][maxtag] > CONF_THRESHOLD:
                    #    if evaltagmap[disttagmap[pos]] == evaltagmap[maxtag]:
                    #        numcorrect += 1
                    #elif evaltagmap[disttagmap[pos]] == evaltagmap[prob_to_type(probranges[word], random.random())]:
                    #    numcorrect += 1

    print "1-to-1 TOKEN ACCURACY"
    print "All tokens\t\t", numallpospairs
    print "Top k tokens\t\t", numpospairs
    print "Correct tokens\t\t", numcorrect
    print "% all correct\t\t", float(numcorrect)/numallpospairs*100
    print "% top k correct\t\t", float(numcorrect)/numpospairs*100

def print_confusions(confusion_map):
    for tag, confusions in confusion_map.iteritems():
        print tag
        for ctag, count in confusions.iteritems():
            print "\t" + ctag + "\t" + str(count)

def evaluate_types(assignments, seedwords, correcttags, evaltagmap):
    numassgn = 0
    numseeds = 0
    numtotal = 0
    numcorrect_assgn = 0
    numcorrect_seeds = 0
    numcorrect_total = 0

    confusion_map = defaultdict(lambda : defaultdict(int))
    multi_map = defaultdict(lambda : defaultdict(int))

    for word, assgn in assignments.iteritems():

        keep = False
        for tag, val in correcttags[word].iteritems():
            if tag not in PUNCT_TAGS:
                keep = True
        if not keep:
            continue

        if word in seedwords:
            numseeds += 1
        else:
            numassgn += 1
        correcttagdict = {}
        for tag, val in correcttags[word].iteritems():
#            print word, tag, val
            if val == 0:
                continue
            newtag = evaltagmap[tag]
            if newtag not in correcttagdict:
                correcttagdict[newtag] = 0
            correcttagdict[newtag] += val

        if list(assgn)[0] == "UNK":
            continue
        maxtag = max(assgn, key=assgn.get)
        if evaltagmap[maxtag] in correcttagdict:
            if word in seedwords:
                numcorrect_seeds += 1
            else:
                numcorrect_assgn += 1
            for tag in correcttagdict:
                multi_map[tag][maxtag] += 1
        for tag in correcttagdict:
            confusion_map[tag][maxtag] += 1
            
    numtotal = numseeds + numassgn
    numcorrect_total = numcorrect_seeds + numcorrect_assgn

    print "Num seeds\t\t", numseeds
    print "Num seeds correct\t", numcorrect_seeds
    print "Num assigned\t\t", numassgn
    print "Num assgn correct\t", numcorrect_assgn
    print "Num total\t\t", numtotal
    print "Num total correct\t", numcorrect_total

    if numseeds == 0:
        numseeds = 1
    if numassgn == 0:
        numassgn = 1
    if numtotal == 0:
        numtotal = 1
    print "% seeds correct\t\t", float(numcorrect_seeds)/numseeds*100
    print "% assgn correct\t\t", float(numcorrect_assgn)/numassgn*100
    print "% total correct\t\t", float(numcorrect_total)/numtotal*100
    print ""

#    print "Confusion pairs"
#    print_confusions(confusion_map)
#    print ""

#    print "Ambiguity pairs"
#    print_confusions(multi_map)
#    print ""


def get_seedwordsunambig(wordsbytagfreq, seedsperpos):
    def sortfunc(x, POS):
        if POS in x[1] and len(x[1]) == 1:
            return x[1][POS]
        return 0


    seedwords_byPOS = {}
    seedwords_byword = {}
    POSes = set([])
    for word, tags in wordsbytagfreq.iteritems():
        POSes = POSes.union(set(tags.keys()))
    try:
        POSes.remove(SKIP_TAG)
    except KeyError:
        pass
    for tag in PUNCT_TAGS:
        try:
            POSes.remove(tag)
        except KeyError:
            pass

    for POS in POSes:
        sortedwords = [word for word in sorted(wordsbytagfreq.iteritems(), key = lambda x: sortfunc(x,POS), reverse=True) if POS in word[1] and len(word[1]) == 1]
#        print POS, len(sortedwords)
#        print [word for word in sorted(wordsbytagfreq.iteritems(), key = lambda x: sortfunc(x,POS), reverse=True) if POS in word[1] and len(word[1]) == 1][0:seedsperpos]
#        print "\n\n"
        seedwords_byPOS[POS] = {word:POS for word, POS in sortedwords[0:seedsperpos]}

    for POS, words in seedwords_byPOS.iteritems():
        for word, wordPOSes in words.iteritems():
            seedwords_byword[word] = wordPOSes

    finalPOSes = seedwords_byPOS.keys()

    print len(finalPOSes), len(seedwords_byword)
    print seedwords_byPOS
    return finalPOSes, seedwords_byword


def get_seedwords(wordsbytagfreq, seedsperpos):
    def sortfunc(x, POS):
        if POS in x[1]:
            return x[1][POS]
        return 0

    seedwords_byPOS = {}
    seedwords_byword = {}
    POSes = set([])
    for word, tags in wordsbytagfreq.iteritems():
        POSes = POSes.union(set(tags.keys()))

    try:
        POSes.remove(SKIP_TAG)
    except KeyError:
        pass
    for tag in PUNCT_TAGS:
        try:
            POSes.remove(tag)
        except KeyError:
            pass
    for POS in POSes:
        sortedwords = [word[0] for word in sorted(wordsbytagfreq.iteritems(), key = lambda x: sortfunc(x,POS), reverse=True)][0:seedsperpos]
        seedwords_byPOS[POS] = sortedwords
        for word in sortedwords:
            if POS in wordsbytagfreq[word]:
                if word not in seedwords_byword:
                    seedwords_byword[word] = {}
                seedwords_byword[word][POS] = wordsbytagfreq[word][POS]
        
        #print POS, sortedwords
    return POSes, seedwords_byword

def get_topk(freqs, k):
    sortedwords = [pair[0] for pair in sorted(freqs.iteritems(), key = lambda x: x[1], reverse=True)][0:k]
    return sortedwords


def get_indexdict(contexts, sides, words_topm=None):
    index = 0
    indexdict = {}
    for wordcontexts in contexts:
        for context in wordcontexts:
            for side in sides:
                if str(side)+context[side] not in indexdict:
                    if not words_topm or context[side] in words_topm:
                        indexdict[str(side)+context[side]] = index
                        index +=1
    return indexdict

def get_contextdict(words, contexts, indices, sides):
    contextdict = {}
    for word in words:
        contextvec = np.repeat(0.0000001, len(indices))
        for context in contexts[word]:
            for side in sides:
                if str(side)+context[side] not in indices:
                    continue
                try:
                    contextvec[indices[str(side)+context[side]]] += 1
                except KeyError:
                    pass
        contextdict[word] = contextvec

    return contextdict



def get_contextvecs(words_topk, contexts, words_topm=None):
    if words_topm:
        words_topm = set(words_topm)
    leftindexdict = get_indexdict(contexts.values(),(0,), words_topm)
    rightindexdict = get_indexdict(contexts.values(),(1,), words_topm)

    leftcontextvecs = get_contextdict(words_topk, contexts, leftindexdict, (0,))
    rightcontextvecs = get_contextdict(words_topk, contexts, rightindexdict, (1,))

    return leftcontextvecs, rightcontextvecs


def get_contextvecs_both(words_topk, contexts, words_topm=None):
    if words_topm:
        words_topm = set(words_topm)
    bothindexdict = get_indexdict(contexts.values(),(0,1), words_topm)
    bothcontextvecs = get_contextdict(words_topk, contexts, bothindexdict, (0,1))
    return bothcontextvecs


def get_contextvecs_suffix(words_topk):
    index = 0
    indexdict = {}
    BILEN = 2
    TRILEN = 3
    for word in words_topk:
        if len(word) > BILEN+1:
            indexdict[word[0-BILEN:]] = index
            index += 1
        if len(word) > TRILEN+1:
            indexdict[word[0-TRILEN:]] = index
            index += 1

    suffdict = {}
    wordtrirootdict = {}
    wordbirootdict = {}
    contextvec = np.repeat(0.0000001, max(indexdict.values())+1)
    for word in words_topk:
        if len(word) > BILEN+1:
            root = word[:0-BILEN]
            if root not in suffdict:
                suffdict[root] = contextvec.copy()
            suff = word[0-BILEN:]
            suffdict[root][indexdict[suff]] += 1
            wordbirootdict[word] = root
        if len(word) > TRILEN+1:
            root = word[:0-TRILEN]
            if root not in suffdict:
                suffdict[root] = contextvec.copy()
            suff = word[0-TRILEN:]
            suffdict[root][indexdict[suff]] += 1
            wordtrirootdict[word] = root


    contextdict = {}
    for word in words_topk:
        try:
            tricontexts = suffdict[wordtrirootdict[word]]
        except KeyError:
            tricontexts = contextvec.copy()
        try:
            bicontexts = suffdict[wordbirootdict[word]]
        except KeyError:
            bicontexts = contextvec.copy()

        contextdict[word] = np.hstack((tricontexts,bicontexts))

    return contextdict


def merge_contextvecs(vecs1, vecs2):
    mergedvecs = {}

    for word in vecs1:
        mergedvecs[word] = np.hstack((vecs1[word],vecs2[word]))
#        print word, vecs1[word].shape, vecs2[word].shape, mergedvecs[word].shape
#    print type(vecs1[word]), type(vecs2[word]), type(mergedvecs[word])
#    print vecs1[word].shape, vecs2[word].shape, mergedvecs[word].shape

    return mergedvecs


def calc_KL_ndarray2(vec1, vec2, word1, word2):
    norm1 = vec1 / np.sum(vec1)
    norm2 = vec2 / np.sum(vec2)
    qp = np.sum(norm1 * np.log(norm1 / norm2), 0)
    pq = np.sum(norm2 * np.log(norm2 / norm1), 0)
    return qp+pq


def calc_KL_ndarray(vec1, vec2, word1, word2):
    mat1 = np.transpose(np.asmatrix(vec1))
    mat2 = np.transpose(np.asmatrix(vec2))
    qp = sum(stats.entropy(pk=mat1, qk=mat2))
    pq = sum(stats.entropy(pk=mat2, qk=mat1))
    return qp+pq


def calc_distances(contextvecs, words_topk, distfunc):
    distances = np.ndarray((len(words_topk),len(words_topk)))
    for i, word1 in enumerate(words_topk):
        for j, word2 in enumerate(words_topk):
            if j > i:
                break
            distances[i,j] = distfunc(contextvecs[word1], contextvecs[word2], word1, word2)
            distances[j,i] = distances[i,j]

    return distance.squareform(distances)


def calc_distances_multi(contextvecs, words_topk, distfunc, numsplits):
    distances = np.zeros((len(words_topk),len(words_topk)))

    def worker(lmin, lmax, rmin, rmax):
        #print lmin, lmax, rmin, rmax
        for i, word1 in enumerate(words_topk):
            if i < lmin:
                continue
            if i >= lmax:
                break
            for j, word2 in enumerate(words_topk):
                if j < rmin:
                    continue
                if j >= rmax:
                    break
                if j > i:
                    break
                if i != j:
                    distances[i,j] = distfunc(contextvecs[word1], contextvecs[word2], word1, word2)
                else:
                    distances[i,j] = 0.0
                distances[j,i] = distances[i,j]

    q, r = divmod(len(words_topk), numsplits)
    indices = [q*i + min(i, r) for i in xrange(numsplits+1)]
    #print indices

    splitsize = len(words_topk) / numsplits
    lindex = 0
    threads = []
    for i in xrange(numsplits):
        for j in xrange(numsplits):
            if j > i:
                continue
            threads.append(Thread(target=worker, args=(indices[i], indices[i+1], indices[j], indices[j+1])))
            #worker(indices[i], indices[i+1], indices[j], indices[j+1])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return distance.squareform(distances)


def make_tree(distances, method):
    tree = hierarchy.linkage(distances, method=method)
    return tree


def plot_dendrogram(tree, title, outname, namelist, k, seedwords):
    def label_func(leafnum):
        word = namelist[leafnum]
        if word in seedwords:
#            print "SEED", word
            return "*** " + word
        return word

    plt.figure(figsize=(int(0.25*k), int(0.2*k)))
    plt.title(title)
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hierarchy.dendrogram(
        tree,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        leaf_label_func = label_func,
        distance_sort='descending',
    )
    plt.savefig(outname)
    plt.show()
#    print(title + " Leaf Initial Merges")
#    for row in range(tree[:,0].size):
#        if tree[row][0] < 400 and tree[row][1] < 400:
#            print(words_topk[int(tree[row][0])], words_topk[int(tree[row][1])])
#    print("\n")

def load_tagsfile(fname):
    with open(fname, "r") as f:
        return set([tag.strip() for tag in f])


def assign_POS(tree, namelist, seedwords, evaltagmap, prevassignments, prevconfidences, distances):
    nodetags = defaultdict(lambda : defaultdict(int))
    nodemembers = defaultdict(lambda : set([]))
    nodeconfidences = {}

    unkbestconfs = defaultdict(int)
    unkbesttags = {}

    use_purity = True

    #merge tag counts from two joined subtrees
    def merge_dicts(ldict, rdict):
        mergeddict = copy.deepcopy(ldict)
        for tag, count in rdict.iteritems():
            if tag not in mergeddict:
                mergeddict[tag] = 0
            mergeddict[tag] += count
        return mergeddict

    #perform assignment from known subtree to unknown subtree
    def assign_tree(knowntreetags, knowntreemembers, unknowntreetags, unknowntreemembers):
        maxtag = max(knowntreetags, key=knowntreetags.get)
        confidence = float(knowntreetags[maxtag])/sum(knowntreetags.values())
        for member in unknowntreemembers.union(knowntreemembers):
            if len(nodetags[member]) == 0:
                if confidence < 0:
#                if confidence < CONF_THRESHOLD:
                    if member not in unkbestconfs:
                        unkbestconfs[member] = confidence
                        unkbesttags[member] = maxtag
                    elif confidence > unkbestconfs[member]:
                        if unkbesttags[member] != maxtag:
                            print "more confidence", namelist[member], "\t", unkbestconfs[member], unkbesttags[member], confidence, maxtag
                        unkbestconfs[member] = confidence
                        unkbesttags[member] = maxtag
                else:
                    unkbestconfs.pop(member, None)
                    unkbesttags.pop(member, None)
                if use_purity:
                    unknowntreetags[maxtag] += 1
                    nodetags[member][maxtag] += 1
                    #factor = sum(knowntreetags.values())
                    #normedtags = {}
                    #for tag, count in knowntreetags.iteritems():
                    #    normedtags[tag] = float(count)/factor
                    #unknowntreetags = {tag:count*len(unknowntreemembers) for tag, count in normedtags.iteritems()}
                    #nodetags[member] = normedtags
                    nodeconfidences[member] = confidence
                else: 
                    if confidence >= CONF_THRESHOLD/2:
                        mindistance = 100000000
                        mindisttag = maxtag
                        seenknownmembers = set([])
                        for tag in knowntreetags:
                            distsum = 0
                            comparisoncount = 0.0
                            for kmember in knowntreemembers:
                                if kmember in seenknownmembers:
                                    continue
                                if namelist[kmember] not in seedwords:
                                    continue
    #                            seenknownmembers.add(kmember)
                                kmembermaxtag = max(nodetags[kmember], key=nodetags[kmember].get)
                                if tag == kmembermaxtag:
                                    comparisoncount += 1
                                    for umember in unknowntreemembers:
                                        distsum += distances[kmember,umember]
                            if comparisoncount:
                                distance = distsum / comparisoncount / comparisoncount
                                if distance < mindistance and tag != "." and tag != ",":
                                    mindistance = distance
                                    mindisttag = tag
     #                               print mindisttag, mindistance
                            else:
                                print "nocount", tag
                            #print "maxtag:\t", maxtag
                            #print "assigning:\t", mindisttag
                        unknowntreetags[mindisttag] += 1
                        nodetags[member][mindisttag] += 1
                    else:
                        print "LOW CONFIDENCE"
                        for umember in unknowntreemembers:
                            mindist = 10000000
                            mindisttag = maxtag
                            for kmember in knowntreemembers:
                                if namelist[kmember] not in seedwords:
                                    continue
                                distance = distances[kmember,umember]
                                ktag = max(nodetags[kmember], key=nodetags[kmember].get)
                                if distance < mindist and ktag != ",":
                                    mindisttag = ktag
                                    mindist = distance
                                unknowntreetags[mindisttag] += 1
                                nodetags[member][mindisttag] += 1
                    nodeconfidences[member] = confidence
#                    print ""


    #initialize with seeds
    for i, word in enumerate(namelist):
        nodemembers[i].add(i)
        nodeconfidences[i] = 1.0
        if word in seedwords:
            maxtag = max(seedwords[word], key=seedwords[word].get)
            #normedtags = {}
            #factor = sum(seedwords[word].values())
            #for tag, count in seedwords[word].iteritems():
            #    normedtags[tag] = float(count)/factor
            #nodetags[i] = normedtags
            nodetags[i][maxtag] = 1

    #iterate through joins
    newnodebase = len(namelist)
    for i,row in enumerate(tree):
        lefttreei = int(row[0])
        righttreei = int(row[1])
        lefttreetags = nodetags[lefttreei]
        righttreetags = nodetags[righttreei]
        lefttreemembers = nodemembers[lefttreei]
        righttreemembers = nodemembers[righttreei]
        if lefttreetags and not righttreetags: #left tree members have tags but right tree's don't
            assign_tree(lefttreetags, lefttreemembers, righttreetags, righttreemembers)
        elif righttreetags and not lefttreetags: #right tree members have tags but left tree's don't
            assign_tree(righttreetags, righttreemembers, lefttreetags, lefttreemembers)
        nodetags[newnodebase+i] = merge_dicts(lefttreetags, righttreetags)
        nodemembers[newnodebase+i] = lefttreemembers.union(righttreemembers)

    #penalize UNKs
    numunknown = 0
    numunknowncorrect = 0
    for i in range(len(namelist)):
        if i in unkbesttags:
            besttag = unkbesttags[i]
            if list(nodetags[i].keys())[0] != besttag:
                print "UNKNOWN", namelist[i], nodetags[i], besttag
            nodetags[i] = {besttag:1}
#            nodetags[i][besttag] += 1

    assignments = {namelist[i]:tag for i, tag in nodetags.iteritems() if i < len(namelist)}
    confidences = {namelist[i]:conf for i, conf in nodeconfidences.iteritems() if i < len(namelist)}

    for word, prevconf in prevconfidences.iteritems():
        if prevconf > confidences[word]:
            maxtag = max(assignments[word], key=assignments[word].get)
            prevmaxtag = max(prevassignments[word], key=prevassignments[word].get)
#            if maxtag != prevmaxtag:
#                print word, "\t", prevconf, prevmaxtag, "\t", confidences[word], maxtag
            confidences[word] = prevconf
            assignments[word] = prevassignments[word]

    return assignments, confidences


def update_seedwords(seedwords_bywords, assignments, confidences):
    for word, tags in assignments.iteritems():
        if confidences[word] >= CONF_THRESHOLD:
            seedwords_bywords[word] = tags
    return seedwords_bywords


def assign_remainder(inputdir, corpus, words, k, m, seeds, outfile, oldassignments, maxtokens):
    tags, freqs, contexts = read_corpusdirfiles(inputdir, corpus, maxtokens)
    words_topk = words[0:k]
    words_topm = None
    if m:
        words_topm = words[0:min(len(words),m)]
    topkset = set(words_topk)
    outside_topk = set([word for word in freqs if word not in topkset])
    words = outside_topk.union(set(seeds.keys()))
    contextvecs= get_contextvecs_both(freqs, contexts, words_topm=words_topm)
    seedmaxtags = {seed:max(seeds[seed], key=seeds[seed].get) for seed in seeds}

    assignments = {}
    distances = {}
    print "Total Remaining Words", len(words)
    print "Original Seeds", len(seeds)
    def worker(numsplits, index):
        for i, word in enumerate(freqs):
            if (i+index) % numsplits != 0:
                continue
            if not i % (1000):
                print i
    #        if i >= 1000:
    #            break
            distances[word] = {}
            mindistance = None
            besttag = "UNK"
            for seed in seeds:
                distance = calc_KL_ndarray2(contextvecs[word], contextvecs[seed], word, seed)
                distances[word][seed] = distance
                if mindistance == None or distance < mindistance:
                    mindistance = distance
                    besttag = seedmaxtags[seed]
            if word in seeds or word in topkset:
                continue
            assignments[word] = {besttag:1}
    
    numsplits = 16
    splitsize = len(freqs) / numsplits
    lindex = 0
    threads = []
    for i in xrange(numsplits):
        threads.append(Thread(target=worker, args=(numsplits,i)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(outfile+".remainder.pickle", 'wb') as outstream:
        pickle.dump(distances, outstream)
        pickle.dump(seeds, outstream)
        print "SAVED"

    return assignments

def combinedistances(dists1, dists2, topkwords):
    normed1 = distance.squareform(dists1 / np.amax(dists1))
    normed2 = distance.squareform(dists2 / np.amax(dists2))

    distances = np.zeros((len(topkwords),len(topkwords)))

    for i, word in enumerate(topkwords):
        for j, word in enumerate(topkwords):
            distances[i,j] = (1.0/(i+j+1)*normed1) + ((1-(1.0/(i+j+1)))*normed2)
            distances[j,i] = distances[i,j]

#    for i in range(0,len(wordsbyfreq)):
#        for j in range(0,len(wordsbyfreq)):
#            if distances[i,j] != distances[j,i]:
#                print i, j, "\t", distances[i,j], distances[j,i], square1[i,j] == square1[j,i], square2[i,j] == square2[j,i]
    return distances.squareform(distances)

def create_matrices(inputdir, corpus, datafile, k, m, ktagsfname, mtagsfname, segment_suffixes, maxtokens):
    print "Reading corpus files from", inputdir
    tags, freqs, contexts = read_corpusdirfiles(inputdir, corpus, maxtokens)
    print "Total types in corpus\t", len(freqs)
    print "Total tokens in corpus\t", sum(freqs.values())

    if ktagsfname:
        tagset = load_tagsfile(ktagsfname)
        words_topk = get_tagset_topk(freqs, k, tagset)
    else:
        words_topk = get_topk(freqs, k)

    tokensaccounted = sum([freqs[word] for word in freqs if word in words_topk])
    print "Types Accounted for in top k, (%)\t", k, float(k)/len(freqs)*100
    print "Tokens Accounted for in top k, (%)\t", tokensaccounted, float(tokensaccounted)/sum(freqs.values())*100

    words_topm = None
    if m > 0:
        if mtagsfname:
            tagset = load_tagsfile(mtagsfname)
            words_topm = get_tagset_topk(freqs, m, tagset)
        else:
            words_topm = get_topk(freqs, m)

    if not m:
        m = len(freqs.keys())
    print "Calculating", k, "context vectors of length", m*2
    leftdist = None
    rightdist = None
    bothdist = None

    bothcontextvecs = get_contextvecs_both(words_topk, contexts, words_topm=words_topm)
    if segment_suffixes:
        suffcontextvecs = get_contextvecs_suffix(words_topk)
        bothcontextvecs = merge_contextvecs(bothcontextvecs, suffcontextvecs)


    print "Calculating vector distances..."
    start = time.time()
    bothdist = calc_distances_multi(bothcontextvecs, words_topk, calc_KL_ndarray2, 16)
#    bothdist = combinedistances(bothdist1, bothdist2, words_topk)
#bothdist = calc_distances_multi(bothcontextvecs, words_topk, calc_KL_ndarray2, 16)
    end = time.time()
    print "\ttook", (end - start), "seconds"
    #print("Calculating both vector distances 4")
    #start = time.time()
    #bothdist2 = calc_distances(bothcontextvecs, words_topk, calc_KL_ndarray2)
    #end = time.time()
    #print (end - start)
    #for i, bd in enumerate(bothdist2):
    #    if round(bothdist[i],6) != round(bothdist2[i],6):
    #        print "ERROR 1,2", i, bothdist[i], bothdist2[i]

    print "Saving to", datafile
    with open(datafile, 'wb') as outstream:
        pickle.dump(tags, outstream)
        pickle.dump(words_topk, outstream)
        pickle.dump(words_topm, outstream)
        pickle.dump(leftdist, outstream)
        pickle.dump(rightdist, outstream)
        pickle.dump(bothdist, outstream)
    print("Finished data tabulations.\n")


def load_matrices(datafile, seedsperPOS, seedwords, evalmap, distmap):
    print "\n\nLoading saved matrices from", datafile
    with open(datafile, "rb") as instream:
        correcttags = pickle.load(instream)
        words_topk = pickle.load(instream)
        words_topm = pickle.load(instream)
        leftdist = pickle.load(instream)
        rightdist = pickle.load(instream)
        bothdist = distance.squareform(distance.squareform(pickle.load(instream)))
#    print set([tag for tags in correcttags.values() for tag in tags])


    if not distmap:
        distmap = {}
        for tagdict in correcttags.values():
            for tag in tagdict:
                distmap[tag] = tag

    mappedtags = {}
    for word, oldtags in correcttags.iteritems():
        mappedtagdict = {}
        for tag, count in oldtags.iteritems():
            mappedtag = distmap[tag]
            if mappedtag not in mappedtagdict:
                mappedtagdict[mappedtag] = 0
            mappedtagdict[mappedtag] += count
        mappedtags[word] = mappedtagdict

    if not evalmap:
        evalmap = {}
        for tagdict in mappedtags.values():
            for tag in tagdict:
                evalmap[tag] = tag

    if not seedwords:
#        POSes, seedwords = get_seedwordsunambig(mappedtags, seedsperPOS)
        POSes, seedwords = get_seedwords(mappedtags, seedsperPOS)
        print "Num tags\t\t", len(POSes)
        print "Max possible seeds\t", len(seedwords)
        print "\n"

    return mappedtags, words_topk, bothdist, distmap, evalmap, seedwords


def assign(mappedtags, words, k, distances, seedwords, evalmap, prevassignments, prevconfidences):
    words_topk = words[0:k]
    distances_topk = distance.squareform(distance.squareform(distances)[0:k,0:k])

    #methods = ['single','complete', 'weighted', 'average']
    methods = ['average']
    for method in methods:
        print "MULTI-to-1 TYPE ACCURACY:\tk=", k
#        fileprefix = datafile.replace(".pickle","")

        bothtree = make_tree(distances_topk, method)

        assignments, confidences = assign_POS(bothtree, words_topk, seedwords, evalmap, prevassignments, prevconfidences, distance.squareform(distances_topk))

        c, coph_dists = hierarchy.cophenet(bothtree, distances_topk)
        print "Cophenetic Correlation:", c
        evaluate_types(assignments, seedwords, mappedtags, evalmap)
#        if k > 400:
#            plot_dendrogram(bothtree, 'Right+Left Context Dendrogram', "both_dendrogram.png", words_topk, len(words_topk), seedwords)
#            exit()

        return assignments, confidences, update_seedwords(seedwords, assignments, confidences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Parkes-style Single-Word Context Clustering")

    parser.add_argument("inputdir", nargs="?", help="corpus directory; must be provided if not --loadmats")
    parser.add_argument("--corpus", nargs="?", help="corpus type: [conll, ctb, lctl, wsj, turkish]")
    parser.add_argument("datafile", help="file containing/to contain stored vocabulary, contexts, and distances")
    parser.add_argument("--loadmats", help="load stored vocabulary, contexts, distances", action="store_true")
    parser.add_argument("-k", "--numclustertypes", help="cluster top k words", type=str)
    parser.add_argument("-s", "--numseedsperpos", help="number of seed words per POS", type=int)
    parser.add_argument("-m", "--numcontexttypes", nargs="?", help="filter out all but top m context words", type=int)    
    parser.add_argument("-t", "--numtokens", nargs="?", help="set a maximum number of tokens to train on. Only used when creating", type=int)
    parser.add_argument("--ktagsfile", nargs="?", help="only keep cluster words with these tags", type=str)
    parser.add_argument("--mtagsfile", nargs="?", help="only keep context words with these tags", type=str)
    parser.add_argument("--both", help="compute a single tree on both contexts rather than one for each", action="store_true")
    parser.add_argument("--evalmap", nargs="?", help="map training tags to eval tags", type=str)
    parser.add_argument("--distmap", nargs="?", help="map input tags to training tags", type=str)
    parser.add_argument("-c", "--confidence", help="seed confidence threshold", type=float)
    parser.add_argument("--guessremainder", help="apply simple classification for tokens outside the top k", action="store_true")
    parser.add_argument("--nopunctseeds", help="don't assign seeds to punctuation tags but leave punctuation in place otherwise", action="store_true")
    parser.add_argument("--segsuff", help="segment off suffixes and classify the roots instead", action="store_true")

    args = parser.parse_args()

    if args.inputdir and not args.corpus:
        exit("Must specify --corpus along with inputdir")
    elif not args.inputdir and not args.loadmats:
        exit("If no input directory is provided, must use --loadmats")
    elif not args.numclustertypes and not args.loadmats:
        exit("error2")
    elif args.loadmats and (args.ktagsfile or args.mtagsfile):
        exit("error3")
#    elif args.inputdir and args.loadmats:
#        exit("error4")
    elif args.corpus and args.corpus.lower() not in set(["wsj","brown", "conll", "lctl", "ctb", "turkishts"]):
        exit("Invalid corpus type")
    elif args.numseedsperpos < 0 and args.loadmats:
        exit("error5")

#    global CONF_THRESHOLD
    CONF_THRESHOLD = args.confidence

    if not args.loadmats:
        create_matrices(args.inputdir, args.corpus, args.datafile, int(args.numclustertypes), args.numcontexttypes, args.ktagsfile, args.mtagsfile, args.segsuff, args.numtokens)
    else:
        if args.nopunctseeds:
            print "EXCLUDING PUNCTUATION FROM SEED SET (NOT SCORED IN TYPE ACC; ALL WRONG IN TOKEN ACC)"
            if args.corpus.lower() != "conll":
                exit("not implemented yet")
            set_puncttags(args.corpus)

        print "CONFIDENCE", CONF_THRESHOLD

        distmap = None
        if args.distmap:
            distmap = load_tagmap(args.distmap)

        evalmap = None
        if args.evalmap:
            evalmap = load_tagmap(args.evalmap)

        originalseeds = None
        seeds = None
        tags, words, distances, distmap, evalmap, seedwords = load_matrices(args.datafile, args.numseedsperpos, seeds, evalmap, distmap)
        if args.numseedsperpos == 0:
            seedwords = {seed:tags for seed, tags in carlsonseeds.iteritems() if seed in words}
            print len(seedwords)
        originalseeds = copy.deepcopy(seedwords)

        assignments = {}
        confidences = {}
        incr = 10
#        print tags, distances, originalseeds

        for k in args.numclustertypes.split(","):
#        for k in np.arange(50,int(args.numclustertypes.split(",")[-1])+incr,incr):
            assignments, confidences, seeds = assign(tags, words, int(k), distances, seedwords, evalmap, assignments, confidences)
        for word in seedwords:
            maxtag = max(seedwords[word], key=seedwords[word].get)
            assignments[word] = {maxtag:seedwords[word][maxtag]}

        if args.guessremainder:
            print "Reading corpus files from", args.inputdir
            remainder_assignments = assign_remainder(args.inputdir, args.corpus, words, int(k), 10000, originalseeds, args.datafile, assignments, args.numtokens)
            both_assignments = copy.deepcopy(originalseeds)
            both_assignments = copy.deepcopy(assignments)
            for word, assgn in remainder_assignments.iteritems():
                both_assignments[word] = assgn
                if word not in assignments or assignments[word] == "UNK":
                    both_assignments[word] = assgn
            print "Simple Classification of Remainder"
            evaluate_tokens(args.inputdir, args.corpus, both_assignments, evalmap, distmap, args.numtokens)        

        print "\nNo Classification of Remainder"
        evaluate_tokens(args.inputdir, args.corpus, assignments, evalmap, distmap, args.numtokens)

        print "\nNGram Tagger"
#        evaluate_tokens(args.inputdir, args.corpus, assignments, evalmap, distmap, outfile=args.datafile, seeds=originalseeds, ngram=True)

        print "\n\nBASELINES...\n"
#        POSes, originalseeds = get_seedwordsunambig(tags, args.numseedsperpos)
        POSes, originalseeds = get_seedwords(tags, args.numseedsperpos)
        if args.numseedsperpos == 0:
            originalseeds = {seed:tags for seed, tags in carlsonseeds.iteritems() if seed in assignments}
        baselinetags = {word:{"UNK":1} for word in assignments}
        for word, seedtags in originalseeds.iteritems():
            if word in assignments:
                baselinetags[word] = seedtags

        assignments, confidences, seeds = assign(tags, words, int(k), distances, originalseeds, evalmap, {}, {})
        evaluate_tokens(args.inputdir, args.corpus, assignments, evalmap, distmap, args.numtokens, originalseeds)
#        assign(args.datafile, int(k), args.numcontexttypes, args.numseedsperpos, None, evalmap, distmap)
        print "\n\nSEEDS ONLY"
        evaluate_types(baselinetags, originalseeds, tags, evalmap)
        evaluate_tokens(args.inputdir, args.corpus, baselinetags, evalmap, distmap, args.numtokens)


