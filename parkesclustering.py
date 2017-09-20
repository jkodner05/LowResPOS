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
import Levenshtein

import nltk

import pycrfsuite
#from pystruct.models import ChainCRF
#from pystruct.learners import FrankWolfeSSVM

START = "#START"
STOP = "#STOP"

CONF_THRESHOLD = 0.0

WSJ_FINDWORD = re.compile(r"\(([\w\d\.,\?!]+)\s([A-Za-z\d'-\.,\?!]+?)\)")

carlsonseeds = {"arm":{"NN":1},"ate":{"VB":1},"baby":{"NN":1},"balloon":{"NN":1},"bed":{"NN":1},"big":{"JJ":1},"black":{"JJ":1},"break":{"VB":1},"climb":{"VB":1},"come":{"VB":1},"cookie":{"NN":1},"crying":{"VB":1},"cut":{"VB":1},"dad":{"NN":1},"dirty":{"JJ":1},"do":{"VB":1},"down":{"RB":1},"eat":{"VB":1},"face":{"NN":1},"fall":{"VB":1},"goes":{"VB":1},"happy":{"JJ":1},"have":{"VB":1},"hide":{"VB":1},"him":{"PRP":1},"hold":{"VB":1},"hug":{"NN":1},"hurt":{"VB":1},"in":{"IN":1},"juice":{"NN":1},"make":{"VB":1},"milk":{"NN":1},"mom":{"NN":1},"move":{"VB":1},"not":{"RB":1},"off":{"RB":1},"on":{"RB":1},"one":{"DT":1},"put":{"VB":1},"run":{"VB":1},"sad":{"JJ":1},"say":{"VB":1},"sleep":{"VB":1},"small":{"JJ":1},"spoon":{"NN":1},"this":{"DT":1},"three":{"DT":1},"through":{"RB":1},"throw":{"VB":1},"to":{"IN":1},"today":{"RB":1},"tomorrow":{"RB":1},"train":{"NN":1},"tree":{"NN":1},"turn":{"NN":1},"two":{"DT":1},"up":{"RB":1},"very":{"RB":1},"walk":{"VB":1},"watch":{"VB":1},"water":{"NN":1},"by":{"IN":1},"cake":{"NN":1},"call":{"VB":1},"car":{"NN":1},"carry":{"VB":1},"chair":{"NN":1},"clean":{"JJ":1},"flower":{"NN":1},"fly":{"VB":1},"from":{"IN":1},"girl":{"NN":1},"give":{"VB":1},"kick":{"VB":1},"last":{"JJ":1},"later":{"RB":1},"look":{"VB":1},"out":{"RB":1},"play":{"VB":1},"stand":{"VB":1},"there":{"RB":1},"we":{"PRP":1},"wet":{"JJ":1},"who":{"PRP":1},"work":{"NN":1},"you":{"PRP":1}}


def tag_word(wordtup, keeptag):
    if keeptag:
        return wordtup[0] + "_" + wordtup[1]
    else:
        return wordtup[1]

def read_turkishtsfile(filename, tags, freqs, contexts):
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
    return tags, freqs, contexts


def read_ctbfile(filename, tags, freqs, contexts):
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

    return tags, freqs, contexts


def read_ctbfile_eval(filename):
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
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs


def read_turkishtsfile_eval(filename):
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
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs



def read_wsjfile(filename, tags, freqs, contexts):
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

    return tags, freqs, contexts


def read_wsjfile_eval(filename):
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
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs


def read_childesbrownfile_eval(filename):
    findword = re.compile(r"([A-Za-z:]+)\|([^\s]+)")
    pospairs = []
    with open(filename, "r") as f:
        for line in f:
            if line[0:5] != "%mor:":
                continue

            words = findword.findall(line.lower())
            pospairs.extend([(tag_word(wordtup,False).decode("utf-8"),wordtup[0]) for wordtup in words])
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs


def read_childesbrownfile(filename, tags, freqs, contexts):
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

    return tags, freqs, contexts


def read_conlluniversalfile_eval(filename):
    findword = re.compile(r"^\d+\t(.+?)\t.+?\t(\w+)")
    pospairs = []
    with open(filename, "r") as f:
        for line in f:
            words = findword.findall(line.lower())
            if not words:
                continue
            pospairs.extend([(tag_word((wordtup[1],wordtup[0]),False).decode("utf-8").lower(),wordtup[1]) for wordtup in words])
            #try:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),CHILDES_TAGMAP[wordtup[0]]) for wordtup in words])
            #except KeyError:
            #    pospairs.extend([(tag_word(wordtup,True).decode("utf-8"),"SKIP") for wordtup in words])
    return pospairs


def read_conlluniversalfile(filename, tags, freqs, contexts):
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

    return tags, freqs, contexts


def read_lctlfile(filename, tags, freqs, contexts):
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
                freqs[word] += 1
                if word not in tags:
                    tags[word] = defaultdict(int)
                tags[word][wordtup[0]] += 1
                contexts[prev].add((prevprev,word))
#                print prevprev, prev, word, "\t", prev, contexts[prev]
                prevprev = prev
                prev = word

    return tags, freqs, contexts



def read_corpusdirfiles(basedir, corpustype):
    tags = {}
    freqs = defaultdict(int)
    contexts = defaultdict(lambda : set([]))
    for subdir, dirs, files in os.walk(basedir):
        for f in files:
            if corpustype == "wsj":
                tags, freqs, contexts = read_wsjfile(os.path.join(subdir,f), tags, freqs, contexts)
            if corpustype == "ctb":
                tags, freqs, contexts = read_ctbfile(os.path.join(subdir,f), tags, freqs, contexts)
            if corpustype == "turkishts":
                tags, freqs, contexts = read_turkishtsfile(os.path.join(subdir,f), tags, freqs, contexts)
            elif corpustype == "brown":
                tags, freqs, contexts = read_childesbrownfile(os.path.join(subdir,f), tags, freqs, contexts)
            elif corpustype == "conll":
                tags, freqs, contexts = read_conlluniversalfile(os.path.join(subdir,f), tags, freqs, contexts)
            elif corpustype == "lctl":
                tags, freqs, contexts = read_lctlfile(os.path.join(subdir,f), tags, freqs, contexts)
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


def read_ngram(corpustype, filename, assignments, sentences, goldsentences, unigramfreqs, bigramfreqs, trigramfreqs, unigramseen, bigramseen, trigramseen, wordset):
    if corpustype == "conll":
        findword = re.compile(r"^\d+\t(.+?)\t.+?\t(\w+)")
    elif corpustype == "wsj":
        findword = WSJ_FINDWORD
    elif corpustype == "turkishts":
        findword = re.compile(r"([^\s]+)_([^\s]+)")

    types = set([])
    with open(filename, "r") as f:
        sentence = []
        goldsentence = []
        prevtag = START
        prevprevtag = START
        for line in f:
            words = findword.findall(line.lower())
            if corpustype == "turkishts":
                if len(types) >= 50000:
                    break

            if not words:
                if sentence:
                    sentences.append(copy.deepcopy(sentence))
                    sentence = []
                    goldsentences.append(copy.deepcopy(goldsentence))
                    goldsentence = []
                prevtag = START
                prevprevtag = START
                continue

            for wordtup in words:
                if corpustype == "wsj":
                    word = wordtup[1].lower()
                    goldtag = wordtup[0]
                elif corpustype == "conll":
                    word = wordtup[0].decode("utf-8")
                    goldtag = wordtup[1]
                elif corpustype == "turkishts":
                    word = tag_word((wordtup[1],wordtup[0]),False).decode("utf-8").lower()
                    goldtag = wordtup[1]
                    if u"//" in word:
                        if sentence:
                            sentences.append(copy.deepcopy(sentence))
                            sentence = []
                            goldsentences.append(copy.deepcopy(goldsentence))
                            goldsentence = []
                        prevtag = START
                        prevprevtag = START
                        continue
                    types.add(word)
                    if word == u"\u0130mparatorlu\u011fu'nu":
                        print "GOT IT"
                wordset.add(word)
                if word in assignments:
                    tag = max(assignments[word], key=assignments[word].get)
                    if corpustype == "wsj":
                        sentence.append((word.decode("utf-8"), tag))
                        goldsentence.append((word.decode("utf-8"), goldtag))
                    elif corpustype == "conll" or corpustype == "turkishts":
                        sentence.append((word, tag))
                        goldsentence.append((word, goldtag))
                    if tag not in unigramfreqs:
                        unigramfreqs[tag] = 0
                    if word not in unigramseen:
                        unigramfreqs[tag] += 1
                        unigramseen.add(word)
                    if prevtag:
                        if prevtag not in bigramfreqs:
                            bigramfreqs[prevtag] = {}
                        if tag not in bigramfreqs[prevtag]:
                            bigramfreqs[prevtag][tag] = 0
                        if prevtag not in bigramseen:
                            bigramseen[prevtag] = set([])
                        if word not in bigramseen[prevtag]:
                            bigramfreqs[prevtag][tag] += 1
                            bigramseen[prevtag].add(word)

                        if prevprevtag:
                            if prevprevtag not in trigramfreqs:
                                trigramfreqs[prevprevtag] = {}
                            if prevtag not in trigramfreqs[prevprevtag]:
                                trigramfreqs[prevprevtag][prevtag] = {}
                            if tag not in trigramfreqs[prevprevtag][prevtag]:
                                trigramfreqs[prevprevtag][prevtag][tag] = 0
                            if prevprevtag not in trigramseen:
                                trigramseen[prevprevtag] = {}
                            if prevtag not in trigramseen[prevprevtag]:
                                trigramseen[prevprevtag][prevtag] = set([])
                            if word not in trigramseen[prevprevtag][prevtag]:
                                trigramfreqs[prevprevtag][prevtag][tag] += 1
                            trigramseen[prevprevtag][prevtag].add(word)
                    prevprevtag = prevtag
                    prevtag = tag
                else:
                    if corpustype == "wsj":
                        sentence.append((word.decode("utf-8"), None))
                        goldsentence.append((word.decode("utf-8"), goldtag))
                    elif corpustype == "conll" or corpustype == "turkishts":
                        sentence.append((word, None))
                        goldsentence.append((word, goldtag))
                    
        if len(sentence) > 0:
            sentences.append(copy.deepcopy(sentence))
        if len(sentence) > 0:
            goldsentences.append(copy.deepcopy(goldsentence))

    return sentences, goldsentences, unigramfreqs, bigramfreqs, trigramfreqs, unigramseen, bigramseen, trigramseen, wordset


def get_nearnessprobs(corpustype, outfile):
    distances = {}
    seeds = {}
    nearnessprobs = {}
    with open(outfile+".remainder.pickle", 'rb') as instream:
        distances = pickle.load(instream)
        seeds = pickle.load(instream)

    for word, dists in distances.iteritems():
        nearnessprobs[word] = {}
        for seed, tags in seeds.iteritems():
            maxtag = max(tags, key=tags.get)
            if maxtag not in nearnessprobs[word]:
                nearnessprobs[word][maxtag] = 1000000
            if dists[seed] < nearnessprobs[word][maxtag]:
                nearnessprobs[word][maxtag] = dists[seed]
        denom = sum(nearnessprobs[word].values())
#        print word, nearnessprobs[word]
        for tag, freq in nearnessprobs[word].iteritems():
            nearnessprobs[word][tag] = 1-(freq/denom)
    return nearnessprobs


def get_orthoprobs(knownwordtags):
    smoothedtagdict = {}
    for word, tags in knownwordtags.iteritems():
        for tag in tags:
            smoothedtagdict[tag] = 0.000001

    unisufffreqs = defaultdict(lambda : copy.deepcopy(smoothedtagdict))
    bisufffreqs = defaultdict(lambda : copy.deepcopy(smoothedtagdict))
    trisufffreqs = defaultdict(lambda : copy.deepcopy(smoothedtagdict))
    hashyphenfreqs = {True:copy.deepcopy(smoothedtagdict), False:copy.deepcopy(smoothedtagdict)}
    hasperiodfreqs = {True:copy.deepcopy(smoothedtagdict), False:copy.deepcopy(smoothedtagdict)}

    

    for word, tags in knownwordtags.iteritems():
        maxtag = max(tags, key=tags.get)
        unisuff = word[-1]
        bisuff = word[-2:]
        trisuff = word[-3:]
        hashyphen = u"-" in word
        hasperiod = u"." in word
        unisufffreqs[unisuff][maxtag] += 1
        bisufffreqs[bisuff][maxtag] += 1
        trisufffreqs[trisuff][maxtag] += 1
        hashyphenfreqs[hashyphen][maxtag] += 1
        hasperiodfreqs[hasperiod][maxtag] += 1

    unisuffprobs = {}
    bisuffprobs = {}
    trisuffprobs = {}
    hashyphenprobs = {}
    hasperiodprobs = {}
    for suff, tags in unisufffreqs.iteritems():
        denom = float(sum(tags.values()))
        unisuffprobs[suff] = {tag:freq/denom for tag, freq in unisufffreqs[suff].iteritems()}
    for suff, tags in bisufffreqs.iteritems():
        denom = float(sum(tags.values()))
        bisuffprobs[suff] = {tag:freq/denom for tag, freq in bisufffreqs[suff].iteritems()}
    for suff, tags in trisufffreqs.iteritems():
        denom = float(sum(tags.values()))
        trisuffprobs[suff] = {tag:freq/denom for tag, freq in trisufffreqs[suff].iteritems()}
    for val, tags in hashyphenfreqs.iteritems():
        denom = float(sum(tags.values()))
        hashyphenprobs[val] = {tag:freq/denom for tag, freq in hashyphenfreqs[val].iteritems()}
    for val, tags in hasperiodfreqs.iteritems():
        denom = float(sum(tags.values()))
        hasperiodprobs[val] = {tag:freq/denom for tag, freq in hasperiodfreqs[val].iteritems()}

    return unisuffprobs, bisuffprobs, trisuffprobs, hashyphenprobs, hasperiodprobs


def get_besttag(word, contextprobs, nearnessprobs, unisuffprobs, bisuffprobs, trisuffprobs, hashyphenprobs, hasperiodprobs):
    bestprob = -1000000000
    besttag = None

    unisuff = word[-1]
    bisuff = word[-2:]
    trisuff = word[-3:]
    tagprobs_byhashyphen = hashyphenprobs[u"-" in word]
    tagprobs_byhasperiod = hashyphenprobs[u"." in word]
    tagprobs_byunisuff = unisuffprobs[unisuff] if unisuff in unisuffprobs else None
    tagprobs_bybisuff = bisuffprobs[bisuff] if bisuff in bisuffprobs else None
    tagprobs_bytrisuff = trisuffprobs[trisuff] if trisuff in trisuffprobs else None

    for tag in contextprobs:
        prob = 0
        prob += log(contextprobs[tag])
        prob += log(nearnessprobs[tag])
        prob += log(tagprobs_byhasperiod[tag])
        prob += log(tagprobs_byhashyphen[tag])
        if tagprobs_byunisuff:
            prob += log(tagprobs_byunisuff[tag])
        if tagprobs_bybisuff:
            prob += log(tagprobs_bybisuff[tag])
        if tagprobs_bytrisuff:
            prob += log(tagprobs_bytrisuff[tag])
        if prob > bestprob:
            bestprob = prob
            besttag = tag

    return besttag


def ngram_tag(basedir, corpustype, assignments, seeds, outfile):
    sentences = []
    goldsentences = []
    unigramfreqs = {}
    bigramfreqs = {}
    trigramfreqs = {}
    unigramseen = set([])
    bigramseen = {}
    trigramseen = {}

    print "Getting Ersatz Distances as Probs"
    nearnessprobs = get_nearnessprobs(corpustype, outfile)
    print "Getting Orthographic Probs"
    unisuffprobs, bisuffprobs, trisuffprobs, hashyphenprobs, hasperiodprobs = get_orthoprobs(assignments)

    print "Getting NGram Freqs"
    for subdir, dirs, files in os.walk(basedir):
        for f in files:
            read_ngram(corpustype, os.path.join(subdir,f), assignments, sentences, goldsentences, unigramfreqs, bigramfreqs, trigramfreqs, unigramseen, bigramseen, trigramseen, {})

    goldpospairs = []
    for sentence in goldsentences:
        for pair in sentence:
            goldpospairs.append(pair)

    if True:
        unigramprobs = {}
        bigramprobs = {}
        trigramprobs = {}
        denom = float(sum(unigramfreqs.values()))
        print unigramfreqs
        unigramprobs = {tag:freq/denom for tag, freq in unigramfreqs.iteritems()}
    #    print sum(unigramprobs.values())
    #    probsum = 0.0
        for prevtag, tags in bigramfreqs.iteritems():
            denom = float(sum(tags.values()))
            bigramprobs[prevtag] = {tag:freq/denom for tag, freq in tags.iteritems()}
    #        probsum += sum(bigramprobs[prevtag].values())
    #    print probsum
    #    probsum = 0
        for prevprevtag, prevtags in trigramfreqs.iteritems():
            denom = float(sum([sum(tags.values()) for prevtag, tags in prevtags.iteritems()]))
            prevtagprobs = {}
            for prevtag, tags in prevtags.iteritems():
                trigramprobs[(prevprevtag,prevtag)] = {tag:freq/denom for tag, freq in tags.iteritems()}
    #            probsum += sum(trigramprobs[(prevprevtag,prevtag)].values())
    #    print probsum





        assgnpospairs = []
        index = 0
        for sentence in sentences:
            prevprevtag = START
            prevtag = START
            for pair in sentence:
                word = pair[0]
                tag = pair[1]
                try:
                    if tag:
                        assgnpospairs.append(pair)
                    elif prevprevtag and prevtag:
                        context = (prevprevtag, prevtag)
                        newtag = None
                        if context in trigramprobs:
                            newtag = get_besttag(word, trigramprobs[context], nearnessprobs[word], unisuffprobs, bisuffprobs, trisuffprobs, hashyphenprobs, hasperiodprobs)
                        assgnpospairs.append((word, newtag))
                    elif prevtag:
                        newtag = None
                        if prevtag in bigramprobs:
                            newtag = get_besttag(word, bigramprobs[prevtag], nearnessprobs[word], unisuffprobs, bisuffprobs, trisuffprobs, hashyphenprobs, hasperiodprobs)
                        #print prevprevtag, prevtag, newtag, word, "\t", goldpospairs[index][0], goldpospairs[index][1]
                        assgnpospairs.append((word, newtag))
                    else:
                        newtag = get_besttag(word, unigramprobs, nearnessprobs[word], unisuffprobs, bisuffprobs, trisuffprobs, hashyphenprobs, hasperiodprobs)
                        assgnpospairs.append((word, newtag))
                except KeyError:
                    print word, word in nearnessprobs
                prevprevtag = prevtag
                prevtag = tag
                index += 1


    return assgnpospairs, goldpospairs



def load_ersatz(corpustype, outfile):
    with open(outfile+".remainder.pickle", 'rb') as instream:
        distances = pickle.load(instream)
        seeds = pickle.load(instream)
    return seeds, distances


def get_insensitivefeats(words, assignments, seeds, distances):
    insensitivefeats = {}
    hasdigit = re.compile(r"\d")

    for word in words:
        wordfeats = {}

        wordfeats["wordform"] = word

        wordfeats["unisuff"] = word[-1]
        wordfeats["bisuff"] = word[-2:]
        wordfeats["trisuff"] = word[-3:]

        wordfeats["hashyphen"] = u"-" in word
        wordfeats["hasperiod"] = u"." in word
        wordfeats["hasdigit"] = hasdigit.search(word) != None

        if word in seeds:
            wordfeats["tag"] = max(seeds[word], key=seeds[word].get)
        elif word in assignments:
            wordfeats["tag"] = max(assignments[word], key=assignments[word].get)
        else:
            wordfeats["tag"] = None

        distancefeats = copy.deepcopy(distances[word])
        wordfeats["distances"] = distancefeats
        
        mindist = 100000.0
        mindist2 = 100000.0
        mindist3 = 100000.0
        mindistseed = None
        mindistseed2 = None
        mindistseed3 = None
        for seed, distance in distancefeats.iteritems():
            if distance < mindist:
                mindist3 = mindist2
                mindistseed3 = mindistseed2
                mindist2 = mindist
                mindistseed2 = mindistseed
                mindist = distance
                mindistseed = seed
            elif distance >= mindist and distance < mindist2:
                mindist3 = mindist2
                mindistseed3 = mindistseed2
                mindist2 = distance
                mindistseed2 = seed
            elif distance >= mindist2 and distance < mindist3:
                mindist3 = distance
                mindistseed3 = seed
        wordfeats["proto1"] = mindistseed
        wordfeats["proto2"] = mindistseed2
        wordfeats["proto3"] = mindistseed3
        wordfeats["protoset"] = set([mindistseed, mindistseed2,mindistseed3])
        insensitivefeats[word] = wordfeats

    return insensitivefeats

def get_sentencefeats(sentences, goldsentences, wordfeats):

    sentencesfeats = []
    sentenceslabels = []
    print "total sentences", len(sentences)
    for i, sentence in enumerate(sentences):
        sentencefeats = []
        sentencelabels = []
        if not i % 200:
            print i
        if i == 10000:
            break
        for j, pair in enumerate(sentence):
            word = pair[0]
            tag = pair[1]
            sentencefeats.append(copy.deepcopy(wordfeats[word]))
            sentencelabels.append(goldsentences[i][j][1])
        if sentencefeats:
            sentencesfeats.append(sentencefeats)
            sentenceslabels.append(sentencelabels)
    
    for sentencefeats in sentencesfeats:
        for i, feats in enumerate(sentencefeats):
            if not feats["tag"]:
                continue
            feats["BOS"] = i == 0
            feats["EOS"] = i == len(sentencefeats)-1
            feats["BOS2"] = i == 1
            feats["EOS2"] = i == len(sentencefeats)-2
            if i < len(sentencefeats)-1 :
                feats["next"] = {}
                nextfeats = sentencefeats[i+1]
                feats["next"]["wordform"] = nextfeats["wordform"]
                feats["next"]["unisuff"] = nextfeats["unisuff"]
                feats["next"]["bisuff"] = nextfeats["bisuff"]
                feats["next"]["trisuff"] = nextfeats["trisuff"]
                feats["next"]["hashyphen"] = nextfeats["hashyphen"]
                feats["next"]["hasdigit"] = nextfeats["hasdigit"]
                feats["next"]["hasperiod"] = nextfeats["hasperiod"]
                if nextfeats["tag"] != None:
                    feats["next"]["tag"] = nextfeats["tag"]
            if i > 0:
                feats["prev"] = {}
                prevfeats = sentencefeats[i-1]
                feats["prev"]["wordform"] = prevfeats["wordform"]
                feats["prev"]["unisuff"] = prevfeats["unisuff"]
                feats["prev"]["bisuff"] = prevfeats["bisuff"]
                feats["prev"]["trisuff"] = prevfeats["trisuff"]
                feats["prev"]["hashyphen"] = prevfeats["hashyphen"]
                feats["prev"]["hasdigit"] = prevfeats["hasdigit"]
                feats["prev"]["hasperiod"] = prevfeats["hasperiod"]
                if prevfeats["tag"] != None:
                    feats["prev"]["tag"] = prevfeats["tag"]

            if i < len(sentencefeats)-2 :
                feats["next2"] = {}
                next2feats = sentencefeats[i+2]
                feats["next2"]["wordform"] = next2feats["wordform"]
                feats["next2"]["unisuff"] = next2feats["unisuff"]
                feats["next2"]["bisuff"] = next2feats["bisuff"]
                feats["next2"]["trisuff"] = next2feats["trisuff"]
                feats["next2"]["hashyphen"] = next2feats["hashyphen"]
                feats["next2"]["hasdigit"] = next2feats["hasdigit"]
                feats["next2"]["hasperiod"] = next2feats["hasperiod"]
                if next2feats["tag"] != None:
                    feats["next2"]["tag"] = next2feats["tag"]
            if i > 1:
                feats["prev2"] = {}
                prev2feats = sentencefeats[i-2]
                feats["prev2"]["wordform"] = prev2feats["wordform"]
                feats["prev2"]["unisuff"] = prev2feats["unisuff"]
                feats["prev2"]["bisuff"] = prev2feats["bisuff"]
                feats["prev2"]["trisuff"] = prev2feats["trisuff"]
                feats["prev2"]["hashyphen"] = prev2feats["hashyphen"]
                feats["prev2"]["hasdigit"] = prev2feats["hasdigit"]
                feats["prev2"]["hasperiod"] = prev2feats["hasperiod"]
                if prev2feats["tag"] != None:
                    feats["prev2"]["tag"] = prev2feats["tag"]


    trainsentences = []
    trainlabels = []
    for i, sentence in enumerate(sentencesfeats):
        items = []
        labels = []
        for j, item in enumerate(sentence):
            if item["tag"] == None:
                if items:
                    trainsentences.append(pycrfsuite.ItemSequence(items))
                    trainlabels.append(labels)
                    items = []
                    labels = []
                continue
            items.append(item)
            labels.append(item["tag"])
        if items:
            trainsentences.append(pycrfsuite.ItemSequence(items))
            trainlabels.append(labels)


    testlabels = sentenceslabels
    testsentences = []
    for i, sentence in enumerate(sentencesfeats):
        items = []
        for j, item in enumerate(sentence):
            if item["tag"] == None:
                del item["tag"]
            items.append(item)
        testsentences.append(pycrfsuite.ItemSequence(items))


    return trainsentences, trainlabels, testsentences, testlabels

def crf_tag(basedir, corpustype, assignments, seeds, outfile):
    sentences = []
    goldsentences = []
    unigramfreqs = {}
    bigramfreqs = {}
    trigramfreqs = {}
    unigramseen = set([])
    bigramseen = {}
    trigramseen = {}
    wordset = set([])

    print "Reading Corpus for Tagging"
    for subdir, dirs, files in os.walk(basedir):
        for f in files:
            read_ngram(corpustype, os.path.join(subdir,f), assignments, sentences, goldsentences, {}, {}, {}, set([]), {}, {}, wordset)

    wordindices = {}
    index = 0
    for word in words:
        wordindices[word] = index
        index += 1

    print "Loading Ersatz Distances"
    seeds, distances = load_ersatz(corpustype, outfile)

#    for word in wordset:
#        if word not in distances:
#            print word, word in seeds, word in assignments

    insensitivefeats = get_insensitivefeats(wordset, assignments, seeds, distances)
    train_X, train_y, test_X, test_y = get_sentencefeats(sentences, goldsentences, insensitivefeats)

    trainer = pycrfsuite.Trainer(verbose=True)
    for X, y in zip(train_X, train_y):
        trainer.append(X, y)

    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 500,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    MODELFILE = "/mnt/nlpgridio2/nlp/users/lorelei/crftest500b.model"
    trainer.train(MODELFILE)

    tagger = pycrfsuite.Tagger()
    tagger.open(MODELFILE)
    
    numtotal = 0.0
    numcorrect = 0.0
    numcorrect_byclustering = 0.0
    numOOV = 0.0
    numOOVcorrect = 0.0
    for X, y in zip(test_X, test_y):
        tags = tagger.tag(X)
        for i, feats in enumerate(X.items()):
            isOOV = True
            for feat in feats:
                if "tag:" in feat:
                    isOOV = False
            for feat in feats:
                if "wordform:" in feat and "prev:" not in feat and "next:" not in feat and "prev2:" not in feat and "next2:" not in feat:
                    print feat.replace("wordform:",""), "\t", tags[i], y[i]
                    if tags[i] == y[i]:
                        if isOOV:
                            numOOVcorrect += 1
                        else:
                            word = feat.replace("wordform:","")
                            maxtag = max(assignments[word], key=assignments[word].get)
                            if maxtag == y[i]:
                                numcorrect_byclustering += 1
                        numcorrect += 1
                    numtotal += 1
                    if isOOV:
                        numOOV += 1
        print ""
    print "Total Correct\t", numcorrect/numtotal, numtotal
    print "OOV Correct\t", numOOVcorrect/numOOV, numOOV
    print "non-OOV Correct\t", (numcorrect-numOOVcorrect)/(numtotal-numOOV)
    print "Clustering Correct\t", (numcorrect_byclustering)/(numtotal-numOOV)
    exit()




def fillin_ersatzfeats(distances, seeds, assignments, wordindices, wordfeats):
    seedindices = {}
    index = 0
    for seed in seeds:
        seedindices[seed] = index
        index += 1
    
    maxdistance = 0
    for word, distances in distances.iteritems():
        for seed, distance in distances.iteritems():
            if distance > maxdistance:
                maxdistance = distance
            wordfeats[word][0,seedindices[seed]] = distance
            
    for word in assignments:
        if word not in distances:
            tag = max(assignments[word], key=assignments[word].get)
            for seed in seeds:
                seedtag = max(assignments[seed], key=assignments[seed].get)
                if tag == seedtag:
                    wordfeats[word][0,seedindices[seed]] = 0
                else:
                    wordfeats[word][0,seedindices[seed]] = maxdistance

    for seed in seeds:
        wordfeats[MISSING_WORD][0,seedindices[seed]] = maxdistance

    return distances


def fillin_binaryorthofeats(wordfeats, startindex):
    hasdigit = re.compile(r"\d")
    for word, feats in wordfeats.iteritems():
        feats[0,startindex:startindex+3] = np.asarray([u"-" in word, u"." in word, hasdigit.search(word) != None])
    wordfeats[MISSING_WORD][0,startindex:startindex+3] = np.zeros([1,3])

def fillin_affixfeats(wordfeats, startindex):
    unigramindices = {}
    bigramindices = {}
    trigramindices = {}
    uniindex = 0
    biindex = 0
    triindex = 0
    for word in wordfeats:
        unisuff = word[-1]
        bisuff = word[-2:]
        trisuff = word[-3:]
        if unisuff not in unigramindices:
            unigramindices[unisuff] = uniindex
            uniindex += 1
        if bisuff not in bigramindices:
            bigramindices[bisuff] = biindex
            biindex += 1
        if trisuff not in trigramindices:
            trigramindices[trisuff] = triindex
            triindex += 1

    for word, feats in wordfeats.iteritems():
        feats[0,startindex:startindex+3] = np.asarray([unigramindices[word[-1]], bigramindices[word[-2:]], trigramindices[word[-3:]]])
    wordfeats[MISSING_WORD][0,startindex:startindex+3] = np.asarray([-1,-1,-1])

def initialize_featarrays(words, numfeats):
    featarrays = {}
    for word in words:
        featarrays[word] = np.zeros([1,numfeats])
    featarrays[MISSING_WORD] = np.zeros([1,numfeats])
    return featarrays
    

def crf_tag_old(basedir, corpustype, assignments, seeds, outfile):
    sentences = []
    goldsentences = []
    unigramfreqs = {}
    bigramfreqs = {}
    trigramfreqs = {}
    unigramseen = set([])
    bigramseen = {}
    trigramseen = {}
    wordset = set([])

    print "Reading Corpus for Tagging"
    for subdir, dirs, files in os.walk(basedir):
        for f in files:
            read_ngram(corpustype, os.path.join(subdir,f), assignments, sentences, goldsentences, {}, {}, {}, set([]), {}, {}, wordset)

    wordindices = {}
    index = 0
    for word in words:
        wordindices[word] = index
        index += 1

    print "Loading Ersatz Distances"
    seeds, distances = load_ersatz(corpustype, outfile)
    tagset = set([])
    for seed in seeds:
        tag = max(seeds[seed], key=seeds[seed].get)
        tagset.add(tag)
    tagindices = {}
    tagindex = 0
    for tag in tagset:
        tagindices[tag] = tagindex
        tagindex += 1
    tagindices[None] = tagindex

    print "Getting Features"
    NUMAFFIXFEATS = 3
    NUMBINARYORTHOFEATS = 3
    wordfeats = initialize_featarrays(wordset, len(seeds) + NUMAFFIXFEATS + NUMBINARYORTHOFEATS)

    fillin_ersatzfeats(distances, seeds, assignments, wordindices, wordfeats)
    fillin_binaryorthofeats(wordfeats, len(seeds))
    fillin_affixfeats(wordfeats, len(seeds) + NUMAFFIXFEATS)

    #create "training" sentences
    X = []
    y_train = []
    y_test = []
    for sentence in sentences:
        featarray = np.zeros([len(sentence), len(seeds) + NUMAFFIXFEATS + NUMBINARYORTHOFEATS])
        tags = np.asarray([tagindices[pair[1]] for pair in sentence])
        for i, pair in enumerate(sentence):
            word = pair[0]
            tag = pair[1]
            featarray[i,:] = wordfeats[word]
        X.append(featarray)
        y_train.append(tags)
    for sentence in goldsentences:
        tags = np.asarray([tagindices[pair[1]] for pair in sentence])
        y_test.append(tags)

    #perform labelling
    #set class weights to equal except for None
    tagweight = 1.0/len(tagset)
    tagweights = [tagweight]*len(tagset)
    tagweights.append(-1000.0)

    print len(tagweights), len(tagindices), len(tagset)

    model=ChainCRF(n_states=len(tagweights), class_weight=tagweights)
    ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
    ssvm.fit(X[0:10000], y_train[0:10000]) 
    predictions = ssvm.predict(X)

    numtotal = 0.0
    numcorrect = 0.0
    for i, sentence in enumerate(predictions):
        for j, tag in enumerate(sentence):
            if tag and tag == y_test[i][j]:
                numcorrect += 1
            numtotal += 1
    print "CRF Predictions Only"
    print "\t", numtotal
    print "\t", numcorrect
    print "\t", numcorrect/numtotal

    for i, sentence in enumerate(sentences):
        for j, pair in enumerate(sentence):
            if pair[1]:
                predictions[i][j] = tagindices[pair[1]]
    numtotal = 0.0
    numcorrect = 0.0
    for i, sentence in enumerate(predictions):
        for j, tag in enumerate(sentence):
            if tag == y_test[i][j]:
                numcorrect += 1
            numtotal += 1
    print "All Predictions"
    print "\t", numtotal
    print "\t", numcorrect
    print "\t", numcorrect/numtotal
#    for i in range(0,5):
#        print predictions[i]
#        print y_train[i]
#        print y_test[i]
#        print sentences[i]
#        print goldsentences[i]
#        print ""
    print ssvm.score(X, y_test)
    exit()


def evaluate_tokens(basedir, corpustype, assignments, evaltagmap, disttagmap, seeds=None, ngram=False, outfile=None):
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
        for subdir, dirs, files in os.walk(basedir):
            for f in files:
                if corpustype == "brown":
                    allpospairs = read_childesbrownfile_eval(os.path.join(subdir,f))
                elif corpustype == "wsj":
                    allpospairs = read_wsjfile_eval(os.path.join(subdir,f))
                elif corpustype == "ctb":
                    allpospairs = read_ctbfile_eval(os.path.join(subdir,f))
                elif corpustype == "turkishts":
                    allpospairs = read_turkishtsfile_eval(os.path.join(subdir,f))
                elif corpustype == "conll":
                    allpospairs = read_conlluniversalfile_eval(os.path.join(subdir,f))
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


def evaluate_types(assignments, seedwords, correcttags, evaltagmap):
    numassgn = 0
    numseeds = 0
    numtotal = 0
    numcorrect_assgn = 0
    numcorrect_seeds = 0
    numcorrect_total = 0

    for word, assgn in assignments.iteritems():
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


def calc_KL(word1, word2, contextvecs):
    print(type(np.transpose(contextvecs[word1].tocsr()).toarray()), np.transpose(contextvecs[word1].tocsr()).toarray())
    qp = sum(stats.entropy(qk=np.transpose(contextvecs[word1].tocsr()).toarray(), pk=np.transpose(contextvecs[word2].tocsr()).toarray()))
    pq = sum(stats.entropy(pk=np.transpose(contextvecs[word1].tocsr()).toarray(), qk=np.transpose(contextvecs[word2].tocsr()).toarray()))
#    qp = sum(stats.entropy(qk=contextvecs[word1], pk=contextvecs[word2]))
#    pq = sum(stats.entropy(pk=contextvecs[word1], qk=contextvecs[word2]))

    return qp+pq


def save_KLmat(words_topk, fname, contextvecs):
    KLmat = sparse.lil_matrix((len(words_topk),len(words_topk)))
    for i, word1 in enumerate(words_topk):
        for j, word2 in enumerate(words_topk):
            if j > i:
                KL = calc_KL(word1,word2,contextvecs)
                KLmat[i,j] = KL
#                print(word1, word2, "\t", calc_KL(word1,word2,leftcontextvecs), "\t", calc_KL(word1,word2,rightcontextvecs))
    with open(fname, 'wb') as outstream:
        pickle.dump(KLmat, outstream)

def calc_editratio_ndarray(vec1, vec2, word1, word2):
    ratio = Levenshtein.distance(word1, word2) + 2*Levenshtein.distance(word1[-3:], word2[-3:]) + 3*Levenshtein.distance(word1[-2:], word2[-2:]) + 4*Levenshtein.distance(word1[-1:], word2[-1:])
    return ratio

def calc_cossim_ndarray(vec1, vec2, word1, word2):
    return spatial.distance.cosine(vec1, vec2)


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


def get_obsmat(words, contextvecs):
#    obsmat = sparse.lil_matrix((len(words_topk),len(contextvecs[words[0]].toarray()[0])))
    obsmat = matlib.zeros(((len(words),contextvecs[words[0]].size)))
    for i, word in enumerate(words):
        obsmat[i,:] = contextvecs[word]
    print "Created Observation Matrix"
    return obsmat

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


def assign_remainder(inputdir, corpus, words, k, m, seeds, outfile, oldassignments):
    tags, freqs, contexts = read_corpusdirfiles(inputdir, corpus)
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
                distance = calc_KL_ndarray2(contextvecs[word], contextvecs[seed])
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

def create_matrices(inputdir, corpus, datafile, k, m, ktagsfname, mtagsfname):
    print "Reading corpus files from", inputdir
    tags, freqs, contexts = read_corpusdirfiles(inputdir, corpus)
    print "Total types in corpus\t", len(freqs)
    print "Total tokens in corpus\t", sum(freqs.values())
    if ktagsfname:
        tagset = load_tagsfile(ktagsfname)
        words_topk = get_tagset_topk(freqs, k, tagset)
    else:
        words_topk = get_topk(freqs, k)

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

    bothcontextvecs= get_contextvecs_both(words_topk, contexts, words_topm=words_topm)

    print "Calculating vector distances..."
    start = time.time()
    bothdist = calc_distances_multi(bothcontextvecs, words_topk, calc_KL_ndarray2, 16)
#    bothdist2 = calc_distances_multi(bothcontextvecs, words_topk, calc_editratio_ndarray, 16)
#    bothdist = combinedistances(bothdist1, bothdist2, words_topk)
#    bothdist = calc_distances_multi(bothcontextvecs, words_topk, calc_cossim_ndarray, 16)
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
        POSes, seedwords = get_seedwordsunambig(mappedtags, seedsperPOS)
#        POSes, seedwords = get_seedwords(mappedtags, seedsperPOS)
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



def cv_assign(mappedtags, words, k, distances, seedwords, evalmap, prevassignments, prevconfidences, numcv):
    def holdout(words, distances, k, i, numcv):
        cvwords = [word for j, word in enumerate(words) if (j+i) % numcv != 0]
        delrange = [j for j in range(0,len(words)) if (j+i) % numcv == 0]
        cvdist = distance.squareform(np.delete(np.delete(distance.squareform(distances), delrange, 1), delrange, 0))
        return cvwords, cvdist

    print "RUNNING CROSSVALIDATION, k=", k

    assignmentslist = []
    confidenceslist = []
#    cvwords, cvdistances = holdout(words, distances, k, 10, numcv)
#    assignments, confidences, newseeds = assign(mappedtags, cvwords, 88, cvdistances, seedwords, evalmap, prevassignments, prevconfidences)

    for i in range(numcv):
        cvwords, cvdistances = holdout(words, distances, k, i, numcv)
        print len(words), len(cvwords)
        print distance.squareform(distances)[:,0].size, distance.squareform(cvdistances)[:,0].size
        assignments, confidences, newseeds = assign(mappedtags, cvwords, (numcv-1)*k/numcv, cvdistances, copy.deepcopy(seedwords), evalmap, prevassignments, prevconfidences)
#        evaluate_types(assignments, seedwords, mappedtags, evalmap)
        assignmentslist.append(assignments)
        confidenceslist.append(confidences)
        raw_input("...")

    joinedassignments = {}
    joinedconfidences = {}
    for a in assignmentslist:
        for word, tags in a.iteritems():
            if word not in joinedassignments:
                joinedassignments[word] = {}
            for tag, count in tags.iteritems():
                if tag not in joinedassignments[word]:
                    joinedassignments[word][tag] = 0
                joinedassignments[word][tag] += count
    for c in confidenceslist:
        for word, conf in c.iteritems():
            if word not in joinedconfidences:
                joinedconfidences[word] = 0
            joinedconfidences[word] += conf/(numcv-1)

    newseedwords = update_seedwords(seedwords, joinedassignments, joinedconfidences)
    evaluate_types(joinedassignments, seedwords, mappedtags, evalmap)
    return joinedassignments, joinedconfidences, newseedwords

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Parkes-style Single-Word Context Clustering")

    parser.add_argument("inputdir", nargs="?", help="corpus directory; must be provided if not --loadmats")
    parser.add_argument("--corpus", nargs="?", help="corpus type: [conll, ctb, lctl, wsj, turkish]")
    parser.add_argument("datafile", help="file containing/to contain stored vocabulary, contexts, and distances")
    parser.add_argument("--loadmats", help="load stored vocabulary, contexts, distances", action="store_true")
    parser.add_argument("-k", "--numclustertypes", help="cluster top k words", type=str)
    parser.add_argument("-s", "--numseedsperpos", help="number of seed words per POS", type=int)
    parser.add_argument("-m", "--numcontexttypes", nargs="?", help="filter out all but top m context words", type=int)    
    parser.add_argument("--ktagsfile", nargs="?", help="only keep cluster words with these tags", type=str)
    parser.add_argument("--mtagsfile", nargs="?", help="only keep context words with these tags", type=str)
    parser.add_argument("--both", help="compute a single tree on both contexts rather than one for each", action="store_true")
    parser.add_argument("--evalmap", nargs="?", help="map training tags to eval tags", type=str)
    parser.add_argument("--distmap", nargs="?", help="map input tags to training tags", type=str)
    parser.add_argument("-c", "--confidence", help="seed confidence threshold", type=float)
    parser.add_argument("--guessremainder", help="apply simple classification for tokens outside the top k", action="store_true")

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
        create_matrices(args.inputdir, args.corpus, args.datafile, int(args.numclustertypes), args.numcontexttypes, args.ktagsfile, args.mtagsfile)
    else:
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
        for k in args.numclustertypes.split(","):
#        for k in np.arange(50,int(args.numclustertypes.split(",")[-1])+incr,incr):
            assignments, confidences, seeds = assign(tags, words, int(k), distances, seedwords, evalmap, assignments, confidences)
        for word in seedwords:
            maxtag = max(seedwords[word], key=seedwords[word].get)
            assignments[word] = {maxtag:seedwords[word][maxtag]}

        if args.guessremainder:
            print "Reading corpus files from", args.inputdir
            remainder_assignments = assign_remainder(args.inputdir, args.corpus, words, int(k), 10000, originalseeds, args.datafile, assignments)
            both_assignments = copy.deepcopy(originalseeds)
            both_assignments = copy.deepcopy(assignments)
            for word, assgn in remainder_assignments.iteritems():
                both_assignments[word] = assgn
                if word not in assignments or assignments[word] == "UNK":
                    both_assignments[word] = assgn
            print "Simple Classification of Remainder"
            evaluate_tokens(args.inputdir, args.corpus, both_assignments, evalmap, distmap)        

        print "\nNo Classification of Remainder"
        evaluate_tokens(args.inputdir, args.corpus, assignments, evalmap, distmap)

        print "\nNGram Tagger"
#        evaluate_tokens(args.inputdir, args.corpus, assignments, evalmap, distmap, outfile=args.datafile, seeds=originalseeds, ngram=True)

        print "\n\nBASELINES...\n"
        POSes, originalseeds = get_seedwordsunambig(tags, args.numseedsperpos)
#        POSes, originalseeds = get_seedwords(tags, args.numseedsperpos)
        if args.numseedsperpos == 0:
            originalseeds = {seed:tags for seed, tags in carlsonseeds.iteritems() if seed in assignments}
        baselinetags = {word:{"UNK":1} for word in assignments}
        for word, seedtags in originalseeds.iteritems():
            if word in assignments:
                baselinetags[word] = seedtags

        assignments, confidences, seeds = assign(tags, words, int(k), distances, originalseeds, evalmap, {}, {})
        evaluate_tokens(args.inputdir, args.corpus, assignments, evalmap, distmap, originalseeds)
#        assign(args.datafile, int(k), args.numcontexttypes, args.numseedsperpos, None, evalmap, distmap)
        print "\n\nSEEDS ONLY"
        evaluate_types(baselinetags, originalseeds, tags, evalmap)
        evaluate_tokens(args.inputdir, args.corpus, baselinetags, evalmap, distmap)


