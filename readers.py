import re

WSJ_FINDWORD = re.compile(r"\(([\w\d\.,\?!]+)\s([A-Za-z\d'-\.,\?!]+?)\)")

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

