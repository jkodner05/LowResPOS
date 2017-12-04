import argparse
import sys, os, os.path, re
from collections import defaultdict

AVAILABLE_CORPORA = set(["utb"])

def read_utb(filename, freqs):
    findword = re.compile(r"^\d+\t(.+?)\t.+?\t\w+")
    with open(filename, "r") as f:
        for line in f:
            words = findword.findall(line.lower())
            for word in words:
                freqs[word] += 1
    return freqs


def read_corpusdir(basedir, corpustype):
    freqs = defaultdict(int)
    for subdir, dirs, files in os.walk(basedir):
        for f in files:
            if corpustype == "utb":
                freqs = read_utb(os.path.join(subdir,f), freqs)
    return dict(freqs)


def write_wordlist(freqs, outfname):
    sortedfreqs = sorted(freqs.iteritems(), key=lambda (k,v): (v, k), reverse=True)
    with open(outfname, "w") as f:
        for word, freq in sortedfreqs:
            f.write("%s\t%s\n" % (freq,word))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Parkes-style Single-Word Context Clustering")

    parser.add_argument("inputdir", help="corpus directory")
    parser.add_argument("outputfile", help="output file")
    parser.add_argument("-c", "--corpustype", help="corpus type: [utb]")
    args = parser.parse_args()

    if args.corpustype.lower() not in AVAILABLE_CORPORA:
        exit("Corpus type not valid")

    print "INPUT DIR", args.inputdir
    print "OUTPUT FILE", args.outputfile

    freqs = read_corpusdir(args.inputdir, args.corpustype.lower())

    write_wordlist(freqs, args.outputfile)
