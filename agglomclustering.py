import numpy as np

class AgglomTree():

    def __init__(self, id, POS="", sub1=None, sub2=None):
        self.topid = id
        self.ids = [id]
        self.POS = POS
        self.size = 1
        if sub1 and sub2:
            self.ids = sub1.ids
            self.ids.extend(sub2.ids)
            self.size = len(self.ids)



class Agglom():
    
    def __init__(self, distances, POSes):
        self.distances = distances
        self.maxbaseid = distances.shape[0]-1
        self.currid = self.maxbaseid+1
        self.POSes = POSes
        self.subtrees = self.init_subtrees()
        self.memoized_distances = {}

    def init_subtrees(self):
        subtrees = {}
        for i in range(self.currid):
            subtrees[i] = AgglomTree(i, self.POSes[i])
        return subtrees

    def add_subtree(self, id1, id2):
        sub1 = self.subtrees[id1]
        sub2 = self.subtrees[id2]
        del self.subtrees[id1]
        del self.subtrees[id2]
        POS = ""
        if sub1.POS == sub2.POS:
            POS = sub1.POS
        else:
            POS = sub1.POS + sub2.POS
        newsubtree = AgglomTree(self.currid, POS, sub1, sub2)
        self.subtrees[self.currid] = newsubtree
        self.currid += 1
        return newsubtree

    def compare_subtrees(self, id1, id2):
        if (id1,id2) in self.memoized_distances:
            return self.memoized_distances[(id1,id2)]
        if (id2,id1) in self.memoized_distances:
            return self.memoized_distances[(id2,id1)]

        sub1 = self.subtrees[id1]
        sub2 = self.subtrees[id2]
        indices1 = np.repeat(sub1.ids,len(sub2.ids))
        indices2 = np.tile(sub2.ids,len(sub1.ids))
        newdistance = np.sum(self.distances[indices1, indices2]) / (sub1.size*sub2.size)
        self.memoized_distances[(id1, id2)] = newdistance
        return newdistance

    def find_next_join(self, skipmixedlabels=False):
        mindist = 1e100
        minid1 = -1
        minid2 = -1
        found = False
        for i in self.subtrees.keys():
            for j in self.subtrees.keys():
                if i == j:
                    continue
                if skipmixedlabels:
                    subiPOS = self.subtrees[i].POS
                    subjPOS = self.subtrees[j].POS
                    if subiPOS != "" and subjPOS != "" and subiPOS != subjPOS:
                        continue
                newdist = self.compare_subtrees(i,j)
                if newdist < mindist:
                    found = True
                    mindist = newdist
                    minid1 = i
                    minid2 = j
        return mindist, minid1, minid2, found
         
    def has_more(self):
        return len(self.subtrees) > 1
       

def make_seedmap(words_topk, seedtags):
    seedmap = {}
    for i, seed in enumerate(words_topk):
        tag = ""
        if seed in seedtags:
            tag = seedtags[seed]
        seedmap[i] = tag
    return seedmap


def make_tree_implemented(distances, seedlabels, avoidmixedlabels, method="average"):
    if method != "average":
        exit("Agglom only implemented with 'average' linkage")

    treemat = np.zeros((distances.shape[0]-1,4))

    agglom = Agglom(distances, seedlabels)
#    print distances.shape
#    print agglom.maxbaseid
#    print agglom.currid

    havepure = avoidmixedlabels
    i = 0
    while agglom.has_more():
        if havepure:
            dist, id1, id2, found = agglom.find_next_join(avoidmixedlabels)
            havepure = found
#            if not havepure:
#                print "\tDONE WITH PURE CLUSTERS"
        if not havepure:
            dist, id1, id2, found = agglom.find_next_join()
        newsubtree = agglom.add_subtree(id1, id2)
#        print id1, id2, dist, newsubtree.size
        treemat[i,:] = np.asarray([id1, id2, dist, newsubtree.size])
        i += 1
    
    return treemat
