import numpy as np

class Counter2D:
    def __init__(self, n=0, m=0 ):
        self.data = np.zeros((n, m))


class Counter3D:
    def __init__(self, n=0, m=0, p=0):
        self.data = np.zeros((n,m,p))

class Index:
    def __init__(self):
        self.value = 0

    def up(self):
        self.value += 1

# consists of a neighbor and direction
class StarEdgeData:
    def __init__(self, nbr, direction):
        self.nbr = nbr
        self.dir = direction

class StarCounter:
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes
        self.pre_sum
        self.pos_sum
        self.mid_sum
        self.pre_counts
        self.pos_counts
        self.mid_counts
        self.pre_nodes
        self.pos_nodes
    
    def preCounts(self, dir1, dir2, dir3):
        return self.pre_counts(dir1, dir2, dir3)

    def posCounts(self, dir1, dir2, dir3):
        return self.pos_counts(dir1, dir2, dir3)

    def midCounts(self, dir1, dir2, dir3):
        return self.mid_counts(dir1, dir2, dir3)

    def initializeCounters(self):
        self.pre_sum = Counter2D(2,2)
        self.pos_sum = Counter2D(2,2)
        self.mid_sum = Counter2D(2,2)
        self.pre_counts = Counter3D(2,2,2)
        self.mid_counts = Counter3D(2,2,2)
        self.pos_counts = Counter3D(2,2,2)
        self.pre_nodes = Counter2D(2, self.max_nodes)
        self.pos_nodes = Counter2D(2, self.max_nodes)

    def popPre(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        self.pre_nodes[direction, nbr] -= 1
        for i in range(2):
            self.pre_sum[direction, i] -= self.pre_nodes[i, nbr]

    def popPos(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        self.pos_nodes[direction, nbr] -= 1
        for i in range(2):
            self.pos_sum[direction, i] -= self.pos_nodes[i, nbr]

    def pushPre(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        for i in range(2):
            self.pre_nodes[direction, nbr] += 1

    def pushPos(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        for i in range(2):
            self.pos_nodes[direction, nbr] += 1

    def processCurrent(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        for i in range(2):
            self.mid_sum[i, direction] -= self.pre_nodes[i, nbr]

        for i in range(2):
            for j in range (2):
                self.pre_counts[i, j, direction] += self.pre_sum[i, j]
                self.pos_counts[direction, i, j] += self.pos_sum[i, j]
                self.mid_counts[i, direction, j] += self.mid_sum[i, j]

        for i in range(2):
            self.mid_sum[direction, i] += self.pos_nodes[i, nbr]

    def count(events, timestapms, delta):
        self.initializeCounters()
        
        if len(events) != len(timestapms):
            print("Number of events must match number of timestamps") 
        else:
            start = 0
            end = 0
            L = len(timestapms)
            for i in range(L):
                tj = timestapms[i]
                while start < L and timestapms[start] < tj - delta:
                    self.popPre(events[start])
                    start += 1
                
                while end < L and timestamps[end] <= tj + delta:
                    self.pushPos(events[end])
                    end += 1

                self.popPos(events[i])
                self.processCurrent(events[i])
                self.pushPre(events[i])

''''
class Node:
    def __init__(self, value):
        self.vertex = value
        self.outEdges = None
        self.inEdges = None

class StaticGraph:
        def __init__(self, num):
            self.V = num
            self.graph = [None] * self.V
        
        def add_edge(self, s, d):
            node = Node(d)
            node.next = self.graph[s]
            self.graph[s] = node
'''

def getNodes(edges):
    centers = []
    for edge in edges:
        u, v = [x for x in edge[0]]
        if u not in centers:
            centers.append(u)
        if v not in centers:
            centers.append(v)
    return centers
        
def getNeighbors(edges, center):
    nbrs = []
    for edge in edges:
        if center in edge[0]:
            nbr, = [x for x in edge[0] if x != center]
            if nbr not in nbrs:
                nbrs.append(nbr)
    return nbrs

def AddStarEdgeData(ts_inidces, events, index, u, v, nbr, key, edges):
    ts_vec = []

    # zasebna funkcija ili klassa?
    for edge in edges:
        if (u,v) in edge:
            ts_vec.append[edge[1]]
    
    j = 0
    for i in range(len(ts_vec)):
        # zasto?
        j+=1
        #### IPAK KORISTENO i!!!
        ts_inidces.append((ts_vec[i], index.value))
        events.append(StarEdgeData(nbr, key))
        index.up()
    #pass

def countStars(delta, pre_counts, pos_counts, mid_counts, edges):
    centers = getNodes(edges)
    for center in centers:
        nbrs = getNeighbors(edges, center)
        nbr_index = 0
        index = Index()
        ts_inidces = []
        events = []
        for nbr in nbrs:
            AddStarEdgeData(ts_inidces, events, index, center, nbr, nbr_index, 0, edges)
            AddStarEdgeData(ts_inidces, events, index, nbr, center, nbr_index, 1, edges)
            nbr_index += 1

        timestapms = []
        ordered_events = []
        for ts in ts_inidces:
            timestapms.append(ts[0])
            ordered_events.append(events[ts[1]])

        tesc = StarCounter(nbr_index)
        tesc.count(ordered_events, timestapms, delta)
        for dir1 in range(2):
            for dir2 in range(2):
                for dir3 in range(2):
                    pre_counts[dir1, dir2, dir3] += tesc.preCounts(dir1, dir2, dir3)
                    pos_counts[dir1, dir2, dir3] += tesc.posCounts(dir1, dir2, dir3)
                    mid_counts[dir1, dir2, dir3] += tesc.midCounts(dir1, dir2, dir3)

        #for nbr in nbrs:

            
    pass

def motifCounter(delta, counts, edges):
    # counts = Counter2D(6,6)

    # count 2 Nodes 3 Edegs

    # count Stars
    pre_counts = Counter3D(2, 2, 2)
    pos_counts = Counter3D(2, 2, 2)
    mid_counts = Counter3D(2, 2, 2)
    countStars(delta, pre_counts, pos_counts, mid_counts, edges)

    # count Triangles

    pass





def getEdges(edges):
    with open("graph1.txt") as file:
        for line in file:
            u, v, t = [int(x) for x in line.rstrip().split(' ')]
            edges.append(((u,v),t))


if __name__ == "__main__":
    edges = []
    getEdges(edges)
    #staticgraph
    #print(edges)
    #print(getNodes(edges))
    #print(getNeighbors(edges, 0))
    counts = Counter2D(6, 6)
    delta = 10
