import numpy as np
import copy

class Counter1D:
    def __init__(self, n=0):
        self.data = np.zeros(n)

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
        self.pre_sum = None
        self.pos_sum = None
        self.mid_sum = None
        self.pre_counts = None
        self.pos_counts = None
        self.mid_counts = None
        self.pre_nodes = None
        self.pos_nodes = None
    
    def preCounts(self, dir1, dir2, dir3):
        return self.pre_counts.data[dir1, dir2, dir3]

    def posCounts(self, dir1, dir2, dir3):
        return self.pos_counts.data[dir1, dir2, dir3]

    def midCounts(self, dir1, dir2, dir3):
        return self.mid_counts.data[dir1, dir2, dir3]

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
        self.pre_nodes.data[direction, nbr] -= 1
        for i in range(2):
            self.pre_sum.data[direction, i] -= self.pre_nodes.data[i, nbr]

    def popPos(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        self.pos_nodes.data[direction, nbr] -= 1
        for i in range(2):
            self.pos_sum.data[direction, i] -= self.pos_nodes.data[i, nbr]

    def pushPre(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        for i in range(2):
            self.pre_sum.data[i, direction] += self.pre_nodes.data[i, nbr]
        self.pre_nodes.data[direction, nbr] += 1

    def pushPos(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        for i in range(2):
            self.pos_sum.data[i, direction] += self.pos_nodes.data[i, nbr]
        self.pos_nodes.data[direction, nbr] += 1

    def processCurrent(self, event: StarEdgeData):
        nbr = event.nbr
        direction = event.dir
        for i in range(2):
            self.mid_sum.data[i, direction] -= self.pre_nodes.data[i, nbr]

        for i in range(2):
            for j in range (2):
                self.pre_counts.data[i, j, direction] += self.pre_sum.data[i, j]
                self.pos_counts.data[direction, i, j] += self.pos_sum.data[i, j]
                self.mid_counts.data[i, direction, j] += self.mid_sum.data[i, j]

        for i in range(2):
            self.mid_sum.data[direction, i] += self.pos_nodes.data[i, nbr]

    def count(self, events, timestapms, delta):
        self.initializeCounters()
        
        if len(events) != len(timestapms):
            print("Number of events must match number of timestamps") 
        else:
            start = 0
            end = 0
            L = len(timestapms)
            for i in range(L):
                tj = timestapms[i]
                while start < L and timestapms[start] < (tj - delta):
                    self.popPre(events[start])
                    start += 1
                
                while end < L and timestapms[end] <= (tj + delta):
                    self.pushPos(events[end])
                    end += 1

                self.popPos(events[i])
                self.processCurrent(events[i])
                self.pushPre(events[i])


class ThreeEdgeMotifCounter:
    def __init__(self, size):
        self.size = size
        self.counts1 = None
        self.counts2 = None
        self.counts3 = None

    def incrementCounts(self, event):
        for i in range(self.size):
            for j in range(self.size):
                self.counts3.data[i, j, event] += self.counts2.data[i, j]
        
        for i in range(self.size):
            self.counts2.data[i, event] += self.counts1.data[i]
        
        self.counts1.data[event] += 1

    def decrementCounts(self, event):
        self.counts1.data[event] -= 1
        for i in range(self.size):
            self.counts2.data[event, i] -= self.counts1.data[i]

    def count(self, event_string, timestamps, delta, counts):
        self.counts1 = Counter1D(self.size)
        self.counts2 = Counter2D(self.size, self.size)
        self.counts3 = Counter3D(self.size, self.size, self.size)

        if len(event_string) != len(timestamps):
            print("Number of events must be equal number of timestamps")
        else:
            start = 0
            for end in range(len(event_string)):
                while (timestamps[start] + delta) < timestamps[end]:
                    self.decrementCounts(event_string[start])
                    start += 1
                self.incrementCounts(event_string[end])

            counts.data = copy.deepcopy(self.counts3.data)


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
            ts_vec.append(edge[1])
    
    j = 0
    for i in range(len(ts_vec)):
        # zasto?
        #j+=1
        #### IPAK KORISTENO i!!!
        ts_inidces.append((ts_vec[i], index.value))
        events.append(StarEdgeData(nbr, key))
        index.up()
    #pass

def addStarEdges(combined, u, v, key):
    # trebali li staticni graf?

    timestamps = []
    for edge in edges:
        if (u,v) in edge:
            timestamps.append(edge[1])
    
    for timestamp in timestamps:
        combined.append((timestamp, key))


def Count2Node3Edge(u, v, delta, counts):
    combined = []
    addStarEdges(combined, u, v, 0)
    addStarEdges(combined, v, u, 1)
    combined.sort()

    counter = ThreeEdgeMotifCounter(2)
    in_out = [0] * len(combined)
    timestamps = [0] * len(combined)
    for k in range(len(combined)):
        in_out[k] = combined[k][1] #Dat
        timestamps[k] = combined[k][0] #Key
    
    counter.count(in_out, timestamps, delta, counts)


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

        ts_inidces.sort()
        timestapms = []
        ordered_events = []
        for ts in ts_inidces:
            timestapms.append(ts[0])
            ordered_events.append(events[ts[1]])

        tesc = StarCounter(nbr_index)
        tesc.count(ordered_events, timestapms, delta)
        # ++j
        #print("Prije")
        #print(mid_counts.data)
        for dir1 in range(2):
            for dir2 in range(2):
                for dir3 in range(2):
                    pre_counts.data[dir1, dir2, dir3] += tesc.preCounts(dir1, dir2, dir3)
                    pos_counts.data[dir1, dir2, dir3] += tesc.posCounts(dir1, dir2, dir3)
                    mid_counts.data[dir1, dir2, dir3] += tesc.midCounts(dir1, dir2, dir3)

        #print("Poslje")
        #print(mid_counts.data)

        for nbr in nbrs:
            edge_counts = Counter3D()
            Count2Node3Edge(center, nbr, delta, edge_counts)
            #print(edge_counts.data)
            #++j
            for dir1 in range(2):
                for dir2 in range(2):
                    for dir3 in range(2):
                        pre_counts.data[dir1, dir2, dir3] -= edge_counts.data[dir1, dir2, dir3]
                        pos_counts.data[dir1, dir2, dir3] -= edge_counts.data[dir1, dir2, dir3]
                        mid_counts.data[dir1, dir2, dir3] -= edge_counts.data[dir1, dir2, dir3]
            

            
    pass

def motifCounter(delta, counts, edges):
    #counts = Counter2D(6,6)

    # count 2 Nodes 3 Edegs

    # count Stars
    pre_counts = Counter3D(2, 2, 2)
    pos_counts = Counter3D(2, 2, 2)
    mid_counts = Counter3D(2, 2, 2)
    countStars(delta, pre_counts, pos_counts, mid_counts, edges)
    counts.data[0, 0] = mid_counts.data[1, 1, 1]
    counts.data[0, 1] = mid_counts.data[1, 1, 0]
    counts.data[0, 4] = pos_counts.data[1, 1, 0]
    counts.data[0, 5] = pos_counts.data[1, 1, 1]

    counts.data[1, 0] = mid_counts.data[1, 0, 1]
    counts.data[1, 1] = mid_counts.data[1, 0, 0]
    counts.data[1, 4] = pos_counts.data[1, 0, 0]
    counts.data[1, 5] = pos_counts.data[1, 0, 1]

    counts.data[2, 0] = mid_counts.data[0, 1, 0]
    counts.data[2, 1] = mid_counts.data[0, 1, 1]
    counts.data[2, 2] = pos_counts.data[0, 1, 0]
    counts.data[2, 3] = pos_counts.data[0, 1, 1]

    counts.data[3, 0] = mid_counts.data[0, 0, 0]
    counts.data[3, 1] = mid_counts.data[0, 0, 1]
    counts.data[3, 2] = pos_counts.data[0, 0, 0]
    counts.data[3, 3] = pos_counts.data[0, 0, 1]


    counts.data[4, 2] = pre_counts.data[0, 1, 0]
    counts.data[4, 3] = pre_counts.data[0, 1, 1]
    counts.data[4, 4] = pre_counts.data[1, 0, 0]
    counts.data[4, 5] = pre_counts.data[1, 0, 1]

    counts.data[5, 2] = pre_counts.data[0, 0, 0]
    counts.data[5, 3] = pre_counts.data[0, 0, 1]
    counts.data[5, 4] = pre_counts.data[1, 1, 0]
    counts.data[5, 5] = pre_counts.data[1, 1, 1]

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
    motifCounter(delta, counts, edges)
    print(counts.data)

    #c = Counter1D(5)
    #print(c.data)
