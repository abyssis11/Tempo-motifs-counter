import numpy as np
import copy
from multiprocessing import Process, Manager, Lock
from multiprocessing.managers import BaseManager
import time
from read_csv import getEdges

class Counter1D:
    def __init__(self, n=0):
        self.data = np.zeros(n)

class Counter2D:
    def __init__(self, n=0, m=0 ):
        self.data = np.zeros((n, m))
    
    def get_value(self, i, j):
        return self.data[i, j]

    def get_all_values(self):
        return self.data

    def update_value(self, i, j, operation, value):
        if operation == '-':
            self.data[i, j] -= value
        elif operation == '+':
            self.data[i, j] += value


class Counter3D:
    def __init__(self, n=0, m=0, p=0):
        self.data = np.zeros((n,m,p))

    def get_value(self, i, j, k):
        return self.data[i, j, k]

    def get_all_values(self):
        return self.data

    def update_value(self, i, j, k, operation, value):
        if operation == '-':
            self.data[i, j, k] -= value
        elif operation == '+':
            self.data[i, j, k] += value 

class MyManager(BaseManager): pass

# Register your custom class in the manager
MyManager.register('Counter3D', Counter3D)
MyManager.register('Counter2D', Counter2D)
MyManager.register('dict', dict)
MyManager.register('list', list)

class Index:
    def __init__(self):
        self.value = 0

    def up(self):
        self.value += 1

# Triad edge data consists of a neighbor, a direction, and an indicator of whether
# the edge connects with wich endpoint (u or v).
class TriadEdgeData:
    def __init__(self, nbr, direction, u_or_v):
        self.nbr = nbr
        self.direction = direction
        self.u_or_v = u_or_v

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

class ThreeTEdgeTriadCounter:
    def __init__(self, max_nodes, node_u, node_v):
        self.max_nodes = max_nodes
        self.node_u = node_u
        self.node_v = node_v
        self.pre_sum = None
        self.pos_sum = None
        self.mid_sum = None
        self.triad_counts = None
        self.pre_nodes = None
        self.pos_nodes = None

    def counts(self, dir1, dir2, dir3):
        return self.triad_counts.data[dir1, dir2, dir3]

    def initializeCounters(self):
        self.pre_sum = Counter3D(2,2,2)
        self.pos_sum = Counter3D(2,2,2)
        self.mid_sum = Counter3D(2,2,2)
        self.triad_counts = Counter3D(2,2,2)
        self.pre_nodes = Counter3D(2,2, self.max_nodes)
        self.pos_nodes = Counter3D(2,2, self.max_nodes)

    def popPre(self, event: TriadEdgeData):
        nbr = event.nbr
        direction = event.direction
        u_or_v = event.u_or_v
        if not self.isEdgeNode(nbr):
            self.pre_nodes.data[direction, u_or_v, nbr] -= 1
            for i in range(2):
                self.pre_sum.data[u_or_v, direction, i] -= self.pre_nodes.data[i, 1 - u_or_v, nbr]

    def popPos(self, event: TriadEdgeData):
        nbr = event.nbr
        direction = event.direction
        u_or_v = event.u_or_v
        if not self.isEdgeNode(nbr):
            self.pos_nodes.data[direction, u_or_v, nbr] -= 1
            for i in range(2):
                self.pos_sum.data[u_or_v, direction, i] -= self.pos_nodes.data[i, 1 - u_or_v, nbr]

    def pushPre(self, event: TriadEdgeData):
        nbr = event.nbr
        direction = event.direction
        u_or_v = event.u_or_v
        if not self.isEdgeNode(nbr):
            for i in range(2):
                self.pre_sum.data[1 - u_or_v, i, direction] += self.pre_nodes.data[i, 1 - u_or_v, nbr]
            self.pre_nodes.data[direction, u_or_v, nbr] += 1

    def pushPos(self, event: TriadEdgeData):
        nbr = event.nbr
        direction = event.direction
        u_or_v = event.u_or_v
        if not self.isEdgeNode(nbr):
            for i in range(2):
                self.pos_sum.data[1 - u_or_v, i, direction] += self.pos_nodes.data[i, 1 - u_or_v, nbr]
            self.pos_nodes.data[direction, u_or_v, nbr] += 1

    def processCurrent(self, event: TriadEdgeData):
        nbr = event.nbr
        direction = event.direction
        u_or_v = event.u_or_v
        if not self.isEdgeNode(nbr):
            for i in range(2):
                self.mid_sum.data[1 - u_or_v, i, direction] -= self.pre_nodes.data[i, 1 - u_or_v, nbr]
                self.mid_sum.data[u_or_v, direction, i] += self.pos_nodes.data[i, 1 - u_or_v, nbr]

        if self.isEdgeNode(nbr):
            u_to_v = 0
            if(nbr == self.node_u and direction == 0) or (nbr == self.node_v and direction == 1):
                u_to_v = 1

            self.triad_counts.data[0, 0, 0] += self.mid_sum.data[u_to_v, 0, 0] + self.pos_sum.data[u_to_v, 0, 1] + self.pre_sum.data[1- u_to_v, 1, 1]
            self.triad_counts.data[1, 0, 0] += self.mid_sum.data[u_to_v, 1, 0] + self.pos_sum.data[1 - u_to_v, 0, 1] + self.pre_sum.data[1 - u_to_v, 0, 1]
            self.triad_counts.data[0, 1, 0] += self.mid_sum.data[1 - u_to_v, 0, 0] + self.pos_sum.data[u_to_v, 1, 1] + self.pre_sum.data[1 - u_to_v, 1, 0]
            self.triad_counts.data[1, 1, 0] += self.mid_sum.data[1 - u_to_v, 1, 0] + self.pos_sum.data[1 - u_to_v, 1, 1] + self.pre_sum.data[1 - u_to_v, 0, 0]
            self.triad_counts.data[0, 0, 1] += self.mid_sum.data[u_to_v, 0, 1] + self.pos_sum.data[u_to_v, 0, 0] + self.pre_sum.data[u_to_v, 1, 1]
            self.triad_counts.data[1, 0, 1] += self.mid_sum.data[u_to_v, 1, 1] + self.pos_sum.data[1 - u_to_v, 0, 0] + self.pre_sum.data[u_to_v, 0, 1]
            self.triad_counts.data[0, 1, 1] += self.mid_sum.data[1 - u_to_v, 0, 1] + self.pos_sum.data[u_to_v, 1, 0] + self.pre_sum.data[u_to_v, 1, 0]
            self.triad_counts.data[1, 1, 1] += self.mid_sum.data[1 - u_to_v, 1, 1] + self.pos_sum.data[1 - u_to_v, 1, 0] + self.pre_sum.data[u_to_v, 0, 0]

    def isEdgeNode(self, nbr):
        return nbr == self.node_u or nbr == self.node_v

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

def AddTriadEdgeData(events, ts_inidces, index, u, v, nbr, key1, key2, edges):
    timestamps = []

    # zasebna funkcija ili klassa?
    for edge in edges:
        if (u,v) in edge:
            timestamps.append(edge[1])
    
    j = 0
    for i in range(len(timestamps)):
        # zasto?
        #j+=1
        #### IPAK KORISTENO i!!!
        ts_inidces.append((timestamps[i], index.value))
        events.append(TriadEdgeData(nbr, key1, key2))
        # ++index ili index++?
        index.up()
    #pass

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

def process_undir_edges(subset_of_undir_edges, counts, delta, lock):
    for undir_edge in subset_of_undir_edges:
        src, dst = [x for x in undir_edge]
        local = Counter3D()
        Count2Node3Edge(src, dst, delta, local)
        with lock:
            counts.update_value(0, 0, '+', local.data[0, 1, 0] + local.data[1, 0, 1])
            counts.update_value(0, 1, '+', local.data[1, 0, 0] + local.data[0, 1, 1])
            counts.update_value(1, 0, '+', local.data[0, 0, 0] + local.data[1, 1, 1])
            counts.update_value(1, 1, '+', local.data[0, 0, 1] + local.data[1, 1, 0])

def Count2Node3Edge_main(delta, counts, edges):
    undir_edges = []
    static = []
    for edge in edges:
        #src, dst = [x for x in edge[0]]
        static.append(edge[0])

    for edge in static:
        src, dst = [x for x in edge]

        if src < dst  or (dst < src and (dst, src) not in static):
            if (src, dst) not in undir_edges:
                undir_edges.append((src, dst))
    
    with MyManager() as manager:
        # Create shared counters and lock
        shared_counts = manager.Counter2D(2, 2)
        lock = Lock()


        num_processes = 4  # Or any number of processes you want
        chunk_size = len(undir_edges) // num_processes
        processes = []
        
        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < num_processes - 1 else len(undir_edges)
            p = Process(target=process_undir_edges, args=(undir_edges[start_index:end_index], shared_counts, delta, lock))
            processes.append(p)
            p.start()

            # Wait for all processes to finish
            for p in processes:
                p.join()

            counts.data = copy.deepcopy(shared_counts.get_all_values())
            #counts.data[0, 0] = shared_counts.get_value(0, 0)
            #counts.data[0, 1] = shared_counts.get_value(0, 1)
            #counts.data[1, 0] = shared_counts.get_value(1, 0)
            #counts.data[1, 1] = shared_counts.get_value(1, 1)

def process_nodes(subset_nodes, edges, degrees, lock):
    for node in subset_nodes:
        src = node
        nbrs = getNeighbors(edges, src)
        #degrees[src] = (len(nbrs), src)
        with lock:
            degrees.insert(src, (len(nbrs), src))
            degrees.pop(src + 1)   

def process_order(start, end, degrees, order, lock):
    for i in range(start, end):
        key, dat = degrees[i]
        with lock:
            order.insert(dat, i)
            order.pop(dat + 1)   
            #order[dat] = i

def process_triangles_centered(subset_nodes, us, vs, ws, order, static, edges, lock):
    for node in subset_nodes:
        src = node
        src_pos = order[src]

        nbrs = getNeighbors(edges, src)
        neighbors_higher = []
        for nbr in nbrs:
            if order[nbr] > src_pos:
                neighbors_higher.append(nbr)

        for ind1 in range(len(neighbors_higher)):
            for ind2 in range(ind1+1, len(neighbors_higher)):
                dst1 = neighbors_higher[ind1]
                dst2 = neighbors_higher[ind2]

                if (dst1, dst2) in static or (dst2, dst1) in static:
                    with lock:
                        us.append(src)
                        vs.append(dst1)
                        ws.append(dst2)


def GetAllStaticTriangles(us, vs, ws, static, edges):
    #max_nodes = max(max(x) for x in static)
    max_nodes = len(static)
    degrees = [(0, 0) for _ in range(max_nodes)]
    #degrees = {}

    nodes = getNodes(edges)

    with MyManager() as manager:
        shared_degrees = manager.list()
        #shared_degrees = degrees
        for tup in degrees:
            shared_degrees.append(tup)
        
        lock = Lock()

        num_processes = 4  # Or any number of processes you want
        chunk_size = len(nodes) // num_processes
        processes = []

        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < num_processes - 1 else len(nodes)
            p = Process(target=process_nodes, args=(nodes[start_index:end_index], edges, shared_degrees, lock))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Copy data back to original counters
        degrees = copy.deepcopy(shared_degrees)

    #degrees = dict(sorted(degrees.items()))
    degrees.sort()
    #print(degrees)
    #order = {}
    order = [None] * max_nodes

    with MyManager() as manager:
        shared_order = manager.list()
        for el in order:
            shared_order.append(el)
        
        lock = Lock()

        num_processes = 4  # Or any number of processes you want
        chunk_size = max_nodes // num_processes
        processes = []

        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < num_processes - 1 else max_nodes
            p = Process(target=process_order, args=(start_index, end_index, degrees, shared_order, lock))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Copy data back to original counters
        order = copy.deepcopy(shared_order)
        
    with MyManager() as manager:
        shared_us = manager.list()
        for el in us:
            shared_us.append(el)
        shared_vs = manager.list()
        for el in vs:
            shared_vs.append(el)
        shared_ws = manager.list()
        for el in ws:
            shared_ws.append(el)
        
        lock = Lock()

        num_processes = 4  # Or any number of processes you want
        chunk_size = len(nodes) // num_processes
        processes = []

        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < num_processes - 1 else len(nodes)
            p = Process(target=process_triangles_centered, args=(nodes[start_index:end_index], shared_us, shared_vs, shared_ws, order, static, edges, lock))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Copy data back to original counters
        us = copy.deepcopy(shared_us)
        vs = copy.deepcopy(shared_vs)
        ws = copy.deepcopy(shared_ws)
    
        return us, vs, ws
    '''   
    #print(order)
    for node in nodes:
        src = node
        src_pos = order[src]

        nbrs = getNeighbors(edges, src)
        neighbors_higher = []
        for nbr in nbrs:
            if order[nbr] > src_pos:
                neighbors_higher.append(nbr)

        for ind1 in range(len(neighbors_higher)):
            for ind2 in range(ind1+1, len(neighbors_higher)):
                dst1 = neighbors_higher[ind1]
                dst2 = neighbors_higher[ind2]

                if (dst1, dst2) in static or (dst2, dst1) in static:
                    us.append(src)
                    vs.append(dst1)
                    ws.append(dst2)
    '''
    #print(ws)

def process_uv_vs_ws(us, vs, ws, edge_counts, assignments, lock):
    for i in range(len(us)):
        u = us[i]
        v = vs[i]
        w = ws[i]
        counts_uv = edge_counts[(min(u,v), max(u, v))]
        counts_uw = edge_counts[(min(u,w), max(u, w))]
        counts_vw = edge_counts[(min(v,w), max(v, w))]

        if counts_uv >= max(counts_uw, counts_vw):
            with lock:
                assignment = assignments.get((min(u,v), max(u, v)))
                assignment.append(w)
                assignments.update({(min(u,v), max(u, v)): assignment})
        elif counts_uw >= max(counts_uv, counts_vw):
            with lock:
                assignment = assignments.get((min(u,w), max(u, w)))
                assignment.append(v)
                assignments.update({(min(u,w), max(u, w)):assignment })
                #assignments.update({(min(u,w), max(u, w)): v})
        elif counts_vw >= max(counts_uv, counts_uw):
            with lock:
                assignment = assignments.get(((min(v, w), max(v, w))))
                assignment.append(u)
                assignments.update({(min(v, w), max(v, w)): assignment})

def process_all_edges(subset_all_edges, counts, assignments, edges, lock):
    # Count triangles on edges with the assigned neighbors
    for edge in subset_all_edges:
        u = edge[0] # Key
        v = edge[1] # Data
        # Continue if no assignment
        if (u, v) not in assignments:
            continue
        uv_assignment = assignments[(u, v)]
        # Continue if no data
        if len(uv_assignment) == 0:
            continue

        # Get all events on (u, v)
        events = []
        ts_inidces = []
        index = Index()
        nbr_index = 0
        # Assign indices from 0, 1, ..., num_nbrs + 2
        AddTriadEdgeData(events, ts_inidces, index, u, v, nbr_index, 0, 1, edges)
        nbr_index += 1
        AddTriadEdgeData(events, ts_inidces, index, v, u, nbr_index, 0, 0, edges)
        nbr_index += 1

        # Get all events on triangles assigned to (u, v)
        for w in uv_assignment:
            AddTriadEdgeData(events, ts_inidces, index, w, u, nbr_index, 0, 0, edges)
            AddTriadEdgeData(events, ts_inidces, index, w, v, nbr_index, 0, 1, edges)
            AddTriadEdgeData(events, ts_inidces, index, u, w, nbr_index, 1, 0, edges)
            AddTriadEdgeData(events, ts_inidces, index, v, w, nbr_index, 1, 1, edges)
            nbr_index+=1

        # Put events in sorted order
        ts_inidces.sort()
        timestamps = []
        sorted_events = []
        for ts in ts_inidces:
            timestamps.append(ts[0])
            sorted_events.append(events[ts[1]])

        # Get the counts and update the counter
        tetc = ThreeTEdgeTriadCounter(nbr_index, 0, 1)
        tetc.count(sorted_events, timestamps, delta)

        with lock:
            for dir1 in range(2):
                for dir2 in range(2):
                    for dir3 in range(2):
                        counts.update_value(dir1, dir2, dir3, '+', tetc.counts(dir1, dir2, dir3))

def countTriangles(delta, counts, edges):
    static = []
    temporal_data = {}
    for edge in edges:
        src, dst = [x for x in edge[0]]
        static.append((src, dst))
        temporal_data.setdefault((src, dst), []).append(edge)

    edge_counts = {}
    assignments = {}
    for edge in static:
        src, dst = [x for x in edge]
        min_node = min(src, dst)
        max_node = max(src, dst)
        #edge_counts[(min_node, max_node)] += len(temporal_data[(src, dst)])
        edge_counts[(min_node, max_node)] = edge_counts.get((min_node, max_node), 0) + len(temporal_data.get((src, dst), []))
        assignments[(min_node, max_node)] = []

    us = [] 
    vs = []
    ws = []
    us, vs, ws = GetAllStaticTriangles(us, vs, ws, static, edges)
    #print(us)
    #print(vs)
    #print(ws)

    with MyManager() as manager:
        # Create shared counters and lock
        shared_assignments = manager.dict()
        # Set the contents of the shared dictionary equal to existing_dict
        for key, value in assignments.items():
            shared_assignments.update({key: value})

        lock = Lock()

        num_processes = 4  # Or any number of processes you want
        chunk_size = len(us) // num_processes
        processes = []

        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < num_processes - 1 else len(us)
            p = Process(target=process_uv_vs_ws, args=(us[start_index:end_index], vs[start_index:end_index], ws[start_index:end_index], edge_counts, shared_assignments, lock))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Copy data back to original counters
        assignments = copy.deepcopy(shared_assignments)
        #for key, value in shared_assignments.items():
        #    assignments[key] = value


    all_edges = []
    all_nodes = getNodes(edges)
    for node in all_nodes:
        nbrs = getNeighbors(edges, node)
        for nbr in nbrs:
            if (node, nbr) in assignments and len(assignments[(node, nbr)]) > 0:
                all_edges.append((node, nbr))

    with MyManager() as manager:
        # Create shared counters and lock
        shared_counts = manager.Counter3D(2, 2, 2)
        lock = Lock()

        num_processes = 4  # Or any number of processes you want
        chunk_size = len(all_edges) // num_processes
        processes = []

        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < num_processes - 1 else len(all_edges)
            p = Process(target=process_all_edges, args=(all_edges[start_index:end_index], shared_counts, assignments, edges, lock))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Copy data back to original counters 
        counts.data = copy.deepcopy(shared_counts.get_all_values())
    

def process_center(subset_of_centers, pre_counts, pos_counts, mid_counts, lock, delta, edges):
    for center in subset_of_centers:
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

        with lock:
            for dir1 in range(2):
                for dir2 in range(2):
                    for dir3 in range(2):
                        pre_counts.update_value(dir1, dir2, dir3, '+', tesc.preCounts(dir1, dir2, dir3))
                        pos_counts.update_value(dir1, dir2, dir3, '+', tesc.posCounts(dir1, dir2, dir3))
                        mid_counts.update_value(dir1, dir2, dir3, '+', tesc.midCounts(dir1, dir2, dir3))

        for nbr in nbrs:
            edge_counts = Counter3D()
            Count2Node3Edge(center, nbr, delta, edge_counts)
            with lock:
                for dir1 in range(2):
                    for dir2 in range(2):
                        for dir3 in range(2):
                            pre_counts.update_value(dir1, dir2, dir3, '-', edge_counts.data[dir1, dir2, dir3])
                            pos_counts.update_value(dir1, dir2, dir3, '-', edge_counts.data[dir1, dir2, dir3])
                            mid_counts.update_value(dir1, dir2, dir3, '-', edge_counts.data[dir1, dir2, dir3])

def countStars(delta, pre_counts, pos_counts, mid_counts, edges):
    centers = getNodes(edges)

    with MyManager() as manager:
        # Create shared counters and lock
        shared_pre_counts = manager.Counter3D(2, 2, 2)
        shared_pos_counts = manager.Counter3D(2, 2, 2)
        shared_mid_counts = manager.Counter3D(2, 2, 2)
        lock = Lock()

        num_processes = 4  # Or any number of processes you want
        chunk_size = len(centers) // num_processes
        processes = []

        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = start_index + chunk_size if i < num_processes - 1 else len(centers)
            p = Process(target=process_center, args=(centers[start_index:end_index], shared_pre_counts, shared_pos_counts, shared_mid_counts, lock, delta, edges))
            processes.append(p)
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Copy data back to original counters
        '''
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    pre_counts.data[i, j, k] = shared_pre_counts.get_value(i, j, k)
                    pos_counts.data[i, j, k] = shared_pos_counts.get_value(i, j, k)
                    mid_counts.data[i, j, k] = shared_mid_counts.get_value(i, j, k)
        '''
        pre_counts.data = copy.deepcopy(shared_pre_counts.get_all_values())
        pos_counts.data = copy.deepcopy(shared_pos_counts.get_all_values())
        mid_counts.data = copy.deepcopy(shared_mid_counts.get_all_values())

def motifCounter(delta, counts, edges):
    #counts = Counter2D(6,6)

    # count 2 Nodes 3 Edegs
    edge_counts = Counter2D(2, 2)
    Count2Node3Edge_main(delta, edge_counts, edges)
    counts.data[4, 0] = edge_counts.data[0, 0]
    counts.data[4, 1] = edge_counts.data[0, 1]
    counts.data[5, 0] = edge_counts.data[1, 0]
    counts.data[5, 1] = edge_counts.data[1, 1]


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
    triad_counts = Counter3D(2, 2, 2)
    countTriangles(delta, triad_counts, edges)
    counts.data[0, 2] = triad_counts.data[0, 0, 0]
    counts.data[0, 3] = triad_counts.data[0, 0, 1]
    counts.data[1, 2] = triad_counts.data[0, 1, 0]
    counts.data[1, 3] = triad_counts.data[0, 1, 1]
    counts.data[2, 4] = triad_counts.data[1, 0, 0]
    counts.data[2, 5] = triad_counts.data[1, 0, 1]
    counts.data[3, 4] = triad_counts.data[1, 1, 0]
    counts.data[3, 5] = triad_counts.data[1, 1, 1]



if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{:0.0f}'.format})
    start = time.time ()

    file = "Cs_5DIZ_24_11_2023_10_22_A2.csv"
    delta = 60

    edges = getEdges(file)
    counts = Counter2D(6, 6)
    motifCounter(delta, counts, edges)
    print(counts.data)

    end = time.time()
    print("Time to complete: ", end - start)
