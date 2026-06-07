import heapq

def prim_mst(graph, start=0):
    n = len(graph)
    visited = [False] * n
    pq = [(0, start)]  # (weight, node)

    total_weight = 0
    mst_edges = []

    while pq:
        w, u = heapq.heappop(pq)

        if visited[u]:
            continue

        visited[u] = True
        total_weight += w

        for v, weight in graph[u]:
            if not visited[v]:
                heapq.heappush(pq, (weight, v))
                mst_edges.append((u, v, weight))

    return total_weight, mst_edges

graph = {
    0: [(1,1),(2,3)],
    1: [(0,1),(2,2),(3,4)],
    2: [(0,3),(1,2),(3,5)],
    3: [(1,4),(2,5)]
}

import math

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def kruskal_clustering(points, k):
    n = len(points)
    edges = []

    for i in range(n):
        for j in range(i+1, n):
            edges.append((dist(points[i], points[j]), i, j))

    edges.sort()

    dsu = DSU(n)
    clusters = n

    mst = []

    for w, u, v in edges:
        if dsu.union(u, v):
            mst.append((u, v, w))
            clusters -= 1
            if clusters == k:
                break

    # build final clusters
    groups = {}
    for i in range(n):
        root = dsu.find(i)
        groups.setdefault(root, []).append(i)

    return groups, mst