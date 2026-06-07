from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def has_cycle(self):
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in list(self.graph):
            if node not in visited:
                if dfs(node):
                    return True

        return False


# ------------------ TESTING ------------------

def test_case(edges):
    g = Graph()
    for u, v in edges:
        g.add_edge(u, v)
    return g.has_cycle()


# Case 1
case1 = [('A','B'), ('B','C'), ('C','D'), ('E','D')]

# Case 2
case2 = [('A','B'), ('B','C'), ('C','A'), ('C','D')]

# Case 3
case3 = [('A','B'), ('B','C'), ('C','A'),
         ('D','E'), ('E','F'), ('F','D'),
         ('G','F')]

print("Case 1 Cycle:", test_case(case1))
print("Case 2 Cycle:", test_case(case2))
print("Case 3 Cycle:", test_case(case3))