import heapq

def schedule_tasks(p, w, dependencies):
    n = len(p)

    # Build graph
    adj = [[] for _ in range(n)]
    indegree = [0]*n

    for u, v in dependencies:
        adj[u].append(v)
        indegree[v] += 1

    # Max heap using negative ratio
    pq = []
    for i in range(n):
        if indegree[i] == 0:
            heapq.heappush(pq, (-(w[i]/p[i]), i))

    time = 0
    total_cost = 0
    order = []

    while pq:
        _, u = heapq.heappop(pq)
        order.append(u+1)

        time += p[u]
        total_cost += w[u] * time

        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(pq, (-(w[v]/p[v]), v))

    return order, total_cost


# Example
p = [3,2,4,1,5]
w = [10,5,7,3,6]

dependencies = [
    (0,2),  # 1 → 3
    (1,3),  # 2 → 4
    (1,4)   # 2 → 5
]

order, cost = schedule_tasks(p, w, dependencies)

print("Order:", order)
print("Total Weighted Completion Time:", cost)