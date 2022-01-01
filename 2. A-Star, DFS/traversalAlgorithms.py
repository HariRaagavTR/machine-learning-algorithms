import heapq as pq

def children(node, cost):
    for child in range(len(cost) - 1, 0, -1):
        if child != node and cost[node][child] != -1:
            yield child

def A_star_Traversal(cost, heuristic, start_point, goals):    
    visited = []
    frontier = []
    pq.heappush(frontier, (heuristic[start_point], [start_point]))
    while frontier:
        t_cost, path = pq.heappop(frontier)
        cur_node = path[-1]
        if cur_node in goals:
            return path
        cur_cost = t_cost - heuristic[cur_node]
        visited.append(cur_node)
        for child in children(cur_node, cost):
            if child not in visited:
                pq.heappush(frontier, (cur_cost + cost[cur_node][child] + heuristic[child], path + [child]))
    return []


def DFS_Traversal(cost, start_point, goals):
    visited = []
    stack = [(start_point, [start_point])]
    while stack:
        cur_node, path = stack.pop()
        if cur_node in goals:
            return path
        visited.append(cur_node)
        for child in children(cur_node, cost):
            if child not in visited:
                stack.append((child, path + [child]))
    return []

