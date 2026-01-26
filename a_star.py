import math

class Node: 
    def __init__(self, name, x, y):
        self.name = name 
        self.x = x
        self.y = y
        self.g = float('infinity')
        self.h = 0 
        self.f = float('infinity')
        self.parent = None
        self.neighbours = []

        def add_neighbour(self, neighbour, cost):
            self.neighbours.append((neighbour, cost))

        def manhattan_distance(self, goal):
            return abs(self.x - goal.x) + abs(self.y - goal.y)

def a_star(start, goal):
    open_list = []
    closed_list = []

    start.g = 0
    start.h = start.manhattan_distance(goal)
    start.f = start.g + start.h

    open_list.append(start)

    while open_list:
        current = open_list[0]

        for node in open_list:
            if node.f < current.f:
                current = node

        if current == goal:
            return reconstruct_path(goal)

        open_list.remove(current)
        closed_list.append(current)

        for neighbour, cost in current.neighbours:
            if neighbour in closed_list:
                continue

        tentative_g = current.g + cost

        if neighbour not in open_list:
            open_list.append(neighbour)
    
        elif tentative_g >= neighbour.g:
            continue

        neighbour.parent = current
        
        neighbour.g = tentative_g
        neighbour.h = neighbour.manhattan_distance(goal)
        neighbour.f = neighbout.g + neighbout.h

    return None

def reconstruct_path(goal):
    path = []
    current = goal

    while current: 
        path.append(current.name)
        current = current.parent

    path.reverse()
    return path


A = Node("A", 0, 0)
B = Node("B", 1, 0)
C = Node("C", 1, 1)
D = Node("D", 2, 1)
E = Node("E", 3, 1)

A.add_neighbour(B, 1)
A.add_neighbour(C, 2)

B.add_neighbour(C, 1)
B.add_neighbour(D, 3)

C.add_neighbour(D, 1)
D.add_neighbour(E, 1)

path = a_star(A,E)
print(f"Shortest path is : {path}")