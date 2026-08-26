from value_class import Value
from neuron import o
from graphiz import draw_dot

frontier = []
visited = set()
def backward(L):
    if(L == None):
        return
    if L not in visited:
        visited.add(L)

        childs = L._prev

        for child in childs:
            backward(child)

        frontier.append(L)
o.grad = 1
backward(o)

for n in reversed(frontier):
    n._backward()

graph = draw_dot(o)
graph.view()