from value_class import Value
from graphiz import draw_dot
import math

w1 = Value(2.34, _label= "w1")
w2 = Value(-1.12, _label= "w2")
#w3 = Value(1.12, _label= "w3")

a1 = Value(0.89, _label = "a1")
a2 = Value(2.12, _label = "a2")
#a3 = Value(0.89, _label = "a3")

b = Value(1.02, _label = "bias")

a1w1 = a1*w1
a1w1._label="a1w1"

a2w2 = a2*w2
a2w2._label = "a2w2"

#a3w3 = a3*w3
#a3w3._label = "a3w3"

a1w1a2w2 = a1w1 + a2w2
a1w1a2w2._label = "a1w1a2w2"

n = a1w1a2w2 + b ; n._label = "n"
o = n.tanh(); o._label = "o"

# Görev ikide kullanılan manuel gradientlar
#o.grad = 1
#n.grad = 0.613
#a1w1a2w2.grad = 0.613
#b.grad = 0.613
#a2w2.grad = 0.613
#a1w1.grad = 0.613
#a2.grad = w2.data * a2w2.grad
#w2.grad = a2.data * a2w2.grad
#a1.grad = w1.data * a1w1.grad
#w1.grad = a1.data * a1w1.grad





graph = draw_dot(o)
graph.view()