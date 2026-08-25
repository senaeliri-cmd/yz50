from graphiz import draw_dot

class Value():
    def __init__(self, data, _op= "", _children=(), _label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._label = _label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, _op="+", _children=(self, other))
        

    def __mul__(self,other):
        return Value(self.data * other.data, _op="*", _children=(self, other))

a = Value(2, _label="a")
b = Value(-3, _label="b")
d = Value(5, _label="d")
g = Value(4, _label="g")
c = a*b
c._label="c"
e = c*d
e._label="e"
f = e + g
f._label = "f"

f.grad = 1.00
a.grad = -15.00
b.grad = 10.00
c.grad = 5.00
d.grad = -6.00
g.grad = 1.00
e.grad = 1.00
dot = draw_dot(f)
dot.view("graph")

def test():
    h = 0.001
    
    a = Value(2, _label="a")
    b = Value(-3, _label="b")
    d = Value(5, _label="d")
    g = Value(4, _label="g")
    c = a*b
    c._label="c"
    e = c*d
    e._label="e"
    f = e + g
    f._label = "f"

    L1 = f.data

    a = Value(2 , _label="a")
    b = Value(-3, _label="b")
    d = Value(5, _label="d")
    g = Value(4, _label="g")
    c = a*b
    #c.data += h
    c._label="c"
    e = c*d
    e.data += h
    e._label="e"
    f = e + g
    f._label = "f"

    L2 = f.data

    #print((L2-L1)/h)

