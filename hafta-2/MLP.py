from value_class import Value
import random
class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))
    def __call__(self, activations):
        wx= zip(self.weights, activations)
        sum = Value(0.0)
        for w, x in wx:
            sum += w * x
        sum += self.bias
        out = sum.tanh()
        return  out
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:
    def __init__(self, nin, outin):
        self.neurons = [Neuron(nin) for i in range(outin)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, outin):
        all_layers = [nin] + outin
        self.layers = [Layer(all_layers[i], all_layers[i+1]) for i in range(len(outin))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
#print(n(x))
#print(len(n.parameters()))
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ydesired = [1.0, -1.0, -1.0, 1.0]
yguessed = [n(x) for x in xs]

for _ in range(100):
    for p in n.parameters():
        p.grad = 0
    yguessed = [n(x) for x in xs]
    loss = sum((y1 - y2)**2 for y1, y2 in zip(ydesired, yguessed))
    loss.backward()
    for p in n.parameters():
        p.data += p.grad * -0.01
    print(loss)
print(yguessed)



    
