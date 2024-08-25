from perceptron import Perceptron 
import math
# data
inputs = [
    [0,1], #0
    [1,0], #0
    [0,0], #0
    [1,1]  #1
]

outputs= [
    0,
    0,
    0,
    1
]
neurona = Perceptron(2, 0.1)

def test():
    for item in inputs:
        print('input: ', item, 'output: ', round(neurona.predict(item)), 'expected: ', outputs[inputs.index(item)])
        # neurona.predict(item)

neurona.fit(inputs=inputs, outputs=outputs, epochs=10)

test()