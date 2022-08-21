from perceptron import Perceptron 

# data
inputs = [
    [0,1], #1
    [1,0], #1
    [0,0]  #0
]

outputs= [
    1,
    1,
    0
]
neurona = Perceptron(2, 0.1)

def test():
    for item in inputs:
        if(neurona.predict(item) > 0.5):
            print('1')
        else:
            print('0')
        # neurona.predict(item)

neurona.fit(inputs=inputs, outputs=outputs, epochs=50)

test()