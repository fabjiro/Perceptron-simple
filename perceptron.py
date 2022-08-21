import array
import random
import math

class Perceptron:

    def __init__(self, n_input:int, eta:float = 0.1):
        self.w:array = [(random.random() - random.random()) for _ in range(n_input)]
        self.umbral:float =  eta
    
    def activation(self,val:float) -> float: #funcion de activacion
        # return 1 / ( 1 + math.exp(-val)) #sigmoidea 
        return math.tan(-val) #tangent
        
    def predict(self, inputs: array) -> float:
        suma = self.umbral
        for (w, i) in zip(self.w, inputs):
            suma += i * w
        return self.activation(suma)

    def fit(self, epochs:int = 100, inputs:array = [], outputs:array = []):
        for _ in range(epochs):
            for (inp , out) in zip(inputs, outputs):
                prediction = self.predict(inp)
                error:float = prediction - out
                
                self.w = [(W + self.umbral * error * X) for(W,X) in zip(self.w, inp)]                    