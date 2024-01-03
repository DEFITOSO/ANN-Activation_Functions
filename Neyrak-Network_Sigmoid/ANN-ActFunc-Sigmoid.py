import numpy as np

def sigmoid(x):
  sig_x=1/(1+np.exp(-x))
  return sig_x

def perceptron(x, w, b):
        """ Function implemented by a perceptron with weight vecto w and bias """
        v=np.dot(w,x)+b
        y=sigmoid(v)
        return y


def NOT_percep(x):
      return perceptron(x, w=-1, b=0.5)

  # Test
print("NOT(0) = {}".format(NOT_percep(0)))
print("NOT(1) = {}".format(NOT_percep(1)))
print("NOT(3) = {}".format(NOT_percep(3)))
print("NOT(10) = {}".format(NOT_percep(10)))
print("NOT(15) = {}".format(NOT_percep(15)))
print("NOT(-10) = {}".format(NOT_percep(-10)))

def AND_percep(x):
  w = np.array([1, 1])
  b = -1.5
  return perceptron(x,w,b)

  # Test
example1 = np.array([0.47,0.49])
example2 = np.array([0.2,0.4])
example3 = np.array([0.099,0.499])
example4 = np.array([0.1,0.9])

print("AND({}, {}) = {}".format(0.49,0.51,AND_percep(example1)))
print("AND({}, {}) = {}".format(0.4,0.6,AND_percep(example2)))
print("AND({}, {}) = {}".format(0.3,0.7,AND_percep(example3)))
print("AND({}, {}) = {}".format(0.1,0.9,AND_percep(example4)))

def OR_percep(x):
    w = np.array([1,1])
    b = -0.5
    return perceptron(x, w, b)

#Test
example1 = np.array([1,0.9])
example2 = np.array([0.51,0.5])
example3 = np.array([0,100])
example4 = np.array([0.49,0.5])

print("OR({}, {}) = {}".format(100,10,OR_percep(example1)))
print("OR({}, {}) = {}".format(0.51,0.5,OR_percep(example2)))
print("OR({}, {}) = {}".format(0,100,OR_percep(example3)))
print("OR({}, {}) = {}".format(0.49,0.5,OR_percep(example4)))



def XOR_percep(x):
  output_AND = AND_percep(x)
  output_NOT = NOT_percep(output_AND)
  output_OR = OR_percep(x)
  x_temp = np.array([output_NOT, output_OR])
  output_AND = AND_percep(x_temp)
  return output_AND

#Test
example1 = np.array([9,9])
example2 = np.array([0.5,50])
example3 = np.array([0.5,0.5])
example4 = np.array([10,0])

print("XOR({}, {}) = {}".format(5,1,XOR_percep(example1)))
print("XOR({}, {}) = {}".format(0.5,0,XOR_percep(example2)))
print("XOR({}, {}) = {}".format(0.5,0.5,XOR_percep(example3)))
print("XOR({}, {}) = {}".format(10,0,XOR_percep(example4)))

def XNOR_percep(x):
  output_AND = AND_percep(x)
  output_NOT1 = NOT_percep(x[0])
  output_NOT2 = NOT_percep(x[1])
  output_NOT_AND = AND_percep(np.array([output_NOT1, output_NOT2]))
  x_temp = np.array([output_AND, output_NOT_AND])
  output_OR = OR_percep(x_temp)
  return output_OR


#Test
example1 = np.array([1,1])
example2 = np.array([1,0])
example3 = np.array([0,1])
example4 = np.array([0,0])

print("XNOR({}, {}) = {}".format(0.6,0.5,XNOR_percep(example1)))
print("XNOR({}, {}) = {}".format(0.75,0.7,XNOR_percep(example2)))
print("XNOR({}, {}) = {}".format(0.55,0.525,XNOR_percep(example3)))
print("XNOR({}, {}) = {}".format(0.85,0.8,XNOR_percep(example4)))