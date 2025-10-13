import numpy as np

# for single neuron with 4 inputs
inputs=[1,2,3,2.5]
Weights=[0.2,0.8,-0.5,1.0]
bias=2.0
output =np.dot(Weights,inputs)+bias
#print(output)

#for multiple neurons
inputs=[1,2,3,2.5]
weights=[
    [0.2,0.8,-0.5,1.0],
    [0.5,-0.91,0.26,-0.5],
    [-0.26,-0.27,0.17,0.87]
    ]
biases=[2,3,0.5]
output=np.dot(weights,inputs)+biases
print(output)
