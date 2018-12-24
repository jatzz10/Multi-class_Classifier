# Multi-class classifier model for classification of Iris Data set 
# Highest accuracy achieved = 70.67%

import random
import math

class NeuralNet(object):
    def __init__(self):
        
        random.seed(1)

        # Neural Network
        # input layer : 4 neurons, represents the feature of Iris
        # hidden layer : 4 neurons, activation using sigmoid
        # output layer : 3 neurons, represents the class of Iris
        
        self.neuron = [4, 4, 3]   # number of neurons in each layer

        # Initialise synaptic weights of layer 1 and layer 2 with 0 value
        self.synaptic_weights_l1 = []
        for i in range(self.neuron[0]):
            self.synaptic_weights_l1.append([])
            for j in range(self.neuron[1]):
                self.synaptic_weights_l1[i].append(0)

        self.synaptic_weights_l2 = []
        for i in range(self.neuron[1]):
            self.synaptic_weights_l2.append([])
            for j in range(self.neuron[2]):
                self.synaptic_weights_l2[i].append(0)
         
        self.bias_l1 = []
        for i in range(self.neuron[1]):
            self.bias_l1.append(0)

        self.bias_l2 = []
        for i in range(self.neuron[2]):
            self.bias_l2.append(0)


    # Matrix multiplication (for Testing)
    def matrix_multiply(self, A, B, bias): 
        C = [[0 for i in range(len(B[0]))] for i in range(len(A))]    
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    C[i][j] += A[i][k] * B[k][j]
                C[i][j] += bias[j]
        return C


    # Vector (A) x matrix (B) multiplication
    def vector_X_matrix(self, A, B, bias):
        C = [0 for i in range(len(B[0]))]
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[j] += A[k] * B[k][j]
                C[j] += bias[j]
        return C


    # Matrix (A) x vector (B) multipilicatoin (for backprop)
    def matrix_X_vector(self, A, B): 
        C = [0 for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B)):
                C[i] += A[i][j] * B[j]
        return C


    # Sigmoid used as Activation function
    def sigmoid(self, A):
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
        return A

    # Build and train the model
    def train(self, train_X, train_Y, test_X, test_Y, epoch, learning_rate):

        # Layer 0 = Input layer
        # Initialise synaptic_weights of layer 1 and layer 2 with random between -1.0 ... 1.0

        for i in range(self.neuron[0]):
            for j in range(self.neuron[1]):
                self.synaptic_weights_l1[i][j] = 2 * random.random() - 1

        for i in range(self.neuron[1]):
            for j in range(self.neuron[2]):
                self.synaptic_weights_l2[i][j] = 2 * random.random() - 1

        for i in range(self.neuron[1]):
            self.bias_l1[i] = 0

        for i in range(self.neuron[2]):
            self.bias_l2[i] = 0

        print "RMS_Error ->"

        for e in range(epoch):
            cost_total = 0

            for idx, data in enumerate(train_X): 
    
                ############## Forward propagation ##############

                layer_1 = self.vector_X_matrix(data, self.synaptic_weights_l1, self.bias_l1)
                layer_1 = self.sigmoid(layer_1)
                layer_2 = self.vector_X_matrix(layer_1, self.synaptic_weights_l2, self.bias_l2)
                layer_2 = self.sigmoid(layer_2)
                
                # Hot-target
                target = [0, 0, 0]
                target[int(train_Y[idx])] = 1

                # Cost function => Root Mean Square Error
                error = 0
                for i in range(3):
                    error +=  (target[i] - layer_2[i]) ** 2 
                cost_total += error

                ############## Backward propagation ##############

                # Update synaptic_weights_l2 and bias_l2 (layer 2)
                delta_2 = []
                for j in range(self.neuron[2]):
                    delta_2.append((target[j]-layer_2[j]) * layer_2[j] * (1-layer_2[j]))   # layer_2[j] * (1-layer_2[j]) --> Sigmid Derivative (Gradient Descent)

                for i in range(self.neuron[1]):
                    for j in range(self.neuron[2]):
                        self.synaptic_weights_l2[i][j] += learning_rate * (delta_2[j] * layer_1[i])
                        self.bias_l2[j] += learning_rate * delta_2[j]
        
                # Update synaptic_weights_l1 and bias_l1 (layer 1)
                delta_1 = self.matrix_X_vector(self.synaptic_weights_l2, delta_2)
                for j in range(self.neuron[1]):
                    delta_1[j] = delta_1[j] * (layer_1[j] * (1-layer_1[j]))
        
                for i in range(self.neuron[0]):
                    for j in range(self.neuron[1]):
                        self.synaptic_weights_l1[i][j] +=  learning_rate * (delta_1[j] * data[i])
                        self.bias_l1[j] += learning_rate * delta_1[j]
    
            cost_total /= len(train_X)
            cost_total = math.sqrt(cost_total)

            if(e % 100 == 0):
                print cost_total

    def load(self, file):
        contents = open(file).read()
        return [item.split(',') for item in contents.split('\n')[:-1]]
        


if __name__ == "__main__":

    neural_network = NeuralNet()

    # Import the data from iris.csv file into dataset
    dataset = neural_network.load('iris.csv')

    # Convert all the values to float and outputs in 0, 1 and 2
    for i in range(len(dataset)):
        if dataset[i][4] == "Iris-setosa":
            dataset[i][4] = 0
        elif dataset[i][4] == "Iris-versicolor":
            dataset[i][4] = 1
        else:
            dataset[i][4] = 2
        for j in range(len(dataset[i])-1):
            dataset[i][j] = float(dataset[i][j])

    # Generalise the data set
    random.shuffle(dataset)

    # Applying K-fold Cross Validation on the model
    # Taking K = 5 for the data set consisting of 150 instances
    # Each fold = 30 instances for test data set
    
    K = 5
    Resultant = 0
    
    # Sliding Window for folds 
    w1 = 0
    w2 = 30

    for folds in range(K):
        
        # Splitting the data set into two parts
        # 80% of data set for training the model = 120 instances
        # remaining 20% for testing the model = 30 instances

        # Splitting of dataset into X and Y 
        # X part denotes the features(4 input parameters) for iris dataset
        # Y part denotes the label(output) for each set of input parameters

        train_X = []
        train_Y = []
        test_X = []
        test_Y = []
        train_idx = -1
        test_idx = -1

        for i in range(len(dataset)):
            if i >= w1 and i < w2:
                test_X.append([])
                test_idx = test_idx + 1
            else:
                train_X.append([])
                train_idx = train_idx + 1
            for j in range(len(dataset[i])):
                if i >= w1 and i < w2:
                    if j < 4:
                        test_X[test_idx].append(dataset[i][j])
                    else:
                        test_Y.append(dataset[i][j])
                else:
                    if j < 4:
                        train_X[train_idx].append(dataset[i][j])
                    else:
                        train_Y.append(dataset[i][j])

        w1 = w1 + 30
        w2 = w2 + 30
    
        learning_rate = 0.005
        epoch = 500   # epoch = "No. of iterations for training the model"

        # Train the model
        neural_network.train(train_X, train_Y, test_X, test_Y, epoch, learning_rate)

        # Forward propagate on the test data 
        layer1_res = neural_network.matrix_multiply(test_X, neural_network.synaptic_weights_l1, neural_network.bias_l1)
        layer2_res = neural_network.matrix_multiply(layer1_res, neural_network.synaptic_weights_l2, neural_network.bias_l2)

        # Get predictions
        predict = []
        for i in range(len(layer2_res)):
            index = 0
            maxm = layer2_res[i][0]
            if layer2_res[i][1] > maxm:
                maxm = layer2_res[i][1]
                index = 1
            if layer2_res[i][2] > maxm:
                maxm = layer2_res[i][2]
                index = 2
            predict.append(index)

        # Calculate accuracy
        acc = 0.0
        for i in range(len(predict)):
            if predict[i] == int(test_Y[i]):
                acc += 1

        print "\nAccuracy =", acc / len(predict) * 100, "%\n\n"

        Resultant += acc / len(predict) * 100

    Resultant /= K
    print "Resultant Accuracy = ", Resultant, "%"









