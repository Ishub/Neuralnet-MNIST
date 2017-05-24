import pandas 
import numpy as np

#Function to convert the file into csv 

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "mnist_train.csv", 60000)
convert("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte",
        "mnist_test.csv", 10000)

# filepath = "mnist_train.csv"
# dataset= pandas.read_csv(filepath)
# print dataset.shape
# (59999, 785)


#sigmoid to be used as activation function
def sigmoid(x):
		return 1/(1+ np.exp(-x))
	 	pass

# neural network class definition
class NeuralNetwork:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.layer0 = inputnodes
        self.layer1 = hiddennodes
        self.layer2 = outputnodes
        self.lr = learningrate
        # weights connecting l0 to l1 
        self.w1 = 2*np.random.random((self.layer1,self.layer0)) - 1
        
        # weights connecting l1 to l2
        self.w2 = 2*np.random.random((self.layer2,self.layer1)) - 1

        pass

    
    # train the simple neural network
    def train(self, pixel_list, label_list):
        # convert inputs list to 2d array

        pixels = np.array(pixel_list, ndmin=2).T
        labels = np.array(label_list, ndmin=2).T
    
        # values into layer 1
        l1_inputs = np.dot(self.w1, pixels)
        # values emerging from layer1 
        l1_outputs = sigmoid(l1_inputs)
        # values into layer2 
        l2_inputs = np.dot(self.w2, l1_outputs)
        # Values emerging from layer2
        l2_outputs = sigmoid(l2_inputs)
        
        # output layer error is the (target - actual)
        l2_errors = labels - l2_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        l1_errors = np.dot(self.w2.T, l2_errors)
        
        # update weights for hidden and output layers
        self.w2 += self.lr * np.dot((l2_errors * l2_outputs * (1.0 - l2_outputs)), np.transpose(l1_outputs))
        
        # update weights for input and hidden layers
        self.w1 += self.lr * np.dot((l1_errors * l1_outputs * (1.0 - l1_outputs)), np.transpose(pixels))
        
        pass

    
    # test the simple neural network
    def test(self, pixel_list):
        # inputs list to 2d array
        pixels = np.array(pixel_list, ndmin=2).T
        l1_inputs = np.dot(self.w1, pixels)
        l1_outputs = sigmoid(l1_inputs)
        l2_inputs = np.dot(self.w2, l1_outputs)
        l2_outputs = sigmoid(l2_inputs)
        
        return l2_outputs


# Values
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.02

# create instance of neural network
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Used only once to reduce the processing time for temporary porpose. 
epochs = 1

for e in range(epochs):

	for record in training_data_list:   
	    all_values = record.split(',')

	    # Normalisation of pixels 0 - 255 to range of 0 to 1
	    pixels = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

	    # the label output values (all 0.01, except the desired label which is 0.99)
	    labels = np.zeros(output_nodes) + 0.01
	    labels[int(all_values[0])] = 0.99
	    n.train(pixels, labels)
	        
	    pass

	print ("Training Done on {0} Epoch").format(e)
	pass

	
# ---------------------------------------- 4. Checking the accuracy of the Network -----------------------------------------------------

# loading the mnist_test csv file
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#if predicted right accuracy = 1 if not accuracy = 0
accuracy = []

	
for record in test_data_list:

	all_values = record.split(',')
	correct_label = int(all_values[0])

	# Normalisation of pixels to range 0 to 1
	inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

	# run the simple neural network on test data 
	outputs = n.test(inputs)
	label = np.argmax(outputs)

	if (label == correct_label):

	    accuracy.append(1)
	else:

	    accuracy.append(0)
	    pass
	    
	pass
	
accuracy_array = np.asarray(accuracy)
correct= float(accuracy_array.sum())
total = float(accuracy_array.size)
print "total number of correct predictions", correct
print "Total predictions", total
per = (correct/total)
print "performance = ", per

