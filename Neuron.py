import numpy as np

class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights;
        self.bias = float(bias);
        self.inputs = None;
        self.output = None;
        self.error = None;


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        x = self.output;
        return x * (1 - x); # derivattive of the sigmoid function

    def relu(self,x):
        return np.maximum(0,x);

    def forward(self):
        sum_of_inputs = np.dot(self.weights, self.inputs) + self.bias; 
        
        self.output = self.sigmoid(sum_of_inputs)
        return self.output;

    def neuron(self, inputs, target_output, learning_rate, epochs):
        self.inputs = inputs;

        print(f"Initial weights: {self.weights}, Initial bias: {self.bias}");

        output = self.forward();

        self.error = self.loss_function(output, target_output);

        self.train_za_neron_boiii(inputs,target_output, learning_rate, epochs);

        print(f"Updated weights: {self.weights}, Updated bias: {self.bias}, Error: {self.error}");

        # first needs a propogation function that takes in the outputs (of various neurons) to create an input for the neuron
        # a bias term can be added to the input
        # needs an activiation function that needs: 1) activation variiable, 2) threshold variable, 3) input from predecessor
        # then it needs to send the results of the activation function into an output function
        return output

    def loss_function(self, predicted_output, actual_output):
        return self.output - actual_output; #ts a formula apparently

    def backpropagation(self, target_output, learning_rate): #learning rate is kinda like the PID; determins how big of a step to take

        learning_rate = 0.1 if learning_rate is None else learning_rate;

        error = self.error; #gets the error from loss function

        derivative_sigmoid = self.sigmoid_derivative(); #gets the derivative of the activation function
        delta = error * derivative_sigmoid;

        gradient_weights = delta * self.inputs; #calculates the gradient of the weights and bias based on the error and the derivative of the activation function

        gradient_bias = delta;

        self.weights -= learning_rate * gradient_weights; #updating the values of weights and bias based on the learning rate and the error values
        self.bias -= learning_rate * gradient_bias;


    def train_za_neron_boiii(self, X, y, learning_rate, epochs):
     
        for epoch in range(epochs): #funny enough its called epochs lol
            total_error = 0
            # Loop through every example in za dataset
            for inputs, target in zip(X, y):
               
                self.inputs = inputs


                self.forward()
                
                self.error = self.loss_function(self.output, target)
                
                self.backpropagation(target, learning_rate) #lets hope ts works
                
                total_error += abs(self.error) #accumulates the error to see how much the neuron is learning
            
            # asked ai to write ts :( - yes i was that lazy
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Cumulative Error: {total_error:.4f}")

        return self.weights, self.bias

    