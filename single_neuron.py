import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd


# Activation Functions
def heaviside(s):
    return np.where(s >= 0, 1, 0)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def sigmoid_derivative(s):
    return sigmoid(s) * (1 - sigmoid(s))

def sin(s):
    return np.sin(s)

def sin_derivative(s):
    return np.cos(s)

def tanh(s):
    return np.tanh(s)

def tanh_derivative(s):
    return 1 - np.tanh(s)**2

def sign(s):
    return np.sign(s)

def relu(s):
    return np.where(s > 0, s, 0.0)

def relu_derivative(s):
    return np.where(s > 0, 1.0, 0.0)

def leaky_relu(s):
    return np.where(s > 0, s, 0.01 * s)

def leaky_relu_derivative(s):
    return np.where(s > 0, 1.0, 0.01)


# Neuron
class Neuron:
    def __init__(self, num_of_inputs, activation="heaviside", learning_rate=0.1):
        self.weights = np.random.randn(num_of_inputs)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.activation =  activation

    def activate(self, s):
        if self.activation == 'heaviside':
            return heaviside(s)
        elif self.activation == 'sigmoid':
            return sigmoid(s)
        elif self.activation == 'sin':
            return sin(s)
        elif self.activation == 'tanh':
            return tanh(s)
        elif self.activation == 'sign':
            return sign(s)
        elif self.activation == 'relu':
            return relu(s)
        elif self.activation == 'leaky relu':
            return leaky_relu(s)
    
    def derivative(self, s):
        if self.activation == 'heaviside':
            return 1
        elif self.activation == 'sigmoid':
            return sigmoid_derivative(s)
        elif self.activation == 'sin':
            return sin_derivative(s)
        elif self.activation == 'tanh':
            return tanh_derivative(s)
        elif self.activation == 'sign':
            return 1
        elif self.activation == 'relu':
            return relu_derivative(s)
        elif self.activation == 'leaky relu':
            return leaky_relu_derivative(s)
    
    def predict(self, x):
        weighted_sum = np.dot(self.weights, x) + self.bias
        return self.activate(weighted_sum)  

    def train(self, X, y):
        for epoch in range(100):
            for x_i, target in zip(X, y):
                output = self.predict(x_i)
                error = target - output
                self.weights += self.learning_rate * error * self.derivative(output) * x_i
                self.bias += self.learning_rate * error * self.derivative(output)

    

# Generate Gaussian Data
def generate_gaussian_mode(mean, variance, num_of_samples):
    cov_matirx = [[variance, 0], [0, variance]]
    samples = np.random.multivariate_normal(mean, cov_matirx, num_of_samples)
    return samples

def generate_class_data(num_of_modes, samples_per_mode):
    all_samples = []
    for _ in range(num_of_modes):
        mean = np.random.uniform(-1, 1, 2) # center
        variance = np.random.uniform(0.05, 0.2) # spread
        samples = generate_gaussian_mode(mean, variance, samples_per_mode)
        all_samples.append(samples)
    return np.vstack(all_samples)


# Main
def main():
    st.title("Single Neuron")

    
    st.sidebar.header("Settings")
    num_of_modes_class_0 = st.sidebar.number_input("Number of Modes for Class 0", min_value=1, max_value=10, value=1)
    num_of_modes_class_1 = st.sidebar.number_input("Number of Modes for Class 1", min_value=1, max_value=10, value=1)
    samples_per_mode = st.sidebar.number_input("Number of Samples per Mode", min_value=10, max_value=1000, value=100)
    activation_function = st.sidebar.selectbox("Activation Function", ["heaviside", "sigmoid", "sin", "tanh", "sign", "relu", "leaky relu"])
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1)

    if st.sidebar.button("Generate Data and Train Neuron"):
        class_0_data = generate_class_data(num_of_modes_class_0, samples_per_mode)
        class_1_data = generate_class_data(num_of_modes_class_1, samples_per_mode)

        class_0_labels = np.zeros(len(class_0_data))
        class_1_labels = np.ones(len(class_1_data))

        data = np.vstack([class_0_data, class_1_data])
        labels = np.hstack([class_0_labels, class_1_labels])

        neuron = Neuron(num_of_inputs=2, activation=activation_function, learning_rate=learning_rate)
        neuron.train(data, labels)

        class_0_df = {"x": class_0_data[:, 0], "y": class_0_data[:, 1], "class": "Class 0"}
        class_1_df = {"x": class_1_data[:, 0], "y": class_1_data[:, 1], "class": "Class 1"}
        data_df = pd.DataFrame(class_0_df)._append(pd.DataFrame(class_1_df))

        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        predictions = np.array([neuron.predict(xi) for xi in grid])
        zz = predictions.reshape(xx.shape)

        fig = px.scatter(data_df, x="x", y="y", color="class", title="Generated Gaussian Data and Neuron Decision Boundary")
        fig.add_contour(x=xx[0], y=yy[:,0], z=zz, colorscale="Blues", opacity=0.3)

        st.plotly_chart(fig)




if __name__ == '__main__':
    main()

