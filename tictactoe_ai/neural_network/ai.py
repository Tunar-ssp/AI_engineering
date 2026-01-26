import numpy as np

class model:
    def __init__(self, input_size=9, hidden1=128, hidden2=64, output_size=9):
        
        self.initialize(input_size, hidden1, hidden2, output_size)
    
    def initialize(self,input_size, hidden1, hidden2, output_size):
        self.W1=np.random.randn(input_size,hidden1)*0.01
        self.b1=np.zeros(hidden1)
        
        self.W2=np.random.randn(hidden1,hidden2)*0.01
        self.b2=np.zeros(hidden2)
        
        self.W3=np.random.randn(hidden2,output_size)*0.01
        self.b3=np.zeros(output_size)

    def ReLU(self,z):
        return np.maximum(0,z)
    
    def calculate_Q_target(self,reward,Q_nextstate,gamma=0.9):
        return reward+gamma* np.max(Q_nextstate)
        
    def forward_propagation(self,training_data):

        if training_data.ndim == 1:
            self.training_data = training_data.reshape(1, -1)
        else:
             self.training_data = training_data
    
        
        self.Z1=self.training_data@self.W1+self.b1
        self.A1=self.ReLU(self.Z1)
        
        self.Z2=self.A1@self.W2+self.b2
        self.A2=self.ReLU(self.Z2)
    
        self.Z3=self.A2@self.W3+self.b3
        self.A3=self.Z3

        

        return self.A3


        
    def gradient_descent(self, loss, l2_reg=0.001):
            m = self.training_data.shape[0]
            
            # Output Layer (Layer 3)
            dZ3 = loss 
            self.dzW3 = (self.A2.T @ dZ3) + l2_reg * self.W3
            self.dzb3 = np.sum(dZ3, axis=0, keepdims=True)

            # Hidden Layer 2
            dA2 = dZ3 @ self.W3.T
            # Derivative of ReLU 
            dZ2 = dA2 * np.where(self.Z2 > 0, 1, 0) 
            self.dzW2 = (self.A1.T @ dZ2) + l2_reg * self.W2
            self.dzb2 = np.sum(dZ2, axis=0, keepdims=True)

            # Hidden Layer 1
            dA1 = dZ2 @ self.W2.T
            # Derivative of ReLU
            dZ1 = dA1 * np.where(self.Z1 > 0, 1, 0) 
            self.dzW1 = (self.training_data.T @ dZ1) + l2_reg * self.W1
            self.dzb1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # clip gradients to prevent explosion
            self.dzW1 = np.clip(self.dzW1, -1.0, 1.0)
            self.dzW2 = np.clip(self.dzW2, -1.0, 1.0)
            self.dzW3 = np.clip(self.dzW3, -1.0, 1.0)
            self.dzb1 = np.clip(self.dzb1, -1.0, 1.0)
            self.dzb2 = np.clip(self.dzb2, -1.0, 1.0)
            self.dzb3 = np.clip(self.dzb3, -1.0, 1.0)

    def backward_propagation(self,learning_rate):
        self.W1-=self.dzW1*learning_rate
        self.dzb1=self.dzb1.squeeze() 
        self.b1-=self.dzb1*learning_rate

        self.W2-=self.dzW2*learning_rate
        self.dzb2=self.dzb2.squeeze()
        self.b2-=self.dzb2*learning_rate
        
        self.W3-=self.dzW3*learning_rate
        self.dzb3=self.dzb3.squeeze()
        self.b3-=self.dzb3*learning_rate

    def save_model(self, file_name="models/model.npz"):
        np.savez(file_name, 
                W1=self.W1, b1=self.b1, 
                W2=self.W2, b2=self.b2, 
                W3=self.W3, b3=self.b3)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name="models/model.npz"):
        data = np.load(file_name)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        print(f"Model loaded from {file_name}")        