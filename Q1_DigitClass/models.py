### code base: ai.berkeley.edu

import nn
from backend import DigitClassificationDataset


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 25        # batch size
        self.n_layers = 4           # Number of layers
        self.l_size = [784, 400, 300, 200]    # Size of layers
        self.num_y = 10             # Number of labels
        self.learning_rate = -0.15    # Learning rate
        self.learning_decay = 0.35    # Changes learning rate
        self.min_learning_rate = -0.001 # Minimum Learning rate

        # Initialize weights and bias parameters
        self.w = []
        self.b = []
        for i in range(1, self.n_layers):
            self.w.append(nn.Parameter(self.l_size[i-1], self.l_size[i]))
            self.b.append(nn.Parameter(1, self.l_size[i]))

        # Output Layer
        self.w.append(nn.Parameter(self.l_size[-1], self.num_y))
        self.b.append(nn.Parameter(1, self.num_y))

    def layer(self, x, w, b, output=False):
        """
        Computes a single layer of the neural network with the given 
        weights and bias.
        """
        x = nn.Linear(x, w)
        x = nn.AddBias(x, b)
        if output:
            return x
        return nn.ReLU(x)
    
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Iterate through layers to compute prediction except for the final layer
        for w, b in zip(self.w[:-1], self.b[:-1]):
            x = self.layer(x, w, b)

        return self.layer(x, self.w[-1], self.b[-1], output=True)
            
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Compute prediction
        h = self.run(x)
        # Calculate loss using SoftmaxxLoss function
        loss = nn.SoftmaxLoss(h, y)
        return loss

    def train(self, dataset: DigitClassificationDataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        k = 0
        while True:
            for row_vect, y in dataset.iterate_once(self.batch_size):
                # Calculate loss and gradient
                loss = self.get_loss(row_vect, y)
                gradients = nn.gradients(loss, (*self.w, *self.b))
                # Update Parameters
                for param, grad in zip((*self.w, *self.b), gradients):
                    param.update(grad, self.learning_rate)
            # Update learning rate after one epoch
            k += 1
            self.learning_rate = min(self.min_learning_rate, 
                                     self.learning_rate * self.learning_decay)
                                    #  self.learning_rate * (1/ (1 + self.learning_decay * k)))
            # Check for model accuracy
            # Return if accuracy > 98*=%
            if dataset.get_validation_accuracy() >= 0.98:
                return
