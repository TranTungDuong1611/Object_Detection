import numpy as np

def one_hot_encode(y, num_classes):
        """
        Convert an array of class labels (shape: (n_samples,)) into one-hot encoded format (shape: (n_samples, num_classes)).
        """
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y.astype(int)] = 1
        return one_hot

class NeuralNetwork:
    def __init__(self, X, y, region_of_interest, hidden_layer, num_classes, epochs=100, learning_rate=0.0001):
        self.X = X
        self.y = y  # Now y will have a size corresponding to the number of proposals in X
        self.region_of_interest = region_of_interest  # New variable for storing ROI box coordinates
        self.hidden_layer = hidden_layer
        self.num_classes = num_classes
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Initialize weights and biases
        layer_sizes = [X.shape[1]] + list(hidden_layer) + [num_classes * 5]  # 1 class + 4 bounding box offsets
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(2 / (layer_sizes[i] + layer_sizes[i + 1]))
            weight_matrix = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            self.weights.append(weight_matrix)

            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            self.biases.append(bias_vector)

        self.node_values = [np.zeros(n) for n in hidden_layer]

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

    def feedforward(self):
        A = self.X  # Input activations
        activations = [A]  # Store activations for each layer
        z = [A]

        # Loop through hidden layers
        for i, W in enumerate(self.weights[:-1]):
            Z = np.dot(A, W) + self.biases[i]
            z.append(Z)
            A = self.relu(Z)
            activations.append(A)

        # Output layer
        Z_output = np.dot(A, self.weights[-1]) + self.biases[-1]
        z.append(Z_output)
        # Classification (use sigmoid) and regression (linear)
        classification_output = self.sigmoid(Z_output[:, -self.num_classes:])
        regression_output = Z_output[:, :-self.num_classes]

        activations.append(Z_output)

        return classification_output, regression_output, activations, z

    def one_hot_encode(y, num_classes):
        """
        Convert an array of class labels (shape: (n_samples,)) into one-hot encoded format (shape: (n_samples, num_classes)).
        """
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y.astype(int)] = 1
        return one_hot

    def compute_cross_entropy_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss. 
        y_true: one-hot encoded true class labels, shape: (n_samples, num_classes)
        y_pred: predicted class probabilities, shape: (n_samples, num_classes)
        """
        epsilon = 1e-9  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Prevent log(0) error
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]  # Normalize by number of samples

    def compute_mse_loss(self, y_true, y_pred, y_class):
        """
        Compute mean squared error loss for bounding box regression.
        y_true: Ground-truth bounding box values (shape: (n_samples, 4)).
        y_pred: Predicted bounding box values for all classes (shape: (n_samples, 12) if 3 classes).
        y_class: True class labels (shape: (n_samples,)) to select the correct bounding box for comparison.
        """
        num_classes = y_pred.shape[1] // 4  # 3 classes (if shape is 200, 12 -> each class has 4 values)
    
        # Reshape y_pred to (n_samples, num_classes, 4)
        y_pred_reshaped = y_pred.reshape(-1, num_classes, 4)
    
        # Select the correct bounding box predictions based on the true class labels
        # np.arange(y_class.size) gives [0, 1, 2, ..., n_samples-1]
        y_pred_selected = y_pred_reshaped[np.arange(y_class.size), y_class.astype(int)]
        # Now compute the mean squared error between the selected predictions and the true bounding boxes
        return np.mean(np.square(y_pred_selected - y_true))

    def compute_bounding_box_error(self, pred_offsets):
        """
        Compute bounding box errors for regression using mean square error.
        The output shape will be (num_proposals, num_classes * 4), 
        with zeros for incorrect class nodes.
        """
        # Initialize the bounding box errors array with zeros
        bbox_errors = np.zeros((self.region_of_interest.shape[0], self.num_classes * 4))  # Shape: (number_of_proposals, num_classes * 4)

        # Reshape pred_offsets to (number_of_proposals, num_classes, 4)
        reshaped_offsets = pred_offsets.reshape(-1, self.num_classes, 4)  # Shape: (number_of_proposals, num_classes, 4)

        for i in range(reshaped_offsets.shape[0]):  # For each proposal
            true_class_index = int(self.y[i, 0])  # Get the true class label (assuming it's the first element)

            if true_class_index < self.num_classes:  # Check if the class index is valid
                true_bbox = self.y[i, 1:]  # Get the true bounding box values (xmin, ymin, xmax, ymax)
            
                # Calculate the predicted bounding box using the region of interest and predicted offsets
                predicted_bbox = self.region_of_interest[i] + reshaped_offsets[i, true_class_index]
            
                # Calculate the error (predicted - true)
                # Store the error in the correct class position
                bbox_errors[i, true_class_index * 4: true_class_index * 4 + 4] = predicted_bbox - true_bbox  # Shape: (4,)

        return bbox_errors

    def compute_cross_entropy_derivative(self, y_true, y_pred):
        """
        Compute the derivative of the cross-entropy loss with respect to the predicted probabilities (y_pred).
        y_true: One-hot encoded true class labels, shape: (n_samples, num_classes)
        y_pred: Predicted class probabilities (after softmax), shape: (n_samples, num_classes)
        """
        epsilon = 1e-12  # A small value to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid 0 or 1
        return (y_pred - y_true) / y_true.shape[0]

    def compute_mse_derivative(self, y_true, y_pred, y_class):
        """
        Compute the derivative of the mean squared error loss for bounding box regression.
        y_true: Ground-truth bounding box values (shape: (h, 4)).
        y_pred: Predicted bounding box values for all classes (shape: (h, 4 * c)).
        y_class: True class labels (shape: (h,)) to select the correct bounding box for comparison.
    
        Output: (h, 4 * c) matrix, where MSE for the true class is computed, and others are set to zero.
        """
        h = y_true.shape[0]  # Number of proposals (regions)
        c = y_pred.shape[1] // 4  # Number of classes

        # Reshape predictions to be (h, c, 4) where each proposal has predictions for all classes
        y_pred_reshaped = y_pred.reshape(h, c, 4)
    
        # Initialize the output error matrix with zeros (h, 4 * c)
        mse_derivative = np.zeros_like(y_pred)
    
        # Iterate over each region proposal
        for i in range(h):
            # Select the predicted bounding box for the true class of the current region
            pred_bbox = y_pred_reshaped[i, int(y_class[i])]  # Shape: (4,)
            
            # Calculate the MSE derivative for the true class
            true_bbox = y_true[i]  # Assuming region_of_interest stores (h, 4) coordinates
            mse = 2 *(pred_bbox - true_bbox)
            # Assign the MSE derivative to the corresponding class columns
            mse_derivative[i, int(y_class[i]) * 4: (int(y_class[i]) + 1) * 4] = mse
    
        return mse_derivative / y_true.shape[0]

    def compute_smooth_l1_derivative(self, y_true, y_pred, y_class):
        """
    Compute the derivative of the Smooth L1 loss for bounding box regression.
    y_true: Ground-truth bounding box values (shape: (h, 4)).
    y_pred: Predicted bounding box values for all classes (shape: (h, 4 * c)).
    y_class: True class labels (shape: (h,)) to select the correct bounding box for comparison.
    
    Output: (h, 4 * c) matrix, where Smooth L1 loss for the true class is computed, and others are set to zero.
        """
        h = y_true.shape[0]  # Number of proposals (regions)
        c = y_pred.shape[1] // 4  # Number of classes

        # Reshape predictions to be (h, c, 4) where each proposal has predictions for all classes
        y_pred_reshaped = y_pred.reshape(h, c, 4)
    
        # Initialize the output error matrix with zeros (h, 4 * c)
        smooth_l1_derivative = np.zeros_like(y_pred)

        # Iterate over each region proposal
        for i in range(h):
        # Select the predicted bounding box for the true class of the current region
            pred_bbox = y_pred_reshaped[i, int(y_class[i])]  # Shape: (4,)
        
            # Calculate the Smooth L1 loss derivative for the true class
            true_bbox = self.region_of_interest[i]  # Assuming region_of_interest stores (h, 4) coordinates
            diff = pred_bbox - true_bbox
        
            # Compute the derivative based on the Smooth L1 loss definition
            smooth_l1_derivative_value = np.where(
                np.abs(diff) < 1,  # When |diff| < 1
                diff,             # Derivative is diff
                np.sign(diff)     # Derivative is sign of diff
            )
        
            # Assign the Smooth L1 derivative to the corresponding class columns
            smooth_l1_derivative[i, int(y_class[i]) * 4: (int(y_class[i]) + 1) * 4] = smooth_l1_derivative_value
    
        return -smooth_l1_derivative / h  # Average over the number of proposals
    def backpropagation(self):
        for epoch in range(self.epochs):
            classification_output, regression_output, activations, z = self.feedforward()
            # Extract labels
            y_class = self.y[:, 0]  # True class labels (first column of y)
            y_regress = self.y[:, 1:]  # True bounding box offsets (next four columns)

            # Step 1: One-hot encode y_class
            y_class_one_hot = one_hot_encode(y_class, self.num_classes)
            # Step 2: Compute the loss for logging
            cross_entropy_loss = self.compute_cross_entropy_loss(y_class_one_hot, classification_output)
            mse_loss = self.compute_mse_loss(y_regress, regression_output, y_class)

            # Backpropagation
            # Step 3: Calculate the output layer errors
            e_L = self.compute_cross_entropy_derivative(y_class_one_hot, classification_output) # Cross-entropy derivative
            e_L_reg = self.compute_mse_derivative(y_regress, regression_output, y_class)  # MSE derivative for regression
            #e_L = np.hstack((e_L, e_L_reg))
            # Step 4: Update weights and biases for the output layer
            #self.weights[-1] -= self.learning_rate * np.clip(np.dot(activations[-2].T, e_L), -1, 1)
            #self.biases[-1] -= self.learning_rate * np.clip(np.sum(e_L, axis=0, keepdims=True), -1, 1)
            # Backpropagation updates for classification and regression
            self.weights[-1][:, -self.num_classes:] -= self.learning_rate * np.dot(activations[-2].T, e_L)
            self.weights[-1][:, :-self.num_classes] -= self.learning_rate * np.dot(activations[-2].T, e_L_reg)
            self.biases[-1][:, -self.num_classes:] -= self.learning_rate * np.sum(e_L, axis=0, keepdims=True)
            self.biases[-1][:, :-self.num_classes] -= self.learning_rate * np.sum(e_L_reg, axis=0, keepdims=True)
            e_L = np.hstack((e_L, e_L_reg))
            e_l_1 = e_L
            # Backpropagation through hidden layerser4
            for l in range(len(self.weights) - 2, -1, -1):
                # Compute the derivative of the activation function (ReLU)
                derivative = self.relu_derivative(z[l + 1])
                # Calculate error for the current layer
                e_l = np.dot(e_l_1, self.weights[l + 1].T) * derivative

                # Update weights and biases
                self.weights[l] -= self.learning_rate * np.clip(np.dot(activations[l].T, e_l), -1, 1)
                self.biases[l] -= self.learning_rate * np.clip(np.sum(e_l, axis=0, keepdims=True), -1, 1)
                e_l_1 = e_l

            # Log loss at each epoch (optional)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Cross Entropy Loss: {cross_entropy_loss}, MSE Loss: {mse_loss}")

# Parameters for the test
num_samples = 1000  # Large number of samples
num_features = 100    # Number of input features
num_classes = 10      # Number of classes (e.g., for object detection)
hidden_layer = [100, 100]  # Hidden layer sizes
epochs = 500        # Number of epochs for training
learning_rate = 0.01  # Learning rate

# Generate synthetic input data (X) and true labels (y)
np.random.seed(42)  # For reproducibility
X = np.random.rand(num_samples, num_features)  # Input data
y_classes = np.random.randint(0, num_classes, size=(num_samples, 1))  # Random class labels
y_bboxes = np.random.rand(num_samples, 4)  # Random bounding box values (xmin, ymin, xmax, ymax)

# Combine class labels and bounding boxes into one y array
y = np.hstack((y_classes, y_bboxes))  # Shape: (n_samples, 5)

# Generate random region of interest values (same shape as y_bboxes)
region_of_interest = np.random.rand(num_samples, 4)

# Initialize and train the neural network
nn = NeuralNetwork(X, y, region_of_interest, hidden_layer, num_classes, epochs, learning_rate)
nn.backpropagation()

# After training, you can also test the feedforward method to see outputs
classification_output, regression_output, activations, r = nn.feedforward()
print("Classification Output Shape:", classification_output.shape)
print("Regression Output Shape:", regression_output.shape)

def calculate_classification_accuracy(y_true_class, y_pred_class):
    """
    Calculate the classification accuracy.
    y_true_class: True class labels (shape: (n_samples,)).
    y_pred_class: Predicted class probabilities (shape: (n_samples, num_classes)).
    """
    # Get the predicted class by selecting the class with the highest probability
    y_pred_labels = np.argmax(y_pred_class, axis=1)
    
    # Calculate accuracy by comparing with true labels
    accuracy = np.mean(y_pred_labels == y_true_class)
    
    return accuracy

def calculate_regression_error(y_true_bbox, y_pred_bbox, y_class):
    """
    Calculate the Mean Squared Error for the bounding box regression.
    y_true_bbox: Ground-truth bounding boxes (shape: (n_samples, 4)).
    y_pred_bbox: Predicted bounding boxes (shape: (n_samples, 4 * num_classes)).
    y_class: True class labels (shape: (n_samples,)) to select the correct predicted bounding box.
    """
    num_classes = y_pred_bbox.shape[1] // 4  # Number of classes
    y_pred_reshaped = y_pred_bbox.reshape(-1, num_classes, 4)

    # Select the predicted bounding box for the true class of each proposal
    y_pred_selected = y_pred_reshaped[np.arange(y_class.size), y_class.astype(int)]
    
    # Compute the mean squared error between the true and predicted bounding boxes
    mse = np.mean(np.square(y_true_bbox - y_pred_selected))

    return mse

# After training, use the feedforward method to get predictions
classification_output, regression_output, activations, r = nn.feedforward()

# True class labels (first column of y)
y_true_class = nn.y[:, 0]

# True bounding boxes (next 4 columns of y)
y_true_bbox = nn.y[:, 1:]

# Step 1: Calculate classification accuracy
classification_accuracy = calculate_classification_accuracy(y_true_class, classification_output)
print(f"Classification Accuracy: {classification_accuracy * 100:.2f}%")

# Step 2: Calculate regression error (MSE)
regression_mse = calculate_regression_error(y_true_bbox, regression_output, y_true_class)
print(f"Bounding Box Regression MSE: {regression_mse:.4f}")