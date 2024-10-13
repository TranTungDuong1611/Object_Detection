import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# One-hot encoding function using torch
def one_hot_encode(y, num_classes):
    """
    Convert an array of class labels (shape: (n_samples,)) into one-hot encoded format (shape: (n_samples, num_classes)).
    """
    y_one_hot = torch.zeros(y.size(0), num_classes).to(y.device)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    return y_one_hot

# Neural network class in PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer, num_classes):
        super(NeuralNetwork, self).__init__()
        
        # Define the network layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_layer[0]))
        
        for i in range(len(hidden_layer) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_layer[i], hidden_layer[i + 1]))
        
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer[-1], num_classes * 5))  # For classification + bounding box offsets
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    def compute_cross_entropy_loss(self, y_true, y_pred):
        return F.cross_entropy(y_pred, y_true)

    def compute_mse_loss(self, y_true, y_pred, y_class):
        """
        Compute mean squared error loss for bounding box regression.
        """
        num_classes = y_pred.size(1) // 4
        y_pred_reshaped = y_pred.view(-1, num_classes, 4)
        y_pred_selected = y_pred_reshaped[torch.arange(y_class.size(0)), y_class]
        return F.mse_loss(y_pred_selected, y_true)

    def compute_bounding_box_error(self, pred_offsets, y, region_of_interest):
        """
        Compute bounding box regression errors.
        """
        bbox_errors = torch.zeros(y.size(0), self.num_classes * 4).to(y.device)
        reshaped_offsets = pred_offsets.view(-1, self.num_classes, 4)

        for i in range(reshaped_offsets.size(0)):
            true_class_index = int(y[i, 0])
            if true_class_index < self.num_classes:
                true_bbox = y[i, 1:]
                predicted_bbox = region_of_interest[i] + reshaped_offsets[i, true_class_index]
                bbox_errors[i, true_class_index * 4: true_class_index * 4 + 4] = predicted_bbox - true_bbox
        
        return bbox_errors

# Training function
def train_nn(model, X, y, region_of_interest, num_classes, epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()

        # Forward pass
        outputs = model(X)
        classification_output = outputs[:, -num_classes:]
        regression_output = outputs[:, :-num_classes]

        # Extract labels
        y_class = y[:, 0].long()
        y_regress = y[:, 1:]

        # One-hot encode class labels
        y_class_one_hot = one_hot_encode(y_class, num_classes)

        # Compute losses
        cross_entropy_loss = model.compute_cross_entropy_loss(y_class, classification_output)
        mse_loss = model.compute_mse_loss(y_regress, regression_output, y_class)

        # Combine the losses
        total_loss = cross_entropy_loss + mse_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log the loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Cross Entropy Loss: {cross_entropy_loss.item()}, MSE Loss: {mse_loss.item()}")

# Accuracy function
def calculate_classification_accuracy(y_true_class, y_pred_class):
    y_pred_labels = torch.argmax(y_pred_class, dim=1)
    accuracy = torch.mean((y_pred_labels == y_true_class).float())
    return accuracy.item()

# Regression error function
def calculate_regression_error(y_true_bbox, y_pred_bbox, y_class):
    num_classes = y_pred_bbox.size(1) // 4
    y_pred_reshaped = y_pred_bbox.view(-1, num_classes, 4)
    y_pred_selected = y_pred_reshaped[torch.arange(y_class.size(0)), y_class]
    mse = F.mse_loss(y_true_bbox, y_pred_selected)
    return mse.item()

# Parameters for the test
num_samples = 200
num_features = 980
num_classes = 10
hidden_layer = [200, 100]
epochs = 3000
learning_rate = 0.01

# Generate synthetic input data (X) and true labels (y)
torch.manual_seed(42)
X = torch.rand(num_samples, num_features)
y_classes = torch.randint(0, num_classes, (num_samples, 1)).float()
y_bboxes = torch.rand(num_samples, 4)
y = torch.cat((y_classes, y_bboxes), dim=1)

region_of_interest = torch.rand(num_samples, 4)

# Initialize and train the neural network
nn_model = NeuralNetwork(num_features, hidden_layer, num_classes)
train_nn(nn_model, X, y, region_of_interest, num_classes, epochs, learning_rate)

# After training, use the feedforward method to get predictions
nn_model.eval()
with torch.no_grad():
    outputs = nn_model(X)
    classification_output = outputs[:, -num_classes:]
    regression_output = outputs[:, :-num_classes]

    # Calculate accuracy and regression error
    y_true_class = y[:, 0].long()
    y_true_bbox = y[:, 1:]

    classification_accuracy = calculate_classification_accuracy(y_true_class, classification_output)
    regression_mse = calculate_regression_error(y_true_bbox, regression_output, y_true_class)

    print(f"Classification Accuracy: {classification_accuracy * 100:.2f}%")
    print(f"Bounding Box Regression MSE: {regression_mse:.4f}")