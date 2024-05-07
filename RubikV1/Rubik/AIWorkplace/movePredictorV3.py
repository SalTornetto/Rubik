import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from testDataGenerator import generate_test_data_with_inverses, moves  # Import data generator


# Define the neural network architecture for predicting cube moves
class RubiksCubeMovePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RubiksCubeMovePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Additional layers
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.batch_norm(x)
        x = self.fc3(x)  # Output layer with 18 move possibilities
        return F.softmax(x, dim=1)  # Softmax for probabilities


# Model parameters
input_size = 351  # Example input size for a 3x3 Rubik's Cube with position and rotation
hidden_size = 64
output_size = 18

numGen = 100  # Number of sample cubes generated to train on
numMoves = 1  # The number of moves trained cubes will be shuffled by

# Initialize the model
model = RubiksCubeMovePredictor(input_size, hidden_size, output_size)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate
criterion = nn.CrossEntropyLoss()  # Loss function for probability distribution

# Generate test data
test_samples = generate_test_data_with_inverses(numGen, numMoves)  # Generate 100 samples
train_loader = DataLoader(test_samples, batch_size=32, shuffle=True)
# print(train_loader.dataset)
# Training loop with batch size consistency check
epochs = 25  # Number of epochs for training
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0  # Track total loss for monitoring

    for batch in train_loader:
        inputs, target_data = batch  # Split the batch into inputs and targets
        # print(target_data)
        print(inputs)
        # Ensure inputs are flattened correctly
        inputs = inputs.view(inputs.shape[0], -1)  # Flatten batch
        # print(inputs)
        # Flatten target data to ensure consistency
        optimal_moves = []
        for item in target_data:
            if isinstance(item, (list, tuple)):
                optimal_moves.extend(item)  # Flatten nested structures
            else:
                optimal_moves.append(item)  # Add single elements

        move_keys = list(moves.keys())

        # Convert optimal moves into valid indices
        optimal_move_indices = [move_keys.index(move) for move in optimal_moves if move in move_keys]

        # Check if batch sizes match
        # if inputs.shape[0] != len(optimal_move_indices):
        #     print(
        #         f"Mismatch in batch sizes. OMI Size {len(optimal_move_indices)}, shape {inputs.shape[0]}. Skipping batch.")
        #     continue  # Skip if there's a mismatch
        # print(optimal_move_indices)
        optimal_move_tensor = torch.tensor(optimal_move_indices, dtype=torch.long)  # Target tensor
        # print(optimal_move_tensor)
        # print(inputs.size())
        optimizer.zero_grad()  # Reset gradients before backpropagation

        # Forward pass and loss calculation
        outputs = model(inputs)
        loss = criterion(outputs, optimal_move_tensor)
        total_loss += loss.item()  # Accumulate loss

        # Backward pass and optimization
        loss.backward()  # Backpropagation
        optimizer.step()  # Gradient descent step

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")  # Average loss

# Save the trained model
model_path = 'rubiks_cube_move_predictor.pth'  # Path to save the model
torch.save(model, model_path)  # Save model state

# To reload the model
# loaded_model = RubiksCubeMovePredictor(input_size, hidden_size, output_size)
# loaded_model.load_state_dict(torch.load(model_path))  # Load the saved state





import torch
from cubeAnimationPractice import EntireCube
from testDataGenerator import starting_rot,starting_position

complete = [1]*27
cuber = EntireCube(3, .1)
cuber.update(0,0,-1)  # Prepare example input - K_F1

cubePos, cubeRot = cuber.state()
for j in range(27):
    if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j]:
        complete[j] = 1
    else:
        complete[j] = 0
position_tensor = torch.tensor(cubePos, dtype=torch.float32)
rotation_tensor = torch.tensor(cubeRot, dtype=torch.float32)
completion_tensor = torch.tensor(complete, dtype=torch.float32)
input_tensor = torch.cat([position_tensor.flatten(), rotation_tensor.flatten(), completion_tensor.flatten()])
input_tensor = input_tensor.view(input_tensor.shape[0], -1)  # Flatten batch
input_tensor = input_tensor.view(-1, 351)
# print(input_tensor)

loaded_model = torch.load("rubiks_cube_move_predictor.pth")
loaded_model.eval()


with torch.no_grad():
    output = loaded_model(input_tensor)
    print(output)
predicted_index = torch.argmax(output, dim=1)
print(predicted_index)