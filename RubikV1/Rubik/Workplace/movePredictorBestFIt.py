import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from trainingDataGenerator import generate_test_data_with_inverses
from RubikCube import moves, inverse_moves, starting_position, starting_rot

# Define the neural network architecture for predicting cube moves
class RubiksCubeMovePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RubiksCubeMovePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
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
input_size = 351  # Input size for 3x3 Rubik's Cube
hidden_size = 64
output_size = 18


numGen = 10000  # Number of sample cubes generated to train on
numMoves = 1  # The number of moves trained cubes will be shuffled by

# Initialize the model
model = RubiksCubeMovePredictor(input_size, hidden_size, output_size)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Loss function for probability distribution

test_samples = generate_test_data_with_inverses(numGen, numMoves)  # Generate 100 samples
train_loader = DataLoader(test_samples, batch_size=32, shuffle=False)


# Training loop with additional loop for sequential moves
epochs = 0  # Number of epochs for training
for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0  # Track total loss for monitoring

    for batch in train_loader:
        inputs, target_data = batch  # Split the batch into inputs and targets
        inputs = inputs.view(inputs.shape[0], -1)  # Flatten batch
        optimizer.zero_grad()  # Reset gradients before backpropagation

        # Sequential loop to handle multiple optimal moves
        for target_step in target_data:
            # Flatten the target_step if it's a list or tuple
            optimal_moves = []
            if isinstance(target_step, (list, tuple)):
                optimal_moves.extend(target_step)
            else:
                optimal_moves.append(target_step)

            move_keys = list(moves.keys())
            # Convert optimal moves into valid indices
            optimal_move_indices = [move_keys.index(move) for move in optimal_moves]

            optimal_move_tensor = torch.tensor(optimal_move_indices, dtype=torch.long)  # Target tensor

            # Forward pass and loss calculation
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, optimal_move_tensor)  # Loss calculation
            total_loss += loss.item()  # Accumulate loss

            # Backward pass and optimization
            loss.backward()  # Backpropagation
            optimizer.step()  # Gradient descent step

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")


model_path = 'rubiks_cube_move_predictor.pth'  # Path to save the model
if epochs:
    torch.save(model, model_path)  # Save model state





import torch
# from cubeAnimationPractice import EntireCube
from RubikCube import EntireCube, moves, starting_position, starting_rot
import copy

# Define a scoring function to evaluate the cube's state
def score_cube(cube):
    cubePos, cubeRot = cube.state()
    score = sum(1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in range(27))
    return score

# Recursive lookahead function to evaluate multiple moves ahead
def lookahead(cube, depth, best_score=float('inf'), best_solution=None):
    if depth == 0:
        # Base case: score the cube's state
        score = score_cube(cube)
        return score, []

    best_local_score = best_score
    best_local_solution = best_solution if best_solution else []

    # Try all possible moves to find the best one
    for move in moves.keys():
        # Clone the cube to avoid modifying the original
        new_cube = copy.deepcopy(cube)
        new_cube.update(*moves[move])

        # Recursive lookahead for the reduced depth
        score, solution = lookahead(new_cube, depth - 1, best_local_score, best_local_solution)

        if score > best_local_score:  # A higher score means a more solved state
            best_local_score = score
            best_local_solution = [move] + solution

    return best_local_score, best_local_solution


def test_model_with_lookahead(model, initial_moves, lookahead_depth=3):
    # Load the model if not already loaded
    model.eval()  # Set to evaluation mode

    # Set up the cube with initial moves
    cuber = EntireCube(3, .1)
    for m in initial_moves:
        cuber.update(*moves[m])  # Apply initial moves

    # Get the predicted moves using lookahead
    score, best_moves = lookahead(cuber, lookahead_depth)

    # Apply the predicted moves
    for move in best_moves:
        cuber.update(*moves[move])

    # Check if the cube is solved
    is_solved = score == 27
    return is_solved, best_moves


# Example test run
model_path = "rubiks_cube_move_predictor.pth"
loaded_model = torch.load(model_path)  # Load the model
# initial_moves = ["K_1", "K_6", "K_8"]  # Example initial moves
initial_moves = []  # Example initial moves
solved, optimal_moves = test_model_with_lookahead(loaded_model, initial_moves)

if solved:
    print("The cube is solved!")
else:
    print("The cube is not solved.")
print("Optimal moves:", optimal_moves)
