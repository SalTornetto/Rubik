import torch
from RubikCube import EntireCube
from RubikCube import moves, starting_position, starting_rot
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


input_size = 351  # Input size for 3x3 Rubik's Cube
hidden_size = 64
output_size = 18


# Load the trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'rubiks_cube_move_predictor_gen_5.pth'


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


model = RubiksCubeMovePredictor(input_size, hidden_size, output_size)
loaded_model = torch.load(model_path, map_location=device)
loaded_model = loaded_model.to(device)
loaded_model.eval()

# Create a 3x3 Rubik's Cube instance
cuber = EntireCube(3, 0.1)


# Define the function to generate the input tensor
def generate_input_tensor(cube):
    cubePos, cubeRot = cube.state()  # Get cube's current position and rotation
    complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in range(27)]
    position_tensor = torch.tensor(cubePos, dtype=torch.float32)
    rotation_tensor = torch.tensor(cubeRot, dtype=torch.float32)
    completion_tensor = torch.tensor(complete, dtype=torch.float32)
    input_tensor = torch.cat([position_tensor.flatten(), rotation_tensor.flatten(), completion_tensor.flatten()])
    return input_tensor.view(-1, 351)  # Flatten the tensor and return


# Define the function to apply a move
def apply_move(cube, move):
    """
    Apply a move to the Rubik's Cube.
    """
    cube.update(*move)  # Apply the move to the cube


# Define the function to test the model with a list of moves
def test_model(moves, cuber, loaded_model, device, max_moves):
    moves_applied = 0
    attempted_moves = []

    for move_name in moves:
        # Apply the provided moves to the cube
        apply_move(cuber, move_name)

    # Testing loop to apply predicted moves
    while moves_applied < max_moves:
        # Generate the input tensor from the cube's current state
        input_tensor = generate_input_tensor(cuber).to(device)

        # Get the model's predicted move
        with torch.no_grad():
            output = loaded_model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()  # Get the predicted move index

        # Apply the predicted move
        apply_move(cuber, moves[predicted_index])
        moves_applied += 1  # Increment the applied moves count

        # Check if the cube is solved
        cubePos, cubeRot = cuber.state()  # Get cube's current position and rotation
        complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in range(27)]

        if all(complete):
            print("The Rubik's Cube is solved!")
            break

        # Track the attempted moves
        attempted_moves.append(list(moves)[predicted_index])

    print(f"Total moves applied: {moves_applied}")
    print(f"Attempted moves: {attempted_moves}")


# List of moves to apply sequentially
move_list = ["K_1", "K_2"]

# Test the model with the list of moves
max_moves = 100  # Example limit for applying moves
test_model(move_list, cuber, loaded_model, device, max_moves)
