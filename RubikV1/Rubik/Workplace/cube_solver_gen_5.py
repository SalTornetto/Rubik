import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Workplace.goingInfinite import generate_all_possible_cube_data
from trainingDataGenerator import generate_test_data_with_inverses
from RubikCube import moves, inverse_moves, starting_position, starting_rot

model_path = "rubiks_cube_move_predictor_gen_4.pth"

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"


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

total_cube_list = 1

# numGen = 100  # Number of sample cubes generated to train on
# numMoves = 3  # The number of moves trained cubes will be shuffled by

# Initialize the model
model = RubiksCubeMovePredictor(input_size, hidden_size, output_size)
model = torch.load(model_path)
model = model.to(device)
print(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Loss function for probability distribution

# test_samples = generate_test_data_with_inverses(numGen, numMoves)  # Generate 100 samples
test_samples = generate_all_possible_cube_data(total_cube_list)
train_loader = DataLoader(test_samples, batch_size=32, shuffle=False)

# Training loop with additional loop for sequential moves
epochs = 0  # Number of epochs for training

for epoch in range(epochs):
    model.train()  # Set model to training mode
    total_loss = 0  # Track total loss for monitoring

    for batch in train_loader:
        inputs, target_data = batch  # Split the batch into inputs and targets
        # Ensure the input tensor is on the correct device
        inputs = inputs.to(device)
        # target_data = list(map(list, zip(*target_data_transpose)))
        # print(target_data)
        inputs = inputs.view(inputs.shape[0], -1)  # Flatten batch
        optimizer.zero_grad()  # Reset gradients before backpropagation

        # Sequential loop to handle multiple optimal moves

        for target_step in target_data:
            # print("target_data")
            # print(target_data)
            # print("target_step")
            # print(target_step)
            # Flatten the target_step if it's a list or tuple
            optimal_moves = []
            if isinstance(target_step, (list, tuple)):
                optimal_moves.extend(target_step)
            else:
                optimal_moves.append(target_step)
            # print("optimal_moves")
            # print(optimal_moves)
            move_keys = list(moves.keys())
            # Convert optimal moves into valid indices
            optimal_move_indices = [move_keys.index(move) for move in optimal_moves]

            # optimal_move_tensor = torch.tensor(optimal_move_indices, dtype=torch.long)  # Target tensor
            # Example initialization of a tensor, ensuring it's on the correct device
            optimal_move_tensor = torch.tensor(optimal_move_indices, dtype=torch.long).to(device)

            # Forward pass and loss calculation
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, optimal_move_tensor)  # Loss calculation
            total_loss += loss.item()  # Accumulate loss

            # Backward pass and optimization
            loss.backward()  # Backpropagation
            optimizer.step()  # Gradient descent step

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# model_path = 'rubiks_cube_move_predictor.pth'  # Path to save the model
if epochs:
    torch.save(model, model_path)  # Save model state

import random
import torch
from cubeAnimationPractice import EntireCube
from RubikCube import moves, starting_position, starting_rot

# Device configuration
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'CPU'

# Load the trained model
model_path = "rubiks_cube_move_predictor_gen_4.pth"
loaded_model = torch.load(model_path, map_location=device)
loaded_model = loaded_model.to(device)
loaded_model.eval()


# Function to shuffle the cube with random moves
def shuffle_cube(cube, num_shuffles):
    shuffle_moves = []
    for _ in range(num_shuffles):
        move_key = random.choice(list(moves.keys()))  # Pick a random move
        cube.update(*moves[move_key])  # Apply the move to the cube
        shuffle_moves.append(move_key)  # Store the applied move
    return shuffle_moves  # Return the moves used to shuffle the cube


# Function to test the AI model's ability to solve the shuffled cube
def test_model(cube, max_moves):
    moves_applied = 0
    attempted_moves = []

    # Function to generate an input tensor for the cube's current state
    def generate_input_tensor(cube):
        cubePos, cubeRot = cube.state()  # Get cube's current position and rotation
        complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in
                    range(27)]
        position_tensor = torch.tensor(cubePos, dtype=torch.float32)
        rotation_tensor = torch.tensor(cubeRot, dtype=torch.float32)
        completion_tensor = torch.tensor(complete, dtype=torch.float32)
        input_tensor = torch.cat(
            [position_tensor.flatten(), rotation_tensor.flatten(), completion_tensor.flatten()])
        return input_tensor.view(-1, 351)  # Return flattened input tensor

    # Loop to apply predicted moves until the cube is solved or the limit is reached
    while moves_applied < max_moves:
        input_tensor = generate_input_tensor(cube)  # Get input tensor
        input_tensor = input_tensor.to(device)

        # Predict the move and apply it to the cube
        with torch.no_grad():
            output = loaded_model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()  # Get the predicted move index

        cube.update(*moves[list(moves.keys())[predicted_index]])  # Apply the predicted move
        moves_applied += 1
        attempted_moves.append(list(moves.keys())[predicted_index])

        # Check if the cube is solved
        cubePos, cubeRot = cube.state()  # Get cube's current position and rotation
        complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in
                    range(27)]

        if all(complete):  # If the cube is solved, return True
            return True, moves_applied, attempted_moves

    return False, moves_applied, attempted_moves  # If not solved, return False


# Parameters for shuffling and testing
num_shuffles = 10  # Number of times to shuffle the cube
target_moves = 25  # Maximum number of moves to solve the cube
attempts = 50000  # Number of times to attempt solving
criteria = 0      # only returns values above the criteria mark - used to remove trivial solutions.

# List to store successful shuffles
successful_shuffles = []

# Main loop to shuffle, test, and store successful shuffles
for attempt in range(attempts):
# while True:
    cube = EntireCube(3, 0.1)  # Create a new 3x3 Rubik's cube object
    shuffle_moves = shuffle_cube(cube, num_shuffles)  # Shuffle the cube
    solved, moves_count, attempted_moves = test_model(cube, target_moves)  # Test the model

    print("Attempt" + str(attempt))

    if solved and moves_count >= criteria:
        successful_shuffles.append((shuffle_moves, attempted_moves, moves_count))  # Store successful shuffles
        # break
# Output results
print("Number of successful shuffles:", len(successful_shuffles))
print("Successful shuffles:", successful_shuffles)

#
# import torch
# from cubeAnimationPractice import EntireCube
# from RubikCube import moves, starting_position, starting_rot
#
#
# def test_model(move_name):
#     # print(move_name)
#     # Load the trained model
#
#     # loaded_model = torch.load(model_path)
#     # loaded_model = model.to(device)
#     loaded_model = torch.load(model_path, map_location=device)
#     loaded_model = loaded_model.to(device)
#     loaded_model.eval()
#
#     # Set up a 3x3 Rubik's Cube and apply a few moves
#     cuber = EntireCube(3, .1)
#     # for m in move_name:
#     #     print(m)
#     #     cuber.update(*moves[m])  # Apply initial moves
#
#     cuber.update(*moves[move_name])  # Apply initial moves
#
#     # cuber.update(*moves["K_2"])
#     # cuber.update(*moves["K_F1"])
#     # cuber.update(*moves["K_F9"])
#     # cuber.update(*moves["K_F3"])
#
#     # Number of allowed moves for solving
#     max_moves = 100  # Example limit for testing
#     moves_applied = 0
#     attempted_moves = []
#
#     # Function to generate an input tensor for the cube's current state
#     def generate_input_tensor(cube):
#         cubePos, cubeRot = cube.state()  # Get cube's current position and rotation
#         complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in range(27)]
#         position_tensor = torch.tensor(cubePos, dtype=torch.float32)
#         rotation_tensor = torch.tensor(cubeRot, dtype=torch.float32)
#         completion_tensor = torch.tensor(complete, dtype=torch.float32)
#         input_tensor = torch.cat([position_tensor.flatten(), rotation_tensor.flatten(), completion_tensor.flatten()])
#         return input_tensor.view(-1, 351)  # Return flattened input tensor
#
#     # Function to apply a predicted move to the cube
#     def apply_predicted_move(cube, move_index):
#         # move_key = list(moves.keys())[move_index]
#         # cube.update(*moves[move_key])
#         cube.update(*moves[list(moves)[move_index]])
#         # print(list(moves)[move_index])
#         # list(moves)[0]
#
#     # Loop to apply multiple moves until the cube is solved or the limit is reached
#     while moves_applied < max_moves:
#         # Generate the input tensor for the cube's current state
#         input_tensor = generate_input_tensor(cuber)
#         input_tensor = input_tensor.to(device)
#
#         # Get the predicted move from the model
#         with torch.no_grad():
#             output = loaded_model(input_tensor)
#             predicted_index = torch.argmax(output, dim=1).item()  # Get the predicted move index
#
#         # Apply the predicted move to the cube
#         apply_predicted_move(cuber, predicted_index)
#         moves_applied += 1
#
#         cubePos, cubeRot = cuber.state()  # Get cube's current position and rotation
#         complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in range(27)]
#
#         # print(output)
#         # print(predicted_index)
#
#         attempted_moves.append(list(moves)[predicted_index])
#         # Check if the cube is solved
#         if all(complete):
#             print("The cube is solved!")
#             break
#
#     print("Total moves applied:", moves_applied)
#     print(attempted_moves)
#
#
# # move_name = [
# #     "K_1",
# # ]
# # move_name = moves
# # test_model(move_name)
# # print(list(moves))
# for move_name in list(moves):
#     # print(move_name)
#     test_model(move_name)
# #
# # import random
# # import torch
# # from copy import deepcopy
# #
# #
# # # Function to shuffle the cube randomly
# # def shuffle_cube(cube, num_shuffles, move_list):
# #     shuffle_moves = []
# #     for _ in range(num_shuffles):
# #         move = random.choice(list(move_list))  # Pick a random move
# #         cube.update(*move_list[move])
# #         shuffle_moves.append(move)
# #     return shuffle_moves  # Return the list of moves that shuffled the cube
# #
# #
# # # Function to test the AI model's ability to solve the shuffled cube
# # def test_model(cube, loaded_model, target_moves):
# #     moves_applied = 0
# #     attempted_moves = []
# #     cube.reset()  # Reset the cube to its initial state
# #
# #     while moves_applied < target_moves:
# #         input_tensor = generate_input_tensor(cube)  # Assume this function is defined
# #         input_tensor = input_tensor.to(device)
# #
# #         with torch.no_grad():
# #             output = loaded_model(input_tensor)
# #             predicted_index = torch.argmax(output, dim=1).item()  # Get the predicted move index
# #
# #         apply_predicted_move(cube, predicted_index)  # Assume this function is defined
# #         moves_applied += 1
# #
# #         attempted_moves.append(list(moves)[predicted_index])
# #
# #         # Check if the cube is solved
# #         cubePos, cubeRot = cube.state()
# #         complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in range(27)]
# #
# #         if all(complete):
# #             return True, moves_applied  # Cube solved, return True and number of moves
# #
# #     return False, moves_applied  # Cube not solved within the target moves
# #
# #
# # # Number of times to attempt solving the shuffled cube
# # attempts = 500
# # num_shuffles = 100  # Example: Number of shuffles
# # target_moves = 200  # Maximum number of moves to solve the cube
# #
# # successful_shuffles = []
# #
# # for _ in range(attempts):
# #     cube = Cube()  # Create a new cube object
# #     shuffle_moves = shuffle_cube(cube, num_shuffles, moves)  # Shuffle the cube
# #     solved, moves_count = test_model(cube, loaded_model, target_moves)  # Test the model
# #
# #     if solved:
# #         successful_shuffles.append((shuffle_moves, moves_count))  # Store the shuffle moves
# #
# # print("Number of successful shuffles:", len(successful_shuffles))
# # print("Successful shuffles:", successful_shuffles)
