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


# model_path = 'rubiks_cube_move_predictor_gen_5.pth'  # Path to save the model
if epochs:
    torch.save(model, model_path)  # Save model state



import torch
from cubeAnimationPractice import EntireCube
from RubikCube import moves, starting_position, starting_rot

def test_model(move_name):
    # print(move_name)
    # Load the trained model

    # loaded_model = torch.load(model_path)
    # loaded_model = model.to(device)
    loaded_model = torch.load(model_path, map_location=device)
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    # Set up a 3x3 Rubik's Cube and apply a few moves
    cuber = EntireCube(3, .1)
    # for m in move_name:
    #     print(m)
    #     cuber.update(*moves[m])  # Apply initial moves
    # move_list_temp = ['K_5', 'K_F5', 'K_F1', 'K_F5', 'K_F8', 'K_8', 'K_F5', 'K_F4', 'K_F3', 'K_F2', 'K_F1', 'K_F9', 'K_F6', 'K_6', 'K_7', 'K_F8', 'K_F7']
    # move_list_temp = ['K_F1', 'K_F5', 'K_F5', 'K_F4', 'K_F9', 'K_F6', 'K_6', 'K_7']
    #
    # for move_temp in move_list_temp:
    #     cuber.update(*moves[move_temp])


    cuber.update(*moves[move_name])  # Apply initial moves

    # cuber.update(*moves["K_2"])
    # cuber.update(*moves["K_F1"])
    # cuber.update(*moves["K_F9"])
    # cuber.update(*moves["K_F3"])


    # Number of allowed moves for solving
    max_moves = 100  # Example limit for testing
    moves_applied = 0
    attempted_moves = []

    # Function to generate an input tensor for the cube's current state
    def generate_input_tensor(cube):
            cubePos, cubeRot = cube.state()  # Get cube's current position and rotation
            complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in range(27)]
            position_tensor = torch.tensor(cubePos, dtype=torch.float32)
            rotation_tensor = torch.tensor(cubeRot, dtype=torch.float32)
            completion_tensor = torch.tensor(complete, dtype=torch.float32)
            input_tensor = torch.cat([position_tensor.flatten(), rotation_tensor.flatten(), completion_tensor.flatten()])
            return input_tensor.view(-1, 351)  # Return flattened input tensor



#
#
#
#     # Define the function to apply a predicted move
# def apply_predicted_move(cube, move):
#     """
#     Apply a given move to the cube.
#
#     :param cube: The Rubik's cube object to be manipulated.
#     :param move: The move to be applied.
#     """
#     # Assuming that the cube has an 'update' method to apply the move
#     cube.update(*move)  # Assuming 'move' is a tuple or list of instructions
#
# # Define the function to test the model
# def test_model(moves, cuber, loaded_model, device, starting_position, starting_rot, max_moves):
#         """
#         Apply a list of moves to the cube and check if it's solved.
#
#         :param moves: List of moves to apply.
#         :param cuber: The Rubik's cube object.
#         :param loaded_model: The pre-trained model for predicting moves.
#         :param device: Device (e.g., 'cpu' or 'cuda') for model execution.
#         :param starting_position: The initial position of the cube.
#         :param starting_rot: The initial rotation of the cube.
#         :param max_moves: The maximum number of moves to apply.
#         """
#         moves_applied = 0
#         attempted_moves = []
#
#         while moves_applied < max_moves:
#             # Generate the input tensor for the cube's current state
#             input_tensor = generate_input_tensor(cuber).to(device)
#
#             # Get the predicted move from the model
#             with torch.no_grad():
#                 output = loaded_model(input_tensor)
#                 predicted_index = torch.argmax(output, dim=1).item()  # Get the predicted move index
#
#             # Apply the predicted move
#             apply_predicted_move(cuber, moves[predicted_index])  # Apply move
#             moves_applied += 1  # Increment the applied moves count
#
#             # Get cube's current position and rotation
#             cubePos, cubeRot = cuber.state()
#             complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in
#                         range(27)]
#
#             # Track the moves attempted
#             attempted_moves.append(moves[predicted_index])
#
#             # Check if the cube is solved
#             if all(complete):
#                 print("The cube is solved!")
#                 break
#
#         print("Total moves applied:", moves_applied)
#         print("Attempted moves:", attempted_moves)
#
#
#
#     # Example list of moves
# move_list = ["K_1", "K_2"]
#
# # Example function call to test the model
# # Assuming 'cuber', 'loaded_model', 'device', 'starting_position', 'starting_rot', and 'max_moves' are predefined
# test_model(move_list, cuber, loaded_model, device, starting_position, starting_rot, max_moves)







    # Function to apply a predicted move to the cube

    def apply_predicted_move(cube, move_index):
        # move_key = list(moves.keys())[move_index]
        # cube.update(*moves[move_key])

        # move_list_temp = ['K_5']
        # # move_list_temp = ['K_5', 'K_F5', 'K_F1', 'K_F5', 'K_F8', 'K_8', 'K_F5', 'K_F4', 'K_F3', 'K_F2', 'K_F1', 'K_F9', 'K_F6', 'K_6', 'K_7']
        #
        # for move_temp in move_list_temp:
        #     cube.update(*moves[move_temp])


        cube.update(*moves[list(moves)[move_index]])
        # print(list(moves)[move_index])
        # list(moves)[0]


    # Loop to apply multiple moves until the cube is solved or the limit is reached
    while moves_applied < max_moves:
        # Generate the input tensor for the cube's current state
        input_tensor = generate_input_tensor(cuber)
        input_tensor = input_tensor.to(device)

        # Get the predicted move from the model
        with torch.no_grad():
            output = loaded_model(input_tensor)
            predicted_index = torch.argmax(output, dim=1).item()  # Get the predicted move index

        # Apply the predicted move to the cube
        apply_predicted_move(cuber, predicted_index)
        moves_applied += 1

        cubePos, cubeRot = cuber.state()  # Get cube's current position and rotation
        complete = [1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j] else 0 for j in range(27)]

        # print(output)
        # print(predicted_index)

        attempted_moves.append(list(moves)[predicted_index])
        # Check if the cube is solved
        if all(complete):
            print("The cube is solved!")
            break

    print("Total moves applied:", moves_applied)
    print(attempted_moves)




# move_name = [
#     "K_1",
# ]
# move_name = moves
# test_model(move_name)
# print(list(moves))
for move_name in list(moves):
    # print(move_name)
    test_model(move_name)
