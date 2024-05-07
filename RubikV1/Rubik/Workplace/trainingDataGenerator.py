import torch
import random
from cubeAnimationPractice import EntireCube
from RubikCube import moves, inverse_moves, starting_position, starting_rot

# Define a function to generate cube configurations and their optimal moves
def generate_test_data_with_inverses(num_samples, num_moves):
    # List to store generated test data
    test_data = []
    complete = [1] * 27
    dim = 3

    for _ in range(num_samples):

        # Create a new cube in solved state
        cube = EntireCube(dim, .1)  # Replace with your cube initialization logic

        # Apply a sequence of random moves to generate a configuration
        applied_moves = random.choices(list(moves.keys()), k=num_moves)

        # Apply the moves to the cube
        for move in applied_moves:
                cube.update(*moves[move])

        # Determine the optimal solution by reversing the applied moves
        optimal_solution = [inverse_moves[move] for move in reversed(applied_moves)]

        cubePos, cubeRot = cube.state()

        for j in range(dim**3):
            if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j]:
                complete[j] = 1
            else:
                complete[j] = 0



        # Convert the cube state to tensors
        position_tensor = torch.tensor(cubePos, dtype=torch.float32)
        rotation_tensor = torch.tensor(cubeRot, dtype=torch.float32)
        completion_tensor = torch.tensor(complete, dtype=torch.float32)

        # Flatten the tensors for the model input
        input_tensor = torch.cat([position_tensor.flatten(), rotation_tensor.flatten(), completion_tensor.flatten()])

        # Store the input tensor and the optimal move(s)
        test_data.append((input_tensor, optimal_solution))
        # print(test_data)
    return test_data


# Generate test data with 10 samples, each with 1 moves from the solved state
# test_samples = generate_test_data_with_inverses(10, 1)
# print(test_samples)
# Create a DataLoader for managing the test data
# test_loader = torch.utils.data.DataLoader(test_samples, batch_size=2, shuffle=True)
