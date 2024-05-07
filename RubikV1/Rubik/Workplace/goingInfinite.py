import torch
import random
from cubeAnimationPractice import EntireCube
from RubikCube import moves, inverse_moves, starting_position, starting_rot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

moves = {
    "K_1": (0, 0, 1), "K_2": (0, 1, 1), "K_3": (0, 2, 1), "K_4": (1, 0, 1), "K_5": (1, 1, 1),
    "K_6": (1, 2, 1), "K_7": (2, 0, 1), "K_8": (2, 1, 1), "K_9": (2, 2, 1),
    "K_F1": (0, 0, -1), "K_F2": (0, 1, -1), "K_F3": (0, 2, -1), "K_F4": (1, 0, -1), "K_F5": (1, 1, -1),
    "K_F6": (1, 2, -1), "K_F7": (2, 0, -1), "K_F8": (2, 1, -1), "K_F9": (2, 2, -1),
}

inverse_moves = {
    "K_1": "K_F1", "K_2": "K_F2", "K_3": "K_F3", "K_4": "K_F4", "K_5": "K_F5",
    "K_6": "K_F6", "K_7": "K_F7", "K_8": "K_F8", "K_9": "K_F9",
    "K_F1": "K_1", "K_F2": "K_2", "K_F3": "K_F3", "K_F4": "K_4", "K_F5": "K_5",
    "K_F6": "K_6", "K_F7": "K_7", "K_F8": "K_8", "K_F9": "K_9"
}


# Recursive function to generate all valid move sequences
def generate_moves(n, current_moves=None):
    if current_moves is None:
        current_moves = []

    # Base case: if the desired length is reached, return the current sequence
    if len(current_moves) == n:
        return [current_moves]

    # Recursive case: add a new valid move
    all_sequences = []
    for move_name in moves:
        if current_moves and inverse_moves[move_name] == current_moves[-1]:
            continue  # Skip inverse moves
        new_moves = current_moves + [move_name]  # Extend the current sequence
        all_sequences.extend(generate_moves(n, new_moves))  # Recursively generate more sequences

    return all_sequences




# Function to generate cube configurations and their optimal moves
def generate_all_possible_cube_data(num_moves):
    # List to store generated cube configurations
    cube_data = []
    complete = [1] * 27
    dim = 3

    # Get all possible move combinations for the given number of moves
    move_combinations = generate_moves(num_moves)

    # Loop through each combination to generate cube states
    for applied_moves in move_combinations:
        # Create a new cube in solved state
        cube = EntireCube(dim, .1)  # Replace with your cube initialization logic

        # Apply the sequence of moves to generate a configuration
        for move in applied_moves:
            cube.update(*moves[move])

        # Determine the optimal solution by reversing the applied moves
        optimal_solution = [inverse_moves[move] for move in reversed(applied_moves)]

        cubePos, cubeRot = cube.state()

        # Determine which cubes are in their original state
        for j in range(dim**3):
            complete[j] = (
                1 if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j]
                else 0
            )

        # Convert the cube state to tensors
        position_tensor = torch.tensor(cubePos, dtype=torch.float32).to(device)
        rotation_tensor = torch.tensor(cubeRot, dtype=torch.float32).to(device)
        completion_tensor = torch.tensor(complete, dtype=torch.float32).to(device)

        # Flatten the tensors for model input
        input_tensor = torch.cat(
            [position_tensor.flatten(), rotation_tensor.flatten(), completion_tensor.flatten()]
        )

        # Store the input tensor and the optimal move(s)
        cube_data.append((input_tensor, optimal_solution))

    return cube_data


# Example usage: Generate data for all possible combinations with 2 moves
# cube_combinations = generate_all_possible_cube_data(2)  # Example for 2 moves
# print(cube_combinations[0])  # Display the first 3 configurations to verify
# print(len(cube_combinations), "cube configurations found.")



# Define the moves and inverse moves
# Example usage: Get all possible combinations for 2 moves
# sequences = generate_moves(5)  # This will generate all 18 * 17 move lists
# # print(len(sequences), "combinations found.")
# # print(sequences)  # Print the first 5 combinations to verify
# print(len(sequences), "combinations found.")