import torch
import random
from cubeAnimationPractice import EntireCube
moves = {
        "K_1": (0, 0, 1), "K_2": (0, 1, 1), "K_3": (0, 2, 1), "K_4": (1, 0, 1), "K_5": (1, 1, 1),
        "K_6": (1, 2, 1), "K_7": (2, 0, 1), "K_8": (2, 1, 1), "K_9": (2, 2, 1),
        "K_F1": (0, 0, -1), "K_F2": (0, 1, -1), "K_F3": (0, 2, -1), "K_F4": (1, 0, -1), "K_F5": (1, 1, -1),
        "K_F6": (1, 2, -1), "K_F7": (2, 0, -1), "K_F8": (2, 1, -1), "K_F9": (2, 2, -1),
    }

    # Create a dictionary of inverse moves
inverse_moves = {
        "K_1": "K_F1", "K_2": "K_F2", "K_3": "K_F3", "K_4": "K_F4", "K_5": "K_F5",
        "K_6": "K_F6", "K_7": "K_F7", "K_8": "K_F8", "K_9": "K_F9",
        "K_F1": "K_1", "K_F2": "K_2", "K_F3": "K_3", "K_F4": "K_F4", "K_F5": "K_F5",
        "K_F6": "K_6", "K_F7": "K_7", "K_F8": "K_8", "K_F9": "K_9"
    }


starting_position= [
    [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0],
    [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1],
    [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2],
    [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]
]


starting_rot = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ]

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
            # for cube in cubie.cubes:
                cube.update(*moves[move])

            # cube.update(*moves[move])

        # Determine the optimal solution by reversing the applied moves
        optimal_solution = [inverse_moves[move] for move in reversed(applied_moves)]

        cubePos, cubeRot = cube.state()

        for j in range(dim**3):
            # print(j)
            if cubePos[j] == starting_position[j] and cubeRot[j] == starting_rot[j]:
                # complete.append(1)
                complete[j] = 1
            else:
                # complete.append(0)
                complete[j] = 0
        # print(cubePos)
        # print("completion")
        # print(complete)

        # print(cubeRot)

# create another tensor for whether or not a cube is fully solved?

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


# Generate test data with 10 samples, each with 3 moves from the solved state
# test_samples = generate_test_data_with_inverses(10, 2)
# print(test_samples)
# Create a DataLoader for managing the test data
# test_loader = torch.utils.data.DataLoader(test_samples, batch_size=2, shuffle=True)
