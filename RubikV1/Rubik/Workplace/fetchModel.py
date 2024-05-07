#WIP

import torch
from cubeAnimationPractice import EntireCube
from RubikCube import moves, inverse_moves, starting_position, starting_rot
# import movePredictorV3

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



loaded_model = torch.load("rubiks_cube_move_predictor.pth")
loaded_model.eval()

with torch.no_grad():
    output = loaded_model(input_tensor)
print(output)
