import time
from pyKey import pressKey, releaseKey  # Ensure correct import for key press and release functions

# Function to perform a series of key moves with a specified delay between presses
def perform_key_moves(sample_moves, delay_between_moves=3):
    """
    Perform a sequence of key presses and releases with a delay between them.

    :param sample_moves: List of key moves (e.g., 'K_F4', 'K_7', etc.)
    :param delay_between_moves: Time in seconds between key presses and releases (default: 1 second)
    """
    time.sleep(2)  # Optional initial delay to give you time to switch contexts

    # Loop through the list of key moves and press/release each with a delay
    for move in sample_moves:
        # Trim the first two characters (assuming the first two are 'K_')
        trimmed_move = move[2:]
        # Press and release with a specified delay
        pressKey(trimmed_move)
        time.sleep(delay_between_moves)
        releaseKey(trimmed_move)

# Example usage of the function
sample_moves = [
    # 'K_F8', 'K_2', 'K_F3', 'K_F3', 'K_1', 'K_F3', 'K_5', 'K_6', 'K_1'
    # 'K_5', 'K_F5', 'K_F1', 'K_F5', 'K_F8', 'K_8', 'K_F5', 'K_F4', 'K_F3', 'K_F2', 'K_F1', 'K_F9', 'K_F6', 'K_6', 'K_7','K_F8'
    # 'K_F1', 'K_F5', 'K_F5', 'K_F4', 'K_F9', 'K_F6', 'K_6', 'K_7','K_F8'
    'K_1', 'K_4', 'K_7', 'K_1', 'K_5', 'K_F8', 'K_F2', 'K_F2'
    # 'K_F1', 'K_F4', 'K_F7', 'K_F1', 'K_F5', 'K_8', 'K_2', 'K_2'

]
movesSolve = [ 'K_F2', 'K_F2', 'K_8', 'K_F5', 'K_F1', 'K_F7', 'K_F4', 'K_F1']

# Call the function to execute the key moves with a delay of 1 second
# perform_key_moves(sample_moves, delay_between_moves=1)
time.sleep(10)
perform_key_moves(movesSolve, delay_between_moves=1)


# import time
# from pyKey import pressKey, releaseKey, press
#
# # Define the array of key moves
# sample_moves = [
# # 'K_F4', 'K_7', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4'
# # 'K_F2', 'K_F2', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5', 'K_F5'
# # 'K_1', 'K_1', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4', 'K_F4'
# # shiffleer
# # 'K_5', 'K_F5', 'K_F1', 'K_F5', 'K_F8', 'K_8', 'K_F5', 'K_F4', 'K_F3', 'K_F2', 'K_F1', 'K_F9', 'K_F6', 'K_6', 'K_7'
# #solver
# 'K_F8', 'K_2', 'K_F3', 'K_F3', 'K_1', 'K_F3', 'K_5', 'K_6', 'K_1'
# ]
#
# time.sleep(5)
#
# # Press each key in the array with a 2-second delay
# for move in sample_moves:
#     trimmed_move = move[2:]  # Trim the first two characters
#     pressKey(trimmed_move)  # Press the trimmed key
#     time.sleep(1)  # Wait for 2 seconds
#     releaseKey(trimmed_move)  # Release the trimmed key
