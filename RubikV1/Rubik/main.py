# import magiccube
# from magiccube.cube_base import Color, Face
#
# # Create the cube in solved state
# cube = magiccube.Cube(
#     3, "YYYYYYYYYRRRRRRRRRGGGGGGGGGOOOOOOOOOBBBBBBBBBWWWWWWWWW")
#
# # Print the cube
# # print(cube)
#
# # Make some cube rotations
# cube.rotate("R' L2 U D' F B'2 R' L")
#
# # Print the cube
# # print(cube)
#
# # Create the cube with a fixed state
# cube = magiccube.Cube(
#     3, "YYYYYYGGGGGWRRRRRROOOGGWGGWYBBOOOOOORRRYBBYBBWWBWWBWWB")
#
# # Reset to the initial position
# cube.reset()
#
# # Scramble the cube
# cube.scramble()
#
# # Print the move history
# # print("History: ", cube.history())
#
# # Print the moves to reverse the cube history
# cube.rotate(cube.reverse_history())
#
# # Check that the cube is done
# assert cube.is_done()
#
# # Print the cube
# # print("Solved Cube")
# # print(cube)
#
#
# print(cube.get_face(Face.L))
# print(cube.get_face(Face.R))
# print(cube.get_face(Face.F))
# print(cube.get_face(Face.B))
# print(cube.get_face(Face.U))
# print(cube.get_face(Face.D))
#
# # print(cube)
# # cube.rotate('L')
# # print("L")
# #
# # print(cube.get_face(Face.L))
# # print(cube.get_face(Face.R))
# # print(cube.get_face(Face.F))
# # print(cube.get_face(Face.B))
# # print(cube.get_face(Face.U))
# # print(cube.get_face(Face.D))
# #
# print(cube)
# # print(cube.get_all_faces)
# #
# # print("Rotate")
# # cube.rotate('X')
# # print(cube.get_all_faces)
# # print(cube)
# # print(cube.get_piece([0,0,0]))
#
#
