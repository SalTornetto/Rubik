import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *



vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)
edges = ((0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7), (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7))
surfaces = ((0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4), (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6))
colors = ((1, 0, 0), (0, 1, 0), (1, 0.5, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0))


class Cube():
    def __init__(self, id, N, scale):
        self.N = N
        self.scale = scale
        self.init_i = [*id]
        self.current_i = [*id]
        self.rot = [[1 if i == j else 0 for i in range(3)] for j in range(3)]

    def isAffected(self, axis, slice, dir):
        return self.current_i[axis] == slice

    def update(self, axis, slice, dir):

        if not self.isAffected(axis, slice, dir):
            return

        i, j = (axis + 1) % 3, (axis + 2) % 3
        for k in range(3):
            self.rot[k][i], self.rot[k][j] = -self.rot[k][j] * dir, self.rot[k][i] * dir

        self.current_i[i], self.current_i[j] = (
            self.current_i[j] if dir < 0 else self.N - 1 - self.current_i[j],
            self.current_i[i] if dir > 0 else self.N - 1 - self.current_i[i])

    def transformMat(self):
        gap = 2.05  # distance between the blocks
        scaleA = [[s * self.scale for s in a] for a in self.rot]
        scaleT = [(p - (self.N - 1) / 2) * gap * self.scale for p in self.current_i]
        return [*scaleA[0], 0, *scaleA[1], 0, *scaleA[2], 0, *scaleT, 1]

    def draw(self, col, surf, vert, animate, angle, axis, slice, dir):

        glPushMatrix()
        if animate and self.isAffected(axis, slice, dir):
            glRotatef(angle * dir, *[1 if i == axis else 0 for i in range(3)])
        glMultMatrixf(self.transformMat())

        glBegin(GL_QUADS)
        for i in range(len(surf)):
            glColor3fv(colors[i])
            for j in surf[i]:
                glVertex3fv(vertices[j])
        glEnd()

        # chat gpt inserted code to make lines black and thicker
        glLineWidth(8.0)  # Set line width to make the edges thicker
        glColor3f(0.0, 0.0, 0.0)  # Set color to black for edges

        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

        glPopMatrix()






class EntireCube():
    def __init__(self, N, scale):
        self.N = N
        cr = range(self.N)
        self.cubes = [Cube((x, y, z), self.N, scale) for x in cr for y in cr for z in cr]

    def update(self, axis, slice, dir):
        for i in range(len(self.cubes)):
            self.cubes[i].update(axis, slice, dir)

    def state(self):
        # # 1. Get the current position for each block
        # current_positions = [vars(cube)['current_i'] for cube in self.cubes]
        #
        # # 2. Get the rotation for each block
        # rotations = [vars(cube)['rot'] for cube in self.cubes]
        #
        # return current_positions, rotations  # Return as a tuple of arrays

        # attempt 2
        # 1. Retrieve the current position for each block
        current_positions = [cube.current_i for cube in self.cubes]

        # 2. Retrieve the rotation for each block
        rotations = [cube.rot for cube in self.cubes]

        return current_positions, rotations  # Return as a tuple of arrays



    def mainloop(self):

        rot_cube_map = {K_UP: (-1, 0), K_DOWN: (1, 0), K_LEFT: (0, -1), K_RIGHT: (0, 1)}
        rot_slice_map = {
            K_1: (0, 0, 1), K_2: (0, 1, 1), K_3: (0, 2, 1), K_4: (1, 0, 1), K_5: (1, 1, 1),
            K_6: (1, 2, 1), K_7: (2, 0, 1), K_8: (2, 1, 1), K_9: (2, 2, 1),
            K_F1: (0, 0, -1), K_F2: (0, 1, -1), K_F3: (0, 2, -1), K_F4: (1, 0, -1), K_F5: (1, 1, -1),
            K_F6: (1, 2, -1), K_F7: (2, 0, -1), K_F8: (2, 1, -1), K_F9: (2, 2, -1),
        }

        ang_x, ang_y, rot_cube = 0, 0, (0,0) #(-0.3, -0.1) #nice animation of constantly turning while moves are performed
        animate, animate_ang, animate_speed = False, 0, 5
        action = (0, 0, 0)

        # for cube in self.cubes:
        #     cube.update(*(0,0,1))
        #     cube.update(*(0, 0, 1))
        #     cube.update(*(1, 0, 1))





        # zoom = -35
        zoom = -2.5

        while True:

            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == KEYDOWN:
                    if event.key in rot_cube_map:
                        rot_cube = rot_cube_map[event.key]
                    if not animate and event.key in rot_slice_map:
                        animate, action = True, rot_slice_map[event.key]
                        # print(event.key)
                        # print(i)
                        # print("here-----------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                if event.type == KEYUP:
                    if event.key in rot_cube_map:
                        rot_cube = (0, 0)
                if event.type == pygame.MOUSEMOTION:
                    if button_down:
                        ang_x += event.rel[1]
                        ang_y += event.rel[0]
                if event.type == pygame.MOUSEBUTTONDOWN:  # scroll up to zoom out
                    if event.button == 4:
                        zoom += 0.2
                    if event.button == 5:
                        zoom -= 0.2

                mouse_buttons = pygame.mouse.get_pressed()
                button_down = mouse_buttons[0] == 1

            ang_x += rot_cube[0] * 2
            ang_y += rot_cube[1] * 2

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0, 0, zoom)
            glRotatef(ang_y, 0, 1, 0)
            glRotatef(ang_x, 1, 0, 0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if animate:
                if animate_ang >= 90:
                    # for cube in self.cubes:
                    #     cube.update(*action)
                    self.update(*action)
                    animate, animate_ang = False, 0

            for cube in self.cubes:
                cube.draw(colors, surfaces, vertices, animate, animate_ang, *action)
            if animate:
                animate_ang += animate_speed

            pygame.display.flip()
            pygame.time.wait(10)

            # print(self.cubes)

            # print(dir(self.cubes[0]))
            # size = 27 or 3^3
            # print(self.N)

            # cube_info = vars(self.cubes[0])  # Using vars() function
            print(self.state())
            # print(cube_info['current_i'], cube_info['rot'],)
            # solved = cube_info['init_i'] == cube_info['current_i'] and cube_info["rot"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            # print("SOLVED: " + str(solved))
            # works to check if an idividual piece is in the correct locaiton

            # idea now is to send the ai all 27 cubes starting i, current i, and rotation matrix and
            # give a list of moves to preform that can result in it getting back to a solved state from a random shuffle

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(.7, .7, .7, 1)#gray background

    # NewEntireCube = EntireCube(3, .1)# dimesions, scale of drawing

    # NewEntireCube.update(0, 0, 1)

    # NewEntireCube = EntireCube(10, .5)
    # NewEntireCube.mainloop()



if __name__ == '__main__':
    main()

    pygame.quit()
    quit()
