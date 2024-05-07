import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from RubikCubeDemo1 import EntireCube
import pyKey
from pyKey import pressKey, releaseKey, press, sendSequence, showKeys



def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(.7, .7, .7, 1)#gray background

    NewCube = EntireCube(3, .1)

    NewCube.mainloop()


if __name__ == '__main__':
    main()

    pygame.quit()
    quit()
