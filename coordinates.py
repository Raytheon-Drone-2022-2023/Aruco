import numpy as np
import math as m

def calculateArucoCoordinates(pitch, roll, yaw, x, y, z, imageX, imageY):
    f = 3.60
    Rx = np.matrix([[1, 0,              0           ],
                    [0, m.cos(pitch),   -m.sin(pitch)],
                    [0, m.sin(pitch),   m.cos(pitch)]])
    
    Ry = np.matrix([[m.cos(roll),  0, m.sin(roll)],
                    [0           , 1, 0          ],
                    [-m.sin(roll), 0, m.cos(roll)]])
    
    Rz = np.matrix([[m.cos(yaw), -m.sin(yaw), 0 ],
                    [m.sin(yaw), m.cos(yaw) , 0 ],
                    [0         , 0          , 1 ]])
    
    A = Rx @ Ry @ Rz
    denom = ((A[0, 2] * x) + (A[1, 2] * y) - (A[2, 2] * f))
    
    X = -z * (((A[0,0] * imageX) + (A[1, 0] * imageY) - (A[2, 0] * f)) / denom) + x
    Y = -z * (((A[0,1] * imageX) + (A[1, 1] * imageY) - (A[2, 1] * f)) / denom) + y
    
    return X, Y

print(calculateArucoCoordinates(0, 0, 0, 100, 100, 100, 25, 75))