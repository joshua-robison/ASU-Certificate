# -*- coding: utf-8 -*-
"""
==============================================================================
Neural Network for Collision Prediction
    
    Helper Functions:
        
        radians
        degrees
        angle
        vector

This file consists of helper functions for our robot simulation.

"""
#-----------------------------------------------------------------------------
# Imports
import math
import pymunkoptions
from pymunk.vec2d import Vec2d
pymunkoptions.options['debug'] = False

#-----------------------------------------------------------------------------
# Constants
PI = math.pi
PIx2 = 2 * PI
GRAVITY = 9.81

#-----------------------------------------------------------------------------
# Helper Functions
def radians(degrees):
    """This function converts degrees to radians"""
    return (2 * PI * degrees) / 360

def degrees(radians):
    """This function converts radians to degrees"""
    return (360 * radians) / PIx2

def angle(vector):
    """This function computes the angle for a vector"""
    return math.atan2(vector[1], vector[0])

def vector(angle):
    """This function computes a vector from a given angle"""
    return Vec2d(math.cos(angle), math.sin(angle))

