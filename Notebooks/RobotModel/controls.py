# -*- coding: utf-8 -*-
"""
==============================================================================
Neural Network for Collision Prediction
    
    Classes:
        
        Wander
        Seek

This file contains the wander and seek classes for our simulation.
These classes control how our Robot class moves within the environment
during the simulation.

"""
#-----------------------------------------------------------------------------
# Imports
import noise
import random
import numpy as np
import numpy.linalg as la
from helper import vector, PI

#-----------------------------------------------------------------------------
# Wander Definition
class Wander():
    """
    
    This is the wander class for moving the robot randomly.
    
    """
    def __init__(self, action_repeat):
        # action parameters
        self.action_repeat = action_repeat
        self.wander_range = 0.1 * PI
        self.max_scaler = 5
        
        # perlin noise parameters
        self.offset0, self.scale0 = random.randint(0,1000000), 250
        self.offset1, self.scale1 = random.randint(0,1000000), 2000
    
    def get_action(self, timestep_i, current_orientation, actions_checked=[]):
        # noise factor
        perlin_noise = noise.pnoise1((float(timestep_i*self.action_repeat)+self.offset0) / self.scale0)
        perlin_noise += noise.pnoise1((float(timestep_i*self.action_repeat)+self.offset1) / self.scale1)
        
        # get action
        action = int(perlin_noise * self.max_scaler)
        if action > self.max_scaler:
            action = self.max_scaler
        elif action < -self.max_scaler:
            action = -self.max_scaler
        
        action_samples = 0
        while action in actions_checked and action_samples < 50:
            # increment
            action_samples += 1
            
            # reset
            self.reset_action()
            
            # calculate noise
            perlin_noise = noise.pnoise1((float(timestep_i*self.action_repeat)+self.offset0) / self.scale0)
            perlin_noise += noise.pnoise1((float(timestep_i*self.action_repeat)+self.offset1) / self.scale1)
            
            # get action
            action = int(perlin_noise * self.max_scaler)
            if action > self.max_scaler:
                action = self.max_scaler
            elif action < -self.max_scaler:
                action = -self.max_scaler
                
        # get steering force
        steering_force = vector(action * self.wander_range + current_orientation)
        
        return action, steering_force
    
    def reset_action(self):
        self.offset0, self.offset1 = random.randint(0,1000000), random.randint(0,1000000)
        
    def get_steering_force(self, action, current_orientation):
        steering_force = vector(action * self.wander_range + current_orientation)
        
        return steering_force
    
#-----------------------------------------------------------------------------
# Seek Definition
class Seek():
    """
    
    This is the seek class for moving the robot to the goal.
    
    """
    def __init__(self, target_position):
        # goal location
        self.target_position = target_position
        self.wander_range = 0.1 * PI
        self.max_scaler = 5
        
    def update_goal(self, new_goal_pos):
        # update goal
        self.target_position = new_goal_pos
        
    def get_action(self, current_position, current_orientation):
        # vector pointing to goal
        seek_vector = self.target_position - current_position
        
        # steering vector
        steering_vector = seek_vector - vector(current_orientation)
        
        # possible actions
        action_space = np.arange(-5, 6)
        min_diff = 9999999
        min_a = 0
        
        # iterate over actions
        for a in action_space:
            # get force for action
            steering_force = vector(a * self.wander_range + current_orientation)
            diff = la.norm(steering_force - steering_vector)
            if diff <= min_diff:
                min_a = a
                min_diff = diff
                
        # get force for best action
        steering_force = vector(min_a * self.wander_range + current_orientation)
        
        return min_a, steering_force
    
    def reset_action(self):
        # nothing
        pass
    
    def get_steering_force(self, action, current_orientation):
        # get force
        steering_force = vector(action * self.wander_range + current_orientation)
        
        return steering_force

