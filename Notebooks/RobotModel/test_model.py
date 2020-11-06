# -*- coding: utf-8 -*-
"""
==============================================================================
Neural Network for Collision Prediction
    
    Functions:
        
        get_network_params
        goal_seeking

This file contains functions for evaluating our trained model.

"""
#-----------------------------------------------------------------------------
# Imports
import pickle
import torch
import numpy as np
import numpy.linalg as la
import simulation as sim
from controls import Seek
from network import NeuralNetwork

#-----------------------------------------------------------------------------
# Functions
def get_network_param(sim_env, action, scaler):
    # get sensor measurements
    sensor_readings = sim_env.raycasting()
    
    # initialize network params
    network_param = np.append(sensor_readings, [action, 0])
    
    # reshape data
    network_param = scaler.transform(network_param.reshape(1,-1))
    network_param = network_param.flatten()[:-1]
    
    # convert to 2D tensor
    network_param = torch.as_tensor(network_param, dtype=torch.float32)
    network_param = network_param.reshape(1, -1)
    
    return network_param

def goal_seeking(goals_to_reach):
    # initialize sim
    sim_env = sim.Environment()
    
    # constant
    action_repeat = 100
    
    # determine orientation to goal
    steering_behavior = Seek(sim_env.goal_body.position)
    
    # load model
    model = NeuralNetwork()
    model.load_state_dict(torch.load('saved/saved_model.pkl'))
    model.eval()
    
    # load normalization parameters
    scaler = pickle.load(open('saved/scaler.pkl', 'rb'))
    
    # iterate until goals are reached
    goals_reached = 0
    while goals_reached < goals_to_reach:
        
        # determine position to goal
        seek_vector = sim_env.goal_body.position - sim_env.robot.body.position
        
        # reached goal
        if la.norm(seek_vector) < 50:
            # make new goal
            sim_env.move_goal()
            
            # update
            steering_behavior.update_goal(sim_env.goal_body.position)
            
            # increment
            goals_reached += 1
            continue
        
        # list of actions
        action_space = np.arange(-5, 6)
        actions_available = []
        
        # iterate over possible actions
        for action in action_space:
            # get features
            network_param = get_network_param(sim_env, action, scaler)
            
            # predict collision probability
            prediction = model(network_param)
            _, prediction = torch.max(prediction, dim=1)
            
            # if chances of collision are slim
            if prediction.item() < 0.25:
                # add action to list
                actions_available.append(action)
                
        # if no possible actions are chosen
        if len(actions_available) == 0:
            # turn around
            sim_env.turn_robot_around()
            continue
        
        # get action
        action, _ = steering_behavior.get_action(sim_env.robot.body.position, sim_env.robot.body.angle)
        
        # constants
        min, closest_action = 9999, 9999
        
        # iterate over plausible actions
        for a in actions_available:
            # get delta
            diff = abs(action - a)
            
            # store best action
            if diff < min:
                min = diff
                closest_action = a
                
        # apply action
        steering_force = steering_behavior.get_steering_force(closest_action, sim_env.robot.body.angle)
        
        # simulate
        for action_timestep in range(action_repeat):
            # determine if collision occurred
            _, collision, _ = sim_env.step(steering_force)
            
            if collision:
                # start over
                steering_behavior.reset_action()
                break

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # set a few goals
    goals_to_reach = 3
    
    # run solution
    goal_seeking(goals_to_reach)

