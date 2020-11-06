# -*- coding: utf-8 -*-
"""
==============================================================================
Neural Network for Collision Prediction
    
    Function:
        
        collect_simulation_data

This file is for generating data for our neural network model.

"""
#-----------------------------------------------------------------------------
# Imports
import numpy as np
import simulation as sim
from controls import Wander

#-----------------------------------------------------------------------------
# Data Generation Procedure
def collect_simulation_data(total_actions):
    """
    
    This function simulates the environment, moves the robot
    randomly, and collects sensor data, action data, and
    whether or not a collision occurred.
    
    """
    # initialize environment
    sim_env = sim.Environment()
    
    # robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)
    
    # parameters: sensor_readings, action, collision
    num_params = 7
    network_params = [[], [], []]
    
    # iterate over the actions
    for action_i in range(total_actions):
        # display progress
        progress = 100 * float(action_i) / total_actions
        print(f'Collecting Training Data {progress}%', end="\n", flush=True)
        
        # steering_force is used for robot control only
        action, steering_force = steering_behavior.get_action(action_i, sim_env.robot.body.angle)
        
        # store the action
        network_params[-2].append(action)
        
        # iterate actions
        for action_timestep in range(action_repeat):
            
            # first timestep
            if action_timestep == 0:
                # get sensor readings and collision result
                _, collision, sensor_readings = sim_env.step(steering_force)
            else:
                # get collision result
                _, collision, _ = sim_env.step(steering_force)
            
            # print(action)
            # print(collision, sensor_readings)
            
            # check if a collision occurred
            if collision:
                # reset
                steering_behavior.reset_action()
                
                # check if current action is very new
                if action_timestep < action_repeat * .3:
                    # share prior action that caused collision
                    network_params[-1][-1] = collision
                break

        # update network_params
        network_params[0].append(sensor_readings)
        network_params[-1].append(collision)
        
    # convert list of parameters to arrays
    sensors = np.array(network_params[0])
    actions = np.array(network_params[1]).reshape(-1, 1)
    results = np.array(network_params[2]).reshape(-1, 1)
    
    # stack arrays
    network_params = np.hstack((sensors, actions, results))
    
    # verify number of columns matches number of params
    assert network_params.shape[1] == num_params

    # save array to .csv without column titles
    # format: sensor_readings (nx5), actions (nx1), collisions (nx1)
    # saved array -> total_actions x 7
    np.savetxt('saved/practice.csv', network_params, delimiter=',')

#-----------------------------------------------------------------------------
if __name__ == '__main__':
    # create data
    total_actions = 1000
    collect_simulation_data(total_actions)

