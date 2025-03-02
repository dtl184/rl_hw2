import argparse
import random
import socket
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from racetrack_agent import RacetrackAgent

import os

def plot_experiment_results(all_experiment_results, save_path="experiment_results_plot.png"):
    """
    Plots the proportion of successful episodes across experiments directly from the results array.

    Args:
        all_experiment_results (numpy.ndarray): 2D array of shape (num_experiments, num_episodes) 
                                                containing whether the goal was reached (1) or not (0).
        save_path (str): File path to save the plot.
    """
    # Compute the mean success rate across experiments for each episode
    goal_reach_proportion = np.mean(all_experiment_results, axis=0)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(goal_reach_proportion) + 1), goal_reach_proportion, label="Proportion of Successful Episodes", color="blue")
    plt.xlabel("Episode Number")
    plt.ylabel("Proportion of Episodes Reaching Goal")
    plt.title("Goal Reaching Proportion Across Experiments")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

    print(f"Experiment results plot saved to {save_path}")

def calculate_reward(track, next_state):
    """
    Calculate reward based on distance to the goal.
    """
    if next_state == None:
        return False, -1
    

    x, y = next_state[0:2]
    reached = False
    reward = -1

    grid_val = get_grid_val(track, x, y)

    if grid_val == 3: # goal state
        reward = 5
        reached = True

    return reached, reward


def load_racetrack(filename):
    """Loads the racetrack from a file into a 2D list and returns it."""
    with open(filename, 'r') as file:
        grid = [list(map(int, line.split())) for line in file]
    
    # Flip the grid vertically so (0,0) is at the bottom-left
    grid.reverse()
    return grid


def get_grid_val(grid, x, y):
    """Returns the state at coordinates (x, y), assuming (0,0) is bottom-left."""
    max_y = len(grid) - 1
    max_x = len(grid[0]) - 1 if grid else -1
    
    if 0 <= y <= max_y and 0 <= x <= max_x:
        return grid[y][x]
    else:
        return 0
    
def update_state(track, state, action):
    vel_x, vel_y = state[2:]

    vel_x_accel, vel_y_accel = action[0], action[1]

    # 0.1 chance that velocity increments are 0
    if np.random.uniform(0, 1) < 0.1:
        vel_x_accel, vel_y_accel = 0, 0


    new_vel_x, new_vel_y = vel_x + vel_x_accel, vel_y + vel_y_accel

    # The velocity components cannot both be zero, neither can be > 4 or < 0
    # If they are reset to prev velocity
    if (new_vel_x == 0 and new_vel_y == 0) or new_vel_x > 4 or new_vel_y > 4 or new_vel_x < 0 or new_vel_y < 0:
        new_vel_x, new_vel_y = vel_x, vel_y

    x_pos, y_pos = state[0:2]

    new_x_pos = x_pos + new_vel_x
    new_y_pos = y_pos + new_vel_y


    grid_val = get_grid_val(track, new_x_pos, new_y_pos)

    if grid_val == 0: # out of bounds
        return True, None

    return False, (new_x_pos, new_y_pos, new_vel_x, new_vel_y) # next state




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--num_episodes", type=int, default=2500, help="Number of episodes per experiment")
    parser.add_argument("--episode_length", type=int, default=100, help="Maximum steps per episode")
    args = parser.parse_args()


    track = load_racetrack('track1.txt')


    all_experiment_results = np.zeros((args.num_experiments, args.num_episodes), dtype=int)  # Store all results

    for experiment in range(args.num_experiments):
        print(f"\nStarting Experiment {experiment + 1}/{args.num_experiments}")

        agent = RacetrackAgent()

        for episode in range(args.num_episodes):

            # state = (x, y, vel_x, vel_y)
            state = (random.choice([3, 4, 5, 6, 7, 8]), 0, 0, 0)

            cur_ep_return = 0
            cnt = 0
            goal_reached_flag = 0  # 1 if goal reached, 0 otherwise

            traj = []

            while not goal_reached_flag:
                cnt += 1
                action_index = agent.choose_action(state)
                out_of_bounds, next_state = update_state(track, state, agent.action_space[action_index])


                goal_reached, reward = calculate_reward(track, next_state)
                cur_ep_return += reward

                if goal_reached:
                    goal_reached_flag = 1
                    break 

                traj.append((state, action_index, reward))

                if cnt >= args.episode_length:
                    break

                if out_of_bounds: # reset to the start line with 0 velocity
                    state = (random.choice([3, 4, 5, 6, 7, 8]), 0, 0, 0)
                    continue
                    
                state = next_state
            
            # after episode finishes, perform MC update
            agent.learning(traj)

            all_experiment_results[experiment, episode] = goal_reached_flag
            print(f"Experiment {experiment + 1}, Episode {episode + 1}, Goal Reached: {goal_reached_flag}")

    
    plot_experiment_results(all_experiment_results)

if __name__ == "__main__":
    main()
