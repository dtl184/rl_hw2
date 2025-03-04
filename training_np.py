import argparse
import random
import cupy as cp
import json
import matplotlib.pyplot as plt
import time
import logging
import os
from agent_np import RacetrackAgent


logging.basicConfig(
    filename="racetrack_training.log",
    filemode='w',
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

#logging.debug('Starting Debugger')
#import ipdb
#ipdb.set_trace()

def plot_experiment_results(all_experiment_results, save_path="experiment_results_plot.png"):
    """Plots the proportion of successful episodes across experiments."""
    goal_reach_proportion = cp.mean(all_experiment_results, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(goal_reach_proportion) + 1), goal_reach_proportion, label="Proportion of Successful Episodes", color="blue")
    plt.xlabel("Episode Number")
    plt.ylabel("Proportion of Episodes Reaching Goal")
    plt.title("Goal Reaching Proportion Across Experiments")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Experiment results plot saved to {save_path}")

def load_racetrack(filename):
    """Loads the racetrack from a file into a NumPy array."""
    with open(filename, 'r') as file:
        grid = cp.array([list(map(int, line.split())) for line in file], dtype=cp.int64)
    return cp.flipud(grid)

def get_grid_val(track, x, y):
    """Returns the state at coordinates (x, y)."""
    if 0 <= y < track.shape[0] and 0 <= x < track.shape[1]:
        return track[y, x]
    return 0

def update_state(track, state, action):
    """Updates the state given the current state and action."""
    vel_x, vel_y = state[2], state[3]
    vel_x_accel, vel_y_accel = action
    
    if cp.random.rand() < 0.1:
        vel_x_accel, vel_y_accel = 0, 0
    
    new_vel_x = cp.clip(vel_x + vel_x_accel, 0, 4)
    new_vel_y = cp.clip(vel_y + vel_y_accel, 0, 4)
    
    new_x_pos = state[0] + new_vel_x
    new_y_pos = state[1] + new_vel_y
    
    grid_val = get_grid_val(track, new_x_pos, new_y_pos)
    
    if grid_val == 0:
        return True, cp.array([random.choice([3, 4, 5, 6, 7, 8]), 0, 0, 0], dtype=cp.int64)
    
    return False, cp.array([new_x_pos, new_y_pos, new_vel_x, new_vel_y], dtype=cp.int64)

def calculate_reward(track, next_state):
    """Calculates reward based on the state."""
    if next_state is None:
        return False, -5
    
    x, y = next_state[:2]
    grid_val = get_grid_val(track, x, y)
    
    if grid_val == 3:
        return True, 5
    return False, -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experiments", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--episode_length", type=int, default=300)
    args = parser.parse_args()
    
    logging.debug('Loading Racetrack...')
    track = load_racetrack('track1.txt')
    logging.debug('Done!')
    
    all_experiment_results = cp.zeros((args.num_experiments, args.num_episodes), dtype=cp.int32)
    total_training_time = 0
    
    for experiment in range(args.num_experiments):
        logging.info(f"Starting Experiment {experiment + 1}/{args.num_experiments}")
        agent = RacetrackAgent()
        experiment_start_time = time.time()
        
        for episode in range(args.num_episodes):
            episode_start_time = time.time()
            
            state = cp.array([random.choice([3, 4, 5, 6, 7, 8]), 0, 0, 0], dtype=cp.int64)
            cur_ep_return = 0
            goal_reached_flag = 0
            traj = []
            
            for _ in range(args.episode_length):
                action_index = agent.choose_action(state)
                out_of_bounds, next_state = update_state(track, state, agent.action_space[action_index])
                
                goal_reached, reward = calculate_reward(track, next_state)
                cur_ep_return += reward
                
                if goal_reached:
                    goal_reached_flag = 1
                    break 
                
                traj.append((state.copy(), action_index, reward))
                state = next_state
            
            agent.learning(traj)
            all_experiment_results[experiment, episode] = goal_reached_flag
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            total_training_time += episode_duration
            
            logging.info(f"Experiment {experiment + 1}, Episode {episode + 1}, Goal Reached: {goal_reached_flag}")
        
        experiment_end_time = time.time()
        logging.info(f"Experiment {experiment + 1} completed in {experiment_end_time - experiment_start_time:.2f} seconds")
    
    average_time_per_experiment = total_training_time / args.num_experiments
    average_time_per_episode = total_training_time / (args.num_experiments * args.num_episodes)
    
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"Average Training Time per Experiment: {average_time_per_experiment:.2f} seconds")
    print(f"Average Training Time per Episode: {average_time_per_episode:.4f} seconds")
    
    plot_experiment_results(all_experiment_results)
    
if __name__ == "__main__":
    main()
