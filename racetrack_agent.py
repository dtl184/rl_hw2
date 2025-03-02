import random
import numpy as np
import json
import matplotlib.pyplot as plt

class RacetrackAgent:
    """
    Optimized Q-learning agent with an epsilon-greedy exploration strategy.
    """

    def __init__(self, alpha=0.3, gamma=0.9, epsilon=0.5, mini_epsilon=0.05, decay=0.9999):
        """
        Initializes the BaseAgent.

        Parameters:
        -----------
        goal : tuple
            The goal position (x, y).
        alpha : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon : float
            Initial exploration probability.
        mini_epsilon : float
            Minimum exploration probability.
        decay : float
            Epsilon decay factor per step.
        x_max : int
            The width of the grid (number of columns).
        y_max : int
            The height of the grid (number of rows).
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        self.action_space = [(0, 0), (0, 1), (1, 0), (1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)]
        self.num_actions = len(self.action_space)
        self.returns = {}
        self.q_function = {}
        self.policy = {}

    def check_add(self, sar):
        """
        Add sa pair to returns and q_function dicts if they do not already exist there
        """
        self.returns.setdefault(sar[0:2], []) # returns tracks the returns for sa pair
        self.q_function.setdefault(sar[0:2], 0) # q value of sa pair

    def compute_return(self, subseq):
        """
        Computes the return of the given subsequence of the full ep trajectory
    
        Args:
            subseq (list): list of state action pairs assumed to be following
                            a particular state.
        """
        sub_return = 0
        # sum discounted rewards for this subseq
        for i, sar in enumerate(subseq):
            sub_return += (self.gamma ** i) * sar[2]
    
        return sub_return
    
    def update_policy(self, traj):
        seen_states = set() # ensure each state is unique
        for sar in traj:
            state = sar[0]
            if state not in seen_states:
                q_values = {action: self.q_function[(state, action)] for (s, action) in self.q_function if s == state}
                max_q = max(q_values.values())
                best_actions = [action for action, value in q_values.items() if value == max_q]
                self.policy[state] = random.choice(best_actions) # make policy greedy wrt to q function
                seen_states.add(state)



    def learning(self, traj):
        """
        Given traj, perform MC update

        Args:
            traj (list): traj generated by current policy
        """
        for i, sar in enumerate(traj): # each is (state, action, reward)
            self.check_add(sar)
            discounted_future_return = self.compute_return(traj[i:])
            self.returns[sar[0:2]].append(discounted_future_return) # sa[0:2] is (state, action)
            self.q_function[sar[0:2]] = np.mean(self.returns[sar[0:2]]) # update q as average of received returns
        
        self.update_policy(traj)


    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state (dict): The current state of the agent.

        Returns:
            int: The chosen action index.
        """
        # First time we see a state the policy is random
        self.policy.setdefault(state, random.randint(0, len(self.action_space) - 1))

        # explore
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, len(self.action_space) - 1)
        else: # exploit
            action = self.policy[state]

        if self.epsilon > self.mini_epsilon:
            self.epsilon *= self.decay

        return action

    def visualize_policy(self, track_filename):
        """
        Visualizes the learned policy on the racetrack grid with correct orientation.

        Args:
            track_filename (str): Path to the track file.
        """
        # Load the track
        with open(track_filename, 'r') as file:
            grid = [list(map(int, line.split())) for line in file]

        # Convert to numpy array
        grid = np.array(grid)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks(np.arange(grid.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(grid.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.tick_params(which="both", size=0, labelbottom=False, labelleft=False)

        # Create a colormap
        cmap = plt.get_cmap("Greys", 4)
        ax.imshow(grid, cmap=cmap, origin="lower")  # Correct orientation

        # Overlay actions from the policy
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == 1:  # Valid track state
                    state = (x, y)
                    if state in self.policy:
                        action_index = self.policy[state]
                        ax.text(x, y, str(action_index), ha='center', va='center', fontsize=12, color='red', weight='bold')
                    else:
                        ax.text(x, y, "X", ha='center', va='center', fontsize=12, color='blue', weight='bold')  # Debug missing states

        plt.savefig('policy_visual.png')
