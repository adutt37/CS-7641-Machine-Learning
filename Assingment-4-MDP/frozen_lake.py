import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from bettermdptools.algorithms.planner import Planner

def run_iteration_method(env, gamma=0.9, method="value_iteration"):
    """
    Run the specified iteration method (value_iteration, policy_iteration, or q_learning) on the FrozenLake environment.
    """
    if method == "value_iteration":
        value_function, delta_list, value_list, steps = perform_value_iteration(env, gamma)
    elif method == "policy_iteration":
        value_function, policy, delta_list, value_list, steps = policy_iteration(env, gamma)
    elif method == "q_learning":
        Q, episode_rewards = perform_q_learning_decay_epsilon(env, gamma)
        value_function = np.max(Q, axis=1)  # Derive value function from Q-table (max Q for each state)
        delta_list = [np.mean(episode_rewards)] * len(episode_rewards)  # Dummy delta list for consistency
        value_list = [np.max(value_function)] * len(episode_rewards)  # Dummy value list for consistency
        steps = len(episode_rewards)  # Total number of episodes
    else:
        raise ValueError("Invalid method. Use 'value_iteration', 'policy_iteration', or 'q_learning'.")
    
    # Reshape the value function to fit the 8x8 grid (FrozenLake8x8-v1)
    V_reshaped = np.reshape(value_function, (8, 8))
    
    return V_reshaped, delta_list, value_list, steps

def perform_value_iteration(env, gamma=1.0):
    # Initialize value function with zeros
    value_function = np.zeros(env.observation_space.n)
    
    # Set maximum iterations and convergence threshold
    max_iterations = 100000
    convergence_threshold = 1e-20
    
    # Initialize lists to store changes and values for plotting
    change_list = []
    max_value_list = []
    iteration_count = 0
    
    for iteration in range(max_iterations):
        previous_value_function = np.copy(value_function)  # Copy previous value function
        
        # Loop over all states and calculate Q-values
        for state in range(env.observation_space.n):
            action_values = []
            
            # Loop through each possible action in the current state
            for action in range(env.action_space.n):
                next_state_rewards = []
                
                # Calculate expected reward for each possible transition
                for transition in env.P[state][action]:
                    transition_probability, next_state, reward, _ = transition
                    next_state_rewards.append(
                        transition_probability * (reward + gamma * previous_value_function[next_state])
                    )
                
                action_values.append(np.sum(next_state_rewards))
            
            # Update the value of the state with the highest Q-value
            value_function[state] = max(action_values)
        
        # Calculate the change (delta) in the value function
        delta = np.sum(np.abs(previous_value_function - value_function))
        change_list.append(delta)
        max_value_list.append(np.max(value_function))  # Track the highest value across all states
        
        iteration_count += 1
        
        # Check for convergence based on delta
        if delta <= convergence_threshold:
            print(f"Value iteration converged after {iteration + 1} iterations.")
            break
    
    return value_function, change_list, max_value_list, iteration_count

def policy_evaluation(env, policy, gamma=1.0, threshold=1e-20):
    # Initialize value table with zeros
    value_function = np.zeros(env.observation_space.n)
    
    while True:
        delta = 0
        # Update value table based on the current policy
        for state in range(env.observation_space.n):
            action = policy[state]
            Q_value = 0
            # Iterate over the transition probabilities for the given action
            for transition in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = transition
                Q_value += trans_prob * (reward_prob + gamma * value_function[next_state])
            
            # Calculate the difference (delta)
            delta = max(delta, np.abs(value_function[state] - Q_value))
            value_function[state] = Q_value
        
        if delta < threshold:
            break
    
    return value_function

def policy_iteration(env, gamma=1.0):
    # Initialize the policy randomly (choose random actions initially)
    policy = np.random.choice(env.action_space.n, size=env.observation_space.n)
    
    # Store steps and deltas for plotting
    delta_list = []
    value_list = []
    steps = 0
    
    while True:
        # Evaluate the policy (calculate the value function for the current policy)
        value_function = policy_evaluation(env, policy, gamma)
        
        # Policy improvement: update the policy based on the current value table
        policy_stable = True
        for state in range(env.observation_space.n):
            old_action = policy[state]
            # Find the action that maximizes the Q-value for this state
            action_values = []
            for action in range(env.action_space.n):
                Q_value = 0
                # Iterate over the transition probabilities for the given action
                for transition in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = transition
                    Q_value += trans_prob * (reward_prob + gamma * value_function[next_state])
                action_values.append(Q_value)
            
            # Choose the action with the highest Q-value
            best_action = np.argmax(action_values)
            # If the best action is different from the old action, update the policy
            if best_action != old_action:
                policy_stable = False
            policy[state] = best_action
        
        # Calculate delta for the value function
        delta = np.sum(np.abs(value_function - policy_evaluation(env, policy, gamma)))
        delta_list.append(delta)
        value_list.append(np.max(value_function))  # Tracking the max value for plotting
        
        steps += 1
        
        # If the policy is stable, the algorithm has converged
        if policy_stable:
            print(f"Policy iteration converged after {steps} iterations.")
            break
    
    return value_function, policy, delta_list, value_list, steps

def perform_q_learning(env, gamma=0.9, alpha=0.1, epsilon=0.1, episodes=1000):
    # Initialize Q-table with zeros
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # List to store rewards for each episode for plotting
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()[0]  # Reset the environment and get the initial state
        total_reward = 0
        
        while True:
            # Choose action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)  # Exploration: choose random action
            else:
                action = np.argmax(Q[state])  # Exploitation: choose action with max Q-value
            
            # Take action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)
            
            # Update Q-value using the Q-learning formula
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
    
    return Q, episode_rewards

import numpy as np

def perform_q_learning_decay_epsilon(env, gamma=0.9, alpha=0.1, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, episodes=1000):
    # Initialize Q-table with zeros
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # List to store rewards for each episode for plotting
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()[0]  # Reset the environment and get the initial state
        total_reward = 0
        
        while True:
            # Choose action using epsilon-greedy policy with decaying epsilon
            if np.random.rand() < epsilon:
                action = np.random.choice(env.action_space.n)  # Exploration: choose random action
            else:
                action = np.argmax(Q[state])  # Exploitation: choose action with max Q-value
            
            # Take action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)
            
            # Update Q-value using the Q-learning formula
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Decay epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        episode_rewards.append(total_reward)
    
    return Q, episode_rewards


def solve_frozenlake_mdp(env_name="FrozenLake8x8-v1", gamma=0.9, iteration_method="value_iteration"):
    # Create the FrozenLake environment
    env = gym.make(env_name, render_mode=None)
    
    # Run either value iteration, policy iteration, or q-learning
    value_function, delta_list, value_list, steps = run_iteration_method(env, gamma, iteration_method)
    
    # Reshape the value function to fit the 8x8 grid (FrozenLake8x8-v1)
    V_reshaped = np.reshape(value_function, (8, 8))
    
    # Plot the state values and iteration progress
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # First plot: Time vs Steps (Delta over time)
    axs[0].plot(range(steps), delta_list)
    axs[0].set_title(f"Frozen Lake (8x8): {iteration_method} - Time vs Steps")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Delta")

    # Second plot: Value and Delta vs Steps
    axs[1].plot(range(steps), value_list, label="Value", color="blue")
    axs[1].plot(range(steps), delta_list, label="Delta", color="red", linestyle='dashed')
    axs[1].set_title(f"Frozen Lake (8x8): {iteration_method} - Value and Delta vs Steps")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Value / Delta")
    axs[1].legend()

    # Show the plot interactively
    plt.tight_layout()
    plt.show()

    # Display the optimal policy (for demonstration purposes, Q-learning returns an optimal policy from Q-table)
    if iteration_method == "q_learning":
        optimal_policy = np.argmax(value_function)  # Extract optimal policy for Q-learning
    else:
        optimal_policy = np.argmax(value_function)  # Simplified version of the optimal policy extraction
    
    print("Optimal Policy:")
    print(optimal_policy)

    return value_function, optimal_policy 

# Example usage
value_function, optimal_policy = solve_frozenlake_mdp(env_name="FrozenLake8x8-v1", gamma=0.9, iteration_method="q_learning")

# Example usage
value_function, optimal_policy = solve_frozenlake_mdp(env_name="FrozenLake8x8-v1", gamma=0.9, iteration_method="value_iteration")

# Running Policy Iteration
value_function, optimal_policy = solve_frozenlake_mdp(env_name="FrozenLake8x8-v1", gamma=0.9, iteration_method="policy_iteration")

# value_function, optimal_policy = solve_frozenlake_mdp(env_name="FrozenLake8x8-v1", gamma=0.9, iteration_method="q_learning")