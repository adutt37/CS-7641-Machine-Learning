
import numpy as np
import matplotlib.pyplot as plt

# Transition model
transition_model = {
    'Healthy': {'Harvest': 'Degraded', 'Conserve': 'Healthy'},
    'Degraded': {'Harvest': 'Healthy', 'Conserve': 'Degraded'}
}

# Reward model
reward_model = {
    ('Healthy', 'Harvest'): -10,
    ('Healthy', 'Conserve'): 0,
    ('Degraded', 'Harvest'): 5,
    ('Degraded', 'Conserve'): -5
}

# Discount factor
gamma = 0.9

# Value iteration
def value_iteration(transition_model, reward_model, gamma, num_steps):
    num_states = len(transition_model)
    V = np.zeros(num_states)

    V_values = []
    for step in range(num_steps):
        V_new = np.copy(V)
        for state in range(num_states):
            for action in transition_model:
                if transition_model[action] == state:
                    next_state = transition_model[action]
                    reward = reward_model[(state, action)]
                    V_new[state] = max(V_new[state], reward + gamma * V[next_state])
        V = V_new
        V_values.append(V.copy())

    return V_values

# Run value iteration
num_steps = 100
V_values = value_iteration(transition_model, reward_model, gamma, num_steps)

# Plot the value function for each state
plt.figure(figsize=(10, 6))
for i, state in enumerate(transition_model.keys()):
    plt.plot(range(num_steps), [V_values[j][i] for j in range(num_steps)], label=state)

plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Convergence of Value Function-value iteration')
plt.legend()
plt.show()

# Policy iteration
def policy_iteration(transition_model, reward_model, gamma, num_steps):
    num_states = len(transition_model)
    num_actions = len(transition_model[list(transition_model.keys())[0]])

    # Initialize policy
    policy = {state: np.random.choice(list(transition_model[state].keys())) for state in transition_model}

    # Initialize value function
    V = np.zeros(num_states)

    V_values = []
    for step in range(num_steps):
        # Policy evaluation
        V_new = np.zeros(num_states)
        for state in range(num_states):
            for action in transition_model:
                if transition_model[action] == state:
                    next_state = transition_model[action]
                    reward = reward_model[(state, policy[state])]
                    V_new[state] = max(V_new[state], reward + gamma * V[next_state])
        V = V_new
        V_values.append(V.copy())

        # Policy improvement
        policy_new = {}
        for state in transition_model:
            best_action = None
            best_value = float('-inf')
            for action in transition_model:
                if transition_model[action] == state:
                    next_state = transition_model[action]
                    reward = reward_model[(state, action)]
                    value = reward + gamma * V[next_state]
                    if value > best_value:
                        best_value = value
                        best_action = action
            policy_new[state] = best_action
        policy = policy_new

    return V_values, policy

# Run policy iteration
num_steps = 100
V_values, policy = policy_iteration(transition_model, reward_model, gamma, num_steps)

# Plot the value function for each state
plt.figure(figsize=(10, 6))
for i, state in enumerate(transition_model.keys()):
    plt.plot(range(num_steps), [V_values[j][i] for j in range(num_steps)], label=state)

plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Convergence of Value Function-policy iteration')
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Transition model
transition_model = {
    'Healthy': {'Harvest': 'Degraded', 'Conserve': 'Healthy'},
    'Degraded': {'Harvest': 'Healthy', 'Conserve': 'Degraded'}
}

# Reward model
reward_model = {
    ('Healthy', 'Harvest'): -10,
    ('Healthy', 'Conserve'): 0,
    ('Degraded', 'Harvest'): 5,
    ('Degraded', 'Conserve'): -5
}

# Discount factor
gamma = 0.9

# Learning rate
alpha = 0.1

# Exploration rate
epsilon = 0.1

# Q-learning
def q_learning(transition_model, reward_model, gamma, alpha, epsilon, num_episodes):
    num_states = len(transition_model)
    num_actions = len(transition_model[list(transition_model.keys())[0]])
    Q = np.zeros((num_states, num_actions))

    Q_values = []
    states = list(transition_model.keys())  # Define states here
    for episode in range(num_episodes):
        state = np.random.choice(states)
        done = False
        rewards = 0

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(list(transition_model[state].keys()))
            else:
                action = np.argmax(Q[states.index(state)])

            if action not in transition_model[state]:
                action = np.random.choice(list(transition_model[state].keys()))

            next_state = transition_model[state][action]
            reward = reward_model[(state, action)]
            Q[states.index(state), list(transition_model[state].keys()).index(action)] = (1 - alpha) * Q[states.index(state), list(transition_model[state].keys()).index(action)] + alpha * (reward + gamma * np.max(Q[states.index(next_state)]))

            state = next_state
            rewards += reward

            if state == 'Healthy':
                done = True

        Q_values.append(Q.copy())

    return Q_values

# Run Q-learning
num_episodes = 1000
Q_values = q_learning(transition_model, reward_model, gamma, alpha, epsilon, num_episodes)

# Plot the Q-values for each state-action pair
plt.figure(figsize=(10, 6))
for i, state in enumerate(transition_model.keys()):
    for j, action in enumerate(transition_model[state].keys()):
        plt.plot(range(num_episodes), [Q_values[k][i, j] for k in range(num_episodes)], label=f'State: {state}, Action: {action}')

plt.xlabel('Episode')
plt.ylabel('Q-value')
plt.title('Convergence of Q-values-Q-learning')
plt.legend()
plt.show()


