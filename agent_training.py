env = InventoryEnv()
agent = DQLAgent(state_size=2, action_size=100)
episodes = 1000
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 2])
    for time in range(500):
        action = agent.act(state)
        next_state, reward = env.step(action)
        agent.remember(state, action, reward, next_state)
        state = next_state
        if done:
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
