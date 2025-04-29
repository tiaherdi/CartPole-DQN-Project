import torch
import gym
import numpy as np
import argparse
from dqn_agent import DQNAgent

# Training function
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(state_dim, action_dim, device)

    num_episodes = 500
    target_update_freq = 10

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(500):  # Max steps per episode
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Save the trained model
    torch.save(agent.q_network.state_dict(), "../dqn_cartpole.pth")
    env.close()

# Testing function
def test():
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(state_dim, action_dim, device)
    agent.q_network.load_state_dict(torch.load("../dqn_cartpole.pth", map_location=device))
    agent.q_network.eval()

    for episode in range(5):  # Test for 5 episodes
        state, _ = env.reset()
        episode_reward = 0

        for t in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Test Episode {episode}, Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
