import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("Ant-v2")
    env.reset()
    done = False
    while True:
        while not done:
            a = env.render()
            action = env.action_space.sample()
            next_state, action, r, _ = env.step(action)
