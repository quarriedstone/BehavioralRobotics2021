import gym
import time

if "__main__" in __name__:
    env = gym.make("CartPole-v0")
    observation = env.reset()

    done = False
    fitness = 0
    step_num = 1
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        fitness += reward
        print(f"Step: {step_num}")
        print(f"Observation vector: {observation}")
        print(f"Action vector: {action}")
        print(f"Reward: {reward}")
        print(f"Fitness: {fitness}\n")
        step_num += 1
        env.render()

    time.sleep(3)
    env.close()
