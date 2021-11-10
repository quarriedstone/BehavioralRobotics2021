import gym
import numpy as np

# Setting seed
np.random.seed(42)


class Network:
    """
    Self-made Neural network implementation of two-layer perceptron with Evolutionary update.
    Made specifically to work in Gym environment
    """

    def __init__(self, env, n_hidden, p_variance=0.1):
        self.p_variance = p_variance
        self.env = env
        self.n_hidden = n_hidden

        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.box.Box) \
            else env.action_space.n

        self._params = Network.gen_rand_params(self.n_inputs, self.n_outputs, n_hidden, p_variance)

    @staticmethod
    def gen_rand_params(n_inputs, n_outputs, n_hidden, p_variance):
        # Initializing params for two-layer perceptron
        w1 = np.random.randn(n_hidden, n_inputs) * p_variance  # first connection layer
        w2 = np.random.randn(n_outputs, n_hidden) * p_variance  # second connection layer
        b1 = np.zeros(shape=(n_hidden, 1))  # bias internal neurons
        b2 = np.zeros(shape=(n_outputs, 1))  # bias motor neurons

        return [(w1, b1), (w2, b2)]

    def propagate(self, obs):
        obs.resize(self.n_inputs, 1)

        # Propagating through network
        (w1, b1), (w2, b2) = self._params
        z1 = np.dot(w1, obs) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = np.tanh(z2)

        # Selecting action
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            action = a2
        else:
            action = np.argmax(a2)
        return action

    def evaluate(self, n_episodes, render=False):
        fitness = 0.0
        for e in range(n_episodes):
            observation = self.env.reset()
            done = False
            while not done:
                action = self.propagate(observation)
                observation, reward, done, info = self.env.step(action)
                fitness += reward
                if render:
                    self.env.render()
        return fitness/n_episodes

    def get_params(self):
        return self._params

    def set_params(self, params):
        self._params = params


if "__main__" in __name__:
    env = gym.make('CartPole-v0')
    network = Network(env, 5)
    n_episodes = 10

    fitness_av = network.evaluate(n_episodes)
    print(f"Fitness average: {fitness_av}")

    env.close()
