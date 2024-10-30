import numpy as np

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_space = action_space

        self.init()

    def init(self):
        """
        Initialise the class and this method is being used from when we test the agent
        so that we've decided to make it available outward
        """

        self.obs = np.zeros((self.num_steps + 1, *self.obs_shape))
        self.rewards = np.zeros((self.num_steps, 1))
        self.costs = np.zeros((self.num_steps, 1))
        self.returns = np.zeros((self.num_steps, 1))
        self.action_log_probs = np.zeros((self.num_steps, 1))
        self.skip_log_probs = np.zeros((self.num_steps, 1))
        self.skip = np.zeros((self.num_steps + 1, 1), dtype=np.long)
        self.action_shape = 1

        self.actions = np.zeros((self.num_steps, self.action_shape), dtype=np.long)
        self.masks = np.zeros((self.num_steps, 1))

        # Masks that indicate whether it's a true terminal state
        # or time limit end state

        self.step = 0
