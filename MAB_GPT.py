import numpy as np

class MultiArmBandit:
    def __init__(self, n_arms, dataframe):
        self.n_arms = n_arms
        self.dataframe = dataframe
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0.0 for _ in range(n_arms)]

    def select_arm(self):
        """Select an arm to pull using an epsilon-greedy strategy."""
        if np.random.rand() > EPSILON:
            # choose arm with highest value
            return np.argmax(self.values)
        else:
            # choose random arm
            return np.random.randint(self.n_arms)

    def update(self, arm, reward):
        """Update the value estimate for the selected arm."""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[arm] = new_value

    def run_algorithm(self):
        """Run the multi-arm bandit algorithm."""
        for i in range(ITERATIONS):
            arm = self.select_arm()
            row = self.dataframe.iloc[arm]
            reward = calculate_reward(row)
            self.update(arm, reward)

def calculate_reward(row):
    """Calculate the reward using a value in the row of the dataframe."""
    value = row['value']
    # Do some calculations to get the reward
    return reward

ITERATIONS = 10000
EPSILON = 0.1
bandit = MultiArmBandit(28, dataframe)
bandit.run_algorithm()


#Thompson Sampling
import numpy as np

class MultiArmBandit:
    def __init__(self, n_arms, dataframe):
        self.n_arms = n_arms
        self.dataframe = dataframe
        self.counts = [0 for _ in range(n_arms)]
        self.alpha = [1.0 for _ in range(n_arms)]
        self.beta = [1.0 for _ in range(n_arms)]

    def select_arm(self):
        """Select an arm to pull using Thompson sampling."""
        theta = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return np.argmax(theta)

    def update(self, arm, reward):
        """Update the posterior distribution for the selected arm."""
        self.counts[arm] += 1
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def run_algorithm(self):
        """Run the multi-arm bandit algorithm."""
        for i in range(ITERATIONS):
            arm = self.select_arm()
            row = self.dataframe.iloc[arm]
            reward = calculate_reward(row)
            self.update(arm, reward)

def calculate_reward(row):
    """Calculate the reward using a value in the row of the dataframe."""
    value = row['value']
    # Do some calculations to get the reward
    return reward

ITERATIONS = 10000
bandit = MultiArmBandit(28, dataframe)
bandit.run_algorithm()

### Note that in this example, calculate_reward is assumed to return binary rewards (1 or 0) instead of continuous rewards. If your rewards are continuous, you would need to use a different distribution, such as the normal distribution, and adjust the update method accordingly.