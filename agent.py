import numpy as np


class Agent:
    def __init__(self, env, sigma=0.1):
        self.env = env
        self.sigma = sigma
        self.position = (0, 0)
        self.theta = np.random.normal(0, 1, (25 * 4))

    """
    This method resets the agent's position to the initial position (0, 0)
    """

    def reset(self):
        self.position = (0, 0)

    def run_episode(self, policy=None):
        trajectory = [self.position]
        if policy:
            while self.position not in self.env.goal:
                action = policy[self.position]
                self.position = self.env.step(self.position, action)
                trajectory.append(self.position)
            return trajectory
        while self.position not in self.env.goal:
            action = self.env.get_action(self.position, self.sigma, self.theta)
            self.position = self.env.step(self.position, action)
            trajectory.append(self.position)
        return trajectory

    def get_trajectory(self, policy=None):
        self.reset()
        if policy:
            return self.run_episode(policy)
        return self.run_episode()

    def get_return(self, trajectory):
        return sum([self.env.get_reward(state) for state in trajectory])

    def get_average_gain(self, episode_count=50):
        return np.mean(
            [self.get_return(self.get_trajectory()) for _ in range(episode_count)]
        )

    def hill_search(self, trial_count=300):
        gains = [self.get_average_gain()]
        max_gain = gains[-1]
        for _ in range(trial_count):
            cur_theta = self.theta
            std_dev_matrix = self.sigma * np.eye(*self.theta.shape)
            new_theta = np.random.multivariate_normal(self.theta, std_dev_matrix)
            self.theta = new_theta
            new_gain = self.get_average_gain()
            gains.append(new_gain)
            if new_gain < max_gain:
                self.theta = cur_theta
            elif new_gain > max_gain:
                max_gain = new_gain
            print(max_gain)
        return gains

    def value_iteration(self, gamma=0.9, threshold=0.000001):
        # initialize V(s) with some random values but V(terminals) = 0
        V = np.random.rand(25)
        V[self.env.state_id[(4, 4)]] = 0
        while True:
            delta = 0
            for s in range(25):
                if s == 24:
                    continue
                state = list(self.env.state_id.keys())[s]
                v = V[s]
                V[s] = max(
                    sum(
                        [
                            p
                            * (
                                self.env.get_reward(s_prime)
                                + gamma
                                * (
                                    V[self.env.state_id[s_prime]]
                                    if s_prime in self.env.state_id
                                    else 0
                                )
                            )
                            for s_prime, p in self.env.get_possible_next_states_and_probabilities(
                                state, action
                            )
                        ]
                    )
                    for action in self.env.actions
                )
                delta = max(delta, abs(v - V[s]))
            if delta < threshold:
                break
        return V

    def get_optimal_policy(self, gamma=0.9):
        V_star = self.value_iteration()
        optimal_policy = {}
        for s in range(25):
            state = list(self.env.state_id.keys())[s]
            if state == (4, 4):
                continue
            best_action = state
            best_return = float("-inf")
            for action in self.env.actions:
                possible_next_states_and_probabilities = (
                    self.env.get_possible_next_states_and_probabilities(state, action)
                )
                expected_return = sum(
                    [
                        p
                        * (
                            self.env.get_reward(s_prime)
                            + gamma
                            * (
                                V_star[self.env.state_id[s_prime]]
                                if s_prime in self.env.state_id
                                else 0
                            )
                        )
                        for s_prime, p in possible_next_states_and_probabilities
                    ]
                )
                if expected_return > best_return:
                    best_return = expected_return
                    best_action = action
            optimal_policy[state] = best_action
        return optimal_policy
