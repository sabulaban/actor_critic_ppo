
# coding: utf-8

# In[ ]:

import numpy as np

class PolicySearch_Agent():
    def __init__(self, task, in_env, add_obsticles = False, add_meteors = False, meteor_vs = None):
        # Task (environment) information
        self.env = in_env
        if add_obsticles:
            self.env.create_static_cieling_objects(1)
            self.env.create_static_floor_objects(2)
            self.env.place_static_obsticles()
        if add_meteors:
            self.env.create_meteors(self.env.target, meteor_vs)
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        self.init_loc = self.env.init_loc
        self.reset_episode()
        self.use_meteors = add_meteors
        if self.use_meteors:
            self.env.clear_meteors()
            self.env.create_meteors(self.init_loc[:3], meteor_vs)
            self.use_meteors = True
        
    def reset_episode(self, create_meteor = False, meteor_vs = None):
        self.env.reset_env()
        self.env.create_static_floor_objects(2)
        self.env.create_static_cieling_objects(1)
        self.env.place_static_obsticles()
        self.init_loc = self.env.init_loc
        if create_meteor:
            self.env.clear_meteors()
            self.env.create_meteors(self.init_loc[:3], meteor_vs)
            self.use_meteors = True
        else:
            self.use_meteors = False
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset(self.init_loc, self.env.target, play=False)
        return state

    def step(self, reward, done, learnFlag=True):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done and learnFlag:
            self.learn()

    def step_no_learn(self, reward, done):
        self.total_reward += reward
        self.count += 1

    def act(self, state):
        # Choose action based on given state and policy
        action = np.dot(state, self.w)  # simple linear policy
        return action

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions

