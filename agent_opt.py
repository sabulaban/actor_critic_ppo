import numpy as np
import matplotlib.pyplot as plt 
import math
from collections import deque
from actor import Actor
from critic import Critic

class Agent:
    def __init__(self, batch_size, env,
                 task, add_meteor = False,
                 c_lr = 0.001, a_lr = 0.001,
                 state_size = 9, repeat = 1,
                 action_size = 4, gamma = 0.99,
                 T = 10, action_repeat = 1,
                 K = 20, max_size = 100000, N = 1000, bias = 900., arch = [[32, 64, 32],[32, 64, 32]],
                 attitude_name = "attitude", old_attitude_name = "old_attitude",
                 actor_name = "actor", old_actor_name = "old_actor", varFactor = 0.1):

        self.actor_name = actor_name
        self.old_actor_name = old_actor_name
        self.attitude_name = attitude_name
        self.old_attitude_name = old_attitude_name
        self.state_size = state_size * action_repeat
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env
        self.task = task
        self.add_meteor = add_meteor
        self.T = T
        self.K = K
        self.N = N
        self.bias = bias
        self.max_size = max_size
        self.last_rewards = deque(maxlen=10000000)
        self.last_total_rewards = deque(maxlen=10000000)
        self.last_steps = deque(maxlen=10000000)
        self.last_aloss = deque(maxlen=10000000)
        self.last_vloss = deque(maxlen=10000000)
        self.varFactor = varFactor
        self.arch = arch
        self.agent_id = str(batch_size) + "-" + str(self.K) + "-" + str(self.gamma) + "-" + str(varFactor)
        self.actor = Actor(self.batch_size, learning_rate = a_lr,
                           state_size = self.state_size, action_size = self.action_size, fc_sizes = self.arch[0],
                           name=actor_name, s_name=actor_name, trainable = True, varFactor = self.varFactor)

        self.old_actor = Actor(self.batch_size, learning_rate = a_lr, fc_sizes = self.arch[0],
                               state_size = self.state_size, action_size = self.action_size,
                               name=old_actor_name, s_name=actor_name, trainable = False, varFactor = self.varFactor)

        self.critic = Critic(self.batch_size, learning_rate = c_lr, fc_sizes = self.arch[1],
                             state_size = self.state_size, action_size = self.action_size, name = self.attitude_name)

        self.buffer = deque(maxlen=max_size)
        self.adv_buffer = deque(maxlen=max_size)
        
    def reset_episode(self, play = False):
        self.env.reset_env()
        self.env.create_static_floor_objects(2)
        self.env.create_static_cieling_objects(1)
        self.env.place_static_obsticles()
        init_loc = self.env.init_loc
        if self.add_meteor:
            self.env.clear_meteors()
            self.env.create_meteors(init_loc[:3], self.meteor_vs)
        state = self.task.reset(init_loc, self.env.target, play)
        return state

    def update_target(self, sess):
        sess.run(self.old_actor.update_target_network_params)

    def add_experience(self, experience):
        self.buffer.append(experience)

    def clear_buffer(self):
        self.buffer.clear()

    def sample_experience(self):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=self.batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]
    
    def sample_adv_experience(self):
        idx = np.random.choice(np.arange(len(self.adv_buffer)), 
                               size=self.batch_size,
                               replace=False)
        return [self.adv_buffer[ii] for ii in idx]

        
    def act(self, state, sess, old = False):
        if old:
            tempActor = self.old_actor
        else:
            tempActor = self.actor
        action = sess.run(tempActor.actions, feed_dict = {
            tempActor.state_input: state
        })
        return action
    
    def get_dr(self, sess, experience):
        
        batch = np.array(experience)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])
        dones = np.array([each[4] for each in batch])
        
        next_state = next_states[-1].reshape(1, self.state_size)
        done = dones[-1]

        if done:
            v_s = 0.
        else:
            v_s = sess.run(self.critic.v, feed_dict = {
                self.critic.state_input: next_state
            })
            v_s = v_s.reshape(1)
        dr = []
        for r in rewards[::-1]:
            v_s = r + (self.gamma * v_s)
            dr.append(v_s)
        dr.reverse()
        dr = np.array(dr)
        return dr
    
    def get_adv(self, sess, batch):
        states = np.array([each[0] for each in batch]).reshape((self.batch_size, self.state_size))
        actions = np.array([each[1] for each in batch]).reshape((self.batch_size, self.action_size))
        drs = np.array([each[2] for each in batch]).reshape((self.batch_size, 1))

        adv = sess.run(self.critic.adv, feed_dict = {
            self.critic.state_input: states,
            self.critic.input_dr: drs
        })

        return adv.reshape((self.batch_size, 1))
    
    def optimize(self, sess, show = False):
        self.update_target(sess)
        aloss = 0
        attloss = 0
        vloss = 0
        for i in range(self.K):
            batch = np.array(self.buffer)
            adv = self.get_adv(sess, batch)
            states = np.array([each[0] for each in batch]).reshape((self.batch_size, self.state_size))
            actions = np.array([each[1] for each in batch]).reshape((self.batch_size, self.action_size))
            drs = np.array([each[2] for each in batch]).reshape((self.batch_size, 1))
            old_pi = sess.run(self.old_actor.pi, feed_dict = {
                self.old_actor.state_input: states,
                self.old_actor.input_actions: actions
            })

            _,loss = sess.run([self.actor.opt, self.actor.loss], feed_dict = {
                self.actor.state_input: states,
                self.actor.input_actions: actions,
                self.actor.input_pi_old: old_pi,
                self.actor.adv: adv
            })
            aloss += loss
        self.last_aloss.append(aloss)
        if show:
            print("A Loss: ", aloss/self.batch_size)
            print("Last A loss: ", loss)
            
        for i in range(self.K):
            batch = np.array(self.sample_adv_experience())
            states = np.array([each[0] for each in batch]).reshape((self.batch_size, self.state_size))
            drs = np.array([each[2] for each in batch]).reshape((self.batch_size, 1))

            _,loss = sess.run([self.critic.opt, self.critic.loss], feed_dict = {
                self.critic.state_input: states,
                self.critic.input_dr: drs
            })
            vloss += loss
            if loss < 0.0001:
                break
        self.last_vloss.append(vloss)
        if show:
            print("V Loss: ", vloss/self.batch_size)
            print("Last V loss: ", loss)

        
    def play_game(self, sess, add_meteor=False, metV=[20.], show = False, Final = False):
        self.add_meteor = add_meteor
        self.meteor_vs = metV
        state = self.reset_episode(play=True)
        init_state = state[:3]
        total_dist = 0
        total_v = 0
        steps = 0
        ep_reward = 0
        actions = []
        dti = []
        steps = 0
        while True:
            action = sess.run(self.actor.actions, feed_dict = {
                self.actor.state_input: state.reshape(1, self.state_size)
            })
            action = action.reshape(self.action_size)
            actions.append(action)
            dist = np.power(np.power(self.task.sim.pose - self.task.init_pose,2).sum(),0.5)
            scaled_action = ((action * 900.) + self.bias) + 0.001
            if self.action_size == 1:
                scaled_action = np.full(4, scaled_action[0])

            if self.add_meteor:
                self.env.update_meteors()
                rads = self.env.get_meteors_rads()
                m_pos = self.env.get_meteors_pos()
                m_ds = np.power(np.power(self.task.sim.pose[:3] - m_pos, 2), 0.5).sum()
                next_state, reward, done, early_stop = self.task.step(scaled_action, self.env.obsticle_bounderies,
                                                                      self.env.wall_boundaries,
                                                                      meteor_ds = m_ds, meteors_rads = rads)
            else:
                next_state, reward, done, early_stop = self.task.step(scaled_action, self.env.obsticle_bounderies,
                                                                  self.env.wall_boundaries)
            if np.any(np.isnan(next_state)):
                print("NAN state: ", state)
                print("NAN Action: ", scaled_action)
                print("NAN NExt States: ", next_state)
                break
            state = state.reshape(self.state_size)
            next_state = next_state.reshape(self.state_size)

            state = next_state
            ep_reward += reward
            steps += 1
            total_dist += np.abs(next_state[:3]).sum()
            total_v += np.abs(next_state[3:6]).sum()
            distance = np.abs(self.task.sim.pose[:3] - self.task.current_loc[:3]).sum()
            if self.add_meteor:
                dti_met = np.array(m_pos).reshape(3)
            else:
                dti_met = np.zeros(3)
            dti.append([self.task.sim.pose[0] - self.task.init_pose[0], 
                        self.task.sim.pose[1] - self.task.init_pose[1],
                        self.task.sim.pose[2] - self.task.init_pose[2],
                        reward,
                        self.task.sim.v[0],
                        self.task.sim.v[1],
                        self.task.sim.v[2],
                        self.task.sim.pose[0],
                        self.task.sim.pose[1],
                        self.task.sim.pose[2],
                        self.task.sim.pose[3],
                        self.task.sim.pose[4],
                        self.task.sim.pose[5],
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        dti_met[0],
                        dti_met[1],
                        dti_met[2],
                        dist])
            if steps >= 250:
                done = True
            if done:
                self.last_steps.append(steps)
                if show:
                    if self.add_meteor:
                        print(m_ds)
                    legend = ['x','y', 'z']
                    plt.plot(np.array(self.last_total_rewards))
                    plt.title("agent: " + self.agent_id + " rewards per episode")
                    plt.show()
                    plt.plot(np.array(self.last_rewards))
                    plt.title("agent: " + self.agent_id + " avg reward per step")
                    plt.show()
                    plt.plot(np.array(self.last_steps))
                    plt.title("agent: " + self.agent_id + " steps per test run")
                    plt.show()
                    plt.plot(np.array(self.last_aloss))
                    plt.title("agent: " + self.agent_id + " Actor loss")
                    plt.show()
                    plt.plot(np.array(self.last_vloss))
                    plt.title("agent: " + self.agent_id + " Critic loss")
                    plt.show()
                if Final:
                    plt.plot(np.array(dti)[:,7], label='x')
                    plt.plot(np.array(dti)[:,8], label='y')
                    plt.plot(np.array(dti)[:,9], label='y')
                    plt.title("Opt Agent Position")
                    plt.show()
                    plt.plot(np.array(dti)[:,4], label='x')
                    plt.plot(np.array(dti)[:,5], label='y')
                    plt.plot(np.array(dti)[:,6], label='y')
                    plt.title("Opt Agent Velocity")
                    plt.show()
                break
    def get_trajs(self, sess, add_meteor=False, metV=[20.]):
        self.add_meteor = add_meteor
        self.meteor_vs = metV
        bs = 0
        end_trajs = False
        self.clear_buffer()
        self.adv_buffer.clear()
        for j in range(self.N):
            state = self.reset_episode()
            ep_rewards = 0
            experience = []
            steps = 0
            episode_ended = False
            for t in range(self.T):
                action = sess.run(self.actor.actions, feed_dict = {
                    self.actor.state_input: state.reshape(1, self.state_size)
                })
                action = action.reshape(self.action_size)
                scaled_action = np.clip((action * 900.) + self.bias, 0.000001, 900.)
                if self.action_size == 1:
                    scaled_action = np.full(4, scaled_action[0])
                scaled_action = scaled_action #+ 0.001
                if self.add_meteor:
                    self.env.update_meteors()
                    rads = self.env.get_meteors_rads()
                    m_pos = self.env.get_meteors_pos()
                    m_ds = np.power(np.power(self.task.sim.pose[:3] - m_pos, 2), 0.5).sum()
                    next_state, reward, done, early_stop = self.task.step(scaled_action, self.env.obsticle_bounderies,
                                                                          self.env.wall_boundaries, 
                                                                          meteor_ds = m_ds, meteors_rads = rads)
                else:
                    next_state, reward, done, early_stop = self.task.step(scaled_action, self.env.obsticle_bounderies, 
                                                             self.env.wall_boundaries)
                state = state.reshape(self.state_size)
                next_state = next_state.reshape(self.state_size)
                action =  action.reshape(self.action_size)
                experience.append([state, action, reward, next_state, done])
                bs += 1
                steps += 1
                ep_rewards += reward
                if done:
                    self.last_total_rewards.append(ep_rewards)
                    self.last_rewards.append(ep_rewards/steps)
                    episode_ended = True
                    dr = self.get_dr(sess, experience)
                    if bs >= self.batch_size:
                        end_trajs = True
                    break
                if bs >= self.batch_size:
                    end_trajs = True
                    break
                state = next_state
            if not episode_ended:
                dr = self.get_dr(sess, experience)
                self.last_total_rewards.append(ep_rewards)
                self.last_rewards.append(ep_rewards/steps)
            for i, exp in enumerate(experience):
                self.buffer.append(np.array([exp[0], exp[1], dr[i]]))
                self.adv_buffer.append(np.array([exp[0], exp[1], dr[i]]))
            if end_trajs:
                break