import numpy as np
from physics_sim_up import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, v_d = .1,
        init_angle_velocities=None, runtime=5., target_pos=None, action_repeat = 3, monitored_meteors=3, bias=450):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = action_repeat

        self.state_size = self.action_repeat * 34
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.fzd_max = 20.
        self.fzd_min = 10.
        
        self.v_d = v_d

        self.prev_act = None
        self.prev_v = None
        self.max_angv = 5.24
        self.min_angv = -5.24
        self.target_angv = np.zeros(3)
        self.bias = bias
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.fzz_max = (self.target_pos[2] + (self.target_pos[2] / 3))[0]
        self.fzz_min = (self.target_pos[2] - (self.target_pos[2] / 3))[0]
        self.end_episode = False
        self.monitored_meteors = monitored_meteors
        self.early_stop = False
        self.init_pose = init_pose
        
    def get_reward(self, prev_state, prev_v, boundaries, wall_boundaries, meteors_ds = None, meteors_rads = None):
        self.end_episode = False
        boundary_penalty = 0.
        hit_boundary = False
        for wall in wall_boundaries:
            if not hit_boundary:
                for i in range(3):
                    if self.sim.pose[i] <= (wall[0][i] + 0.7) or self.sim.pose[i] >= (wall[1][i] - 0.7):
                        boundary_penalty = 10000.
                        hit_boundary = True
                        break
                if hit_boundary:
                    self.end_episode = True
                    break
#                
        for boundary in boundaries:
            if not hit_boundary:
                if self.sim.pose[0] > boundary[0][0] and self.sim.pose[0] < boundary[1][0] and self.sim.pose[1] > boundary[0][1] and self.sim.pose[1] < boundary[1][1] and self.sim.pose[2] > boundary[0][2] and self.sim.pose[2] < boundary[1][2]:
                    boundary_penalty = 10000.
                    hit_boundary = True
                    self.end_episode = True
                    print("Hit obsticle")
                    break
        
        if meteors_ds is not None:
            if meteors_ds <= 0.7 + meteors_rads[0]:
                self.end_episode = True
        else:
            meteors_ds = 100.
        spatial_distance = np.power(np.power(self.sim.pose[:3] - self.init_pose[:3],2).sum(),0.5)
#        reward = -np.clip(np.power(spatial_distance/13., (meteors_ds + 1.0)/13.),0., 1.)
#        reward = -np.clip(spatial_distance/0.25, 0., 1.)
        reward = np.exp(-spatial_distance/1.) #+ np.exp(-np.abs(self.sim.v).sum()))/2.
        return reward
    def get_rot(self, axis, angle):
        rot = np.zeros((3,3))
        if axis == "x":
            rot[0,0] = 1.
            rot[1,1] = np.cos(angle)
            rot[2,2] = np.cos(angle)
            rot[1,2] = -np.sin(angle)
            rot[2,1] = np.sin(angle)
        elif axis == "y":
            rot[1,1] = 1.
            rot[0,0] = np.cos(angle)
            rot[2,2] = np.cos(angle)
            rot[0,2] = np.sin(angle)
            rot[2,0] = -np.sin(angle)
        elif axis == "z":
            rot[2,2] = 1.
            rot[0,0] = np.cos(angle)
            rot[1,1] = np.cos(angle)
            rot[0,1] = -np.sin(angle)
            rot[1,0] = np.sin(angle)
        return rot
    def get_rm(self):
        R_x_alpha = self.get_rot("x", self.sim.pose[3])
        R_y_beta = self.get_rot("y", self.sim.pose[4])
        R_z_gamma = self.get_rot("z", self.sim.pose[5])
        rm = np.dot(np.dot(R_z_gamma, R_y_beta), R_x_alpha)
        rm = rm.reshape(rm.shape[0]*rm.shape[1])
        return rm
    
    def step(self, rotor_speeds, boundaries, wall_boundaries, meteor_vs = None, meteor_ds = None, meteors_rads = None):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        relative_vs = []
        prev_state = self.sim.pose[:3]
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
#            pose_all.append(self.sim.pose - self.init_pose)
            rm = self.get_rm()
            pose_all.append(rm)
            pose_all.append(self.sim.pose[:3] - self.init_pose[:3])
            pose_all.append(self.sim.v)
#            pose_all.append(self.sim.angular_v)
#            pose_all.append(self.sim.pose)
            pose_all.append(self.prev_act)
            pose_all.append(self.prev_rm)
            pose_all.append(self.prev_s)
#            pose_all.append(self.prev_s)
            pose_all.append(self.prev_v)
#            pose_all.append(self.prev_angular_v)
#            pose_all.append(self.prev_state)
#            pose_all.append(self.prev_act)
#            pose_all.append(self.sim.angular_v)
#            pose_all.append(self.target_pos)
#            if meteor_vs is not None:
#                num_meteors = len(meteor_vs)
#            else:
#                num_meteors = 0
#            for i in range(self.monitored_meteors):
#                if i < num_meteors:
#                    relative_v = meteor_vs[i]
#                else:
#                    relative_v = [0., 0., 0.]
#                pose_all.append(relative_v)
#            pose_all.append(relative_vs)
            reward += self.get_reward(prev_state, self.prev_v, boundaries, wall_boundaries, meteor_ds, meteors_rads)
        if self.end_episode:
            done = True
        self.prev_rm = rm
        self.prev_s = self.sim.pose[:3] - self.init_pose[:3]
        self.prev_state = self.sim.pose
        self.prev_act = (rotor_speeds - self.bias)/900.
        self.prev_v = self.sim.v
        self.prev_angular_v = self.sim.angular_v
        next_state = np.concatenate(pose_all)
        return next_state, reward, done, self.early_stop
    def reset(self, init_pose, target, play):
        """Reset the sim to start a new episode."""
 #       self.sim.init_pose[:3] = np.random.rand(3) * 10.0 
        if not play:
            self.target_angv = (np.random.rand(3) * (self.max_angv - self.min_angv)) + self.min_angv
        else:
            self.target_angv = np.zeros(3)
        self.init_pose = init_pose
        init_dist = np.random.rand() * 0. 
        self.z_c = init_dist * np.random.rand() * np.random.choice([-1., 1.], 1)
        z = self.z_c + init_pose[2]
        y_b = (init_dist - self.z_c) * np.random.rand()
        y = y_b + init_pose[1]
        x_a = init_dist - self.z_c - y_b
        x = x_a + init_pose[0]
        self.current_loc = np.zeros(3)
        self.current_loc[0] = x
        self.current_loc[1] = y
        self.current_loc[2] = z
        self.target_pos = np.array(target).reshape(-1)
        self.fzz_max = self.target_pos[2] + (self.target_pos[2] / 24)
        self.fzz_min = self.target_pos[2] - (self.target_pos[2] / 24)
        self.current_loc = init_pose[:3]
        self.sim.init_velocities = None
        self.sim.init_angle_velocities = None
        self.sim.init_pose[:3] = self.init_pose[:3]
        self.sim.reset()
        self.end_episode = False
        self.early_stop = False
        pose_all = []
        self.prev_act = np.zeros(4)
#        self.prev_s = np.zeros(3)
        self.prev_v = np.zeros(3)
        self.prev_angular_v = np.zeros(3)
        self.prev_state = self.sim.pose
        
#        pose_all.append(self.sim.pose - self.init_pose)
        rm = self.get_rm()
        pose_all.append(rm)
        pose_all.append(self.sim.init_pose[:3] - self.init_pose[:3]) # distance to init 3
        pose_all.append(np.zeros(3)) # velocity 3
#        pose_all.append(np.zeros(3)) # angular velocity 3
        pose_all.append(np.zeros(4))
        pose_all.append(np.zeros(9))
#        pose_all.append(np.zeros(3))
        pose_all.append(np.zeros(3))
        pose_all.append(np.zeros(3))
#        pose_all.append(self.sim.pose) # current position 6
#        pose_all.append(self.prev_act) # 4
#        pose_all.append(self.prev_s) # 3
#        pose_all.append(self.prev_v) # 3
#        pose_all.append(self.prev_angular_v) # 3
#        pose_all.append(self.sim.pose) # 6
        
#        pose_all.append(self.prev_act)
#        pose_all.append(np.zeros(3))
#        pose_all.append(np.zeros(3))
#        pose_all.append(self.prev_act)
#        pose_all.append(self.sim.angular_v) 
#        pose_all.append(self.target_pos)
#        for i in range(self.monitored_meteors):
#            pose_all.append([0., 0., 0.])
        self.prev_rm = rm
        self.prev_s = self.sim.init_pose[:3] - self.init_pose[:3]
#        self.prev_s = np.array(pose_all[:18]).reshape(18)
        self.prev_v = np.zeros(3)
        self.prev_angular_v = np.zeros(3)
        state = np.concatenate(pose_all * self.action_repeat) 
        return state