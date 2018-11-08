
# coding: utf-8

# In[89]:


import numpy as np
class meteor:
    def __init__(self, target, velocity, boundaries):
        self.rad = 0.3 * np.random.rand()
        self.target_pos = np.array(target)
        self.velocity = velocity
        self.boundaries = boundaries
        self.init_pos = np.array((self.boundaries[0][1][0] * np.random.rand(),
                           self.boundaries[0][1][1] * np.random.rand(),
                           self.boundaries[0][1][2] * np.random.rand()))
        self.current_pos = self.init_pos.reshape(3)
        self.vector = self.target_pos - self.current_pos
        self.vector_m = self.get_distance(self.current_pos)
        self.vector_time =  self.vector_m / self.velocity
        self.vector_vel = self.vector / self.vector_time

    def get_distance(self, p1):
        return np.power(np.power(p1 - self.target_pos,2).sum(),0.5)
    
    def get_timestep(self):
        self.current_pos = self.current_pos + (self.vector_vel * 1 / 50)    
        
    def reset_pos(self, target):
        self.init_pos = np.array([self.boundaries[0][1][0] * np.random.rand(),
                           self.boundaries[0][1][1] * np.random.rand(),
                           self.boundaries[0][1][2] * np.random.rand()])
        self.rad = 0.5 * np.random.rand()
        self.current_pos = self.init_pos
        self.vector = self.target_pos - self.init_pos
        self.vector_m = self.get_distance(self.init_pos)
        self.vector_time =  self.vector_m / self.velocity
        self.vector_vel = self.vector / self.vector_time
        
class env:
    def __init__(self, x=10., y=10., z=3.):
        self.lower_bound = [0., 0., 0.]
        self.upper_bound = [x, y, z]
        self.wall_boundaries = [[self.lower_bound,self.upper_bound]]
        self.obsticle_bounderies = []
        self.target = np.array([np.random.normal(loc=x/2, scale=x/24, size=1), 
                       np.random.normal(loc=y/2, scale=y/24, size=1),
                       np.random.normal(loc=1.6, scale=1.6/24, size=1)])
        self.fzz = [self.target[2] + (self.target[2] / 24.), self.target[2] - (self.target[2] / 24.)]
        self.fzd = [1.8, 1.4]
        self.fzd_max = 1.8
        self.fzd_min = 1.4
        self.fzz_max = self.target[2] + (self.target[2] / 24.)
        self.fzz_min = self.target[2] - (self.target[2] / 24.)
        self.floor_obsticles = []
        self.cieling_obsticles = []
        self.meteors = []
        self.meteors_vel = []
        self.meteors_pos = []
        self.meteors_dist = []
        self.x = x
        self.y = y
        self.z = z
        self.init_loc = self.get_init_pose()
    
    def get_init_pose(self):
        init_loc = np.zeros(6)
        init_loc[2] = self.fzz_min + (np.random.normal(loc=0.5, scale=0.5/12., size=1) * (self.fzz_max - self.fzz_min))
        delta_z = np.power(self.target[2] - init_loc[2], 2.)
        dist = self.fzd_min + (np.random.normal(loc=0.5, scale=0.5/12., size=1) * (self.fzd_max - self.fzd_min))
        dist2 = np.power(dist, 2)
        val = dist2 - delta_z
        delta_y = np.random.rand() * val
        delta_x = val - delta_y
        init_loc[0] = self.target[0] + np.power(delta_x, 0.5)
        init_loc[1] = self.target[1] + np.power(delta_y, 0.5)
        return init_loc

    
    def create_static_floor_objects(self, numObjs = 0):
        obsticles = []
        for i in range(numObjs):
            obsticle = [np.random.normal(loc=1.5, scale=1.5/6, size=1)[0], 
                                      np.random.normal(loc=0.65, scale=0.65/12., size=1)[0],
                                      np.random.normal(loc=0.40, scale=0.4/12., size=1)[0]]
            self.floor_obsticles.append(obsticle)
    def create_static_cieling_objects(self, numObjs = 0):
        obsticles = []
        for i in range(numObjs):
            obsticle = [np.random.normal(loc=0.5, scale=0.2/6., size=1)[0], 
                                      np.random.normal(loc=0.5, scale=0.2/6., size=1)[0],
                                      np.random.normal(loc=0.5, scale=0.2/6., size=1)[0]]
            self.cieling_obsticles.append(obsticle)
    def reset_env(self):
        self.wall_boundaries = [[self.lower_bound,self.upper_bound]]
        self.obsticle_bounderies = []
        self.target = np.array([np.random.normal(loc=self.x/2., scale=self.x/24., size=1)[0], 
                       np.random.normal(loc=self.y/2., scale=self.y/24., size=1)[0],
                       np.random.normal(loc=1.6, scale=1.6/24, size=1)[0]])
        self.fzz = [self.target[2] + (self.target[2] / 24.), self.target[2] - (self.target[2] / 24.)]
        self.floor_obsticles = []
        self.cieling_obsticles = []
        self.meteors = []
        self.meteors_vel = []
        self.meteors_pos = []
        self.meteors_dist = []
        self.init_loc = self.get_init_pose()
    def clear_obsticles(self):
        self.floor_obsticles = []
        self.cieling_obsticles = []
        self.obsticle_bounderies = [[self.lower_bound,self.upper_bound]]
    def place_static_obsticles(self):
        for i, obsticle in enumerate(self.floor_obsticles):
            if i == 0:
                y = 0
                max_x = self.upper_bound[0] - obsticle[0]
                x = max_x * np.random.rand()
                self.obsticle_bounderies.append([[x, y, 0.], [x + obsticle[0], y + obsticle[1], obsticle[2]]])
            if i == 1:
                max_x = self.upper_bound[0] - obsticle[0]
                x = max_x * np.random.rand()
                max_y = self.upper_bound[1] - self.obsticle_bounderies[0][1][2]
                y = self.obsticle_bounderies[0][1][2] + (max_y * np.random.rand())
                self.obsticle_bounderies.append([[x, y, 0.], [x + obsticle[0], y + obsticle[1], obsticle[2]]])
        for obsticle in self.cieling_obsticles:
            x_center = (self.upper_bound[0]/2.) + ((self.upper_bound[0] / 4.) * np.random.rand())
            y_center = (self.upper_bound[1]/2.) + ((self.upper_bound[1] / 4.) * np.random.rand())
            self.obsticle_bounderies.append([[x_center - (obsticle[0]/2.), y_center - (obsticle[1]/2.), self.upper_bound[2] - obsticle[2]],
                                            [x_center + (obsticle[0]/2.), y_center + (obsticle[1]/2.), self.upper_bound[2]]])
        
    def create_meteors(self, target, velocities):
        for velocity in velocities:
            new_meteor = meteor(target, velocity, self.obsticle_bounderies)
            self.meteors.append(new_meteor)
    
    def update_meteors(self):
        for new_meteor in self.meteors:
            new_meteor.get_timestep()
    
    def reset_meteors(self, target):
        for new_meteor in self.meteors:
            new_meteor.reset_pos(target)
    
    def clear_meteors(self):
        self.meteors = []
        self.meteors_vel = []
        self.meteors_dist = []
        
    def get_meteors_vel(self):
        self.meteors_vel = []
        for new_meteor in self.meteors:
            self.meteors_vel.append(new_meteor.vector_vel)
        return self.meteors_vel
    
    def get_meteors_pos(self):
        self.meteors_pos = []
        for new_meteor in self.meteors:
            self.meteors_pos.append(new_meteor.current_pos)
        return self.meteors_pos
    
    def get_meteors_dist(self):
        self.meteors_dist = []
        for new_meteor in self.meteors:
            self.meteors_dist.append(new_meteor.get_distance(new_meteor.current_pos))
        return self.meteors_dist
    
    def get_meteors_rads(self):
        self.meteors_rads = []
        for new_meteor in self.meteors:
            self.meteors_rads.append(new_meteor.rad)
        return self.meteors_rads
        

