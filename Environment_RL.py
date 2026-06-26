#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:52:25 2023

@author: tianyuan
"""

import numpy as np
import numpy.linalg as la

import math
import argparse
import importlib

from PhantomGenerator import PhantomGenerator

from skimage.transform import radon, iradon_sart


# def reconstruction_noise(P, proj_angles, proj_size, vol_geom, n_iter_sirt, percentage=0.0):
def forward_eval(image, theta, percentage=0.0):
    """
    :param P: images (not sure what type)
    :param proj_angles (np.ndarray): array of projection angles 
    :param proj_size: number of detector elements in a single projection [default: int(1.5*self.image_size)]
        See: https://astra-toolbox.com/docs/geom2d.html#projection-geometries
    :param vol_geom: [default: astra.create_vol_geom(self.image_size, self.image_size)]
        See: https://astra-toolbox.com/docs/geom2d.html#volume-geometry
    :param n_iter_sirt: ? [default: 150]
    :param percentage: ? [default: 0.0]
    """

    """ astra code (1.0 is for 'distance between the centers of two adjacent detector elements')
    proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, proj_angles)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    """
    sinogram = radon(image, theta=theta) # circle=False
    
    # construct the OpTomo object
    """ astra code
    W = astra.OpTomo(proj_id)
    sinogram = W * P
    sinogram = sinogram.reshape([len(proj_angles), proj_size])
    """ 
    
    n = np.random.normal(0, np.std(sinogram), (len(sinogram), len(theta))) * percentage
    
    # gauss1 = gauss.reshape(len(proj_angles), proj_size)
    
    sinogram_n = sinogram + n
    
    """ astra code <- not needed since we will not do reconstruction
    rec_sirt = W.reconstruct('SIRT_CUDA', sinogram_n, iterations=n_iter_sirt, extraOptions={'MinConstraint':0.0,'MaxConstraint':1.0})
    """
    
    return sinogram_n

def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    #print(rmse)
    return 20*math.log10(1.0/rmse)

def angle_range(N_a):
    return np.linspace(0,np.pi,N_a,False)

class env():
    
    def __init__(
        self, seed, n_images, num_angles, reward_type, image_size, action_size
    ):
        # Generate your phantoms
        gen = PhantomGenerator(seed=seed, image_size=image_size)
        self.P_all = gen.generate_mixed(n_samples=n_images)

        # Select phantom index
        self.n = np.random.randint(0,len(self.P_all))
        # print("Random instance %d" % self.n)
        self.criteria = 0
        # Total number of angles for this experiment
        self.num_angles = num_angles
        # Reward mode: forward of PNSR
        self.reward_type = reward_type
        # Image size 
        self.image_size = image_size
        # The size of action space
        self.action_size = action_size
        # Parameters for astra
        self.proj_size = int(1.5*self.image_size)
        self.vol_geom = None # astra.create_vol_geom(self.image_size, self.image_size)
        self.angles = angle_range(self.action_size)
        self.n_iter_sirt = 150
        self.init_start = 0
        self.first_step = True

 
    def step(self, action):
        
        # # The number of angles after selecting the next angle
        self.a_start += 1
            
        # Find which angle is selected and store it together with previous selected angles
        self.angle_action = self.angles[action]
        
        self.angles_seq.append(self.angle_action)
        
        # Use all selected angles to do reconstruction using SIRT as a belief state
        sinogram_n = forward_eval(self.P_all[self.n].astype('float'), theta=self.angles_seq)
        self.state = iradon_sart(sinogram_n, theta=self.angles_seq) # re-constructed image
       
        # Get reward for new state
        """ ASTRA
        if self.reward_type == "increment":
            self.reward  = self._get_reward_increm()
        elif self.reward_type == "endtoend":
            self.reward = self._get_reward_end()
        """
        bootstrap_image = radon(self.state, theta=self.angles_seq)
        reconstruction_error = la.norm(sinogram_n - bootstrap_image, ord=2)

        if self.reward_type == "PNSR":
            self.reward = self._get_reward_end()
        elif self.reward_type == "forward":
            self.reward = -reconstruction_error/la.norm(sinogram_n, ord=2)
        else:
            if self.first_step:
                print("reward_type %s unknown, defaulting to 'forward" % self.reward_type)
            self.reward = -reconstruction_error/la.norm(sinogram_n, ord=2)
        self.first_step = False

        # Calculate the total rewards
        self.total_reward += self.reward
            
        # The stop criteria depends on the number of angles; if the criteria is reached, go another round 
        if self.a_start > self.num_angles:
            # self.n = np.random.randint(0,4)
            self.a_start = self.init_start
            self.angles_seq = []
            self.done = True
         
        return np.array(self.state), self.reward, self.done, self.angle_action, self.angles_seq, self.n
    
    def reset(self):
        self.n = np.random.randint(0,len(self.P_all))
        # print("Reset random instance %d" % self.n)
       
        self.a_start = self.init_start
        
        self.curr_iteration = 0
              
        self.total_reward = 0.0
        self.angles_seq = []
        
        # initialization for action
        self.previous_reward = 0
       
        self.reward = 0
        
        self.done=False 
        #self.n = 0
        
        # set a zero matrix as the first reconstruction or belief state
        self.state = np.zeros((128,128))
        # self.criteria = 0
        
        return self.state
    
    def _get_reward_increm(self,):
        # calculate the psnr value for the current reconstruction
        self.current_reward = psnr(self.P_all[self.n], self.state)
        
        # incremental reward setting
        reward = self.current_reward - self.previous_reward
        self.previous_reward = self.current_reward
        
     
        self.previous_action=self.angles_seq[-1]

        return reward

    def _get_reward_end(self,):
        # calculate the psnr value for the current reconstruction
        self.current_reward = psnr(self.P_all[self.n], self.state)   
        # end-to-end reward setting
        if self.a_start > self.num_angles:
            reward = self.current_reward
        else:
            reward = 0
        
        return reward
