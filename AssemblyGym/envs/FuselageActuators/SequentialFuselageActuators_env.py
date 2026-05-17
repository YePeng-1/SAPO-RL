import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump, load
from datetime import datetime
import psutil
import os
from os import path
import time
import random
from typing import Optional
import tkinter as tk
from tkinter.messagebox import showinfo, showwarning
import gym
from gym import spaces
import copy

import scipy.linalg as la

# Files Name


train_files = [["SolutionInputDP08.npy","SolutionInputDP37.npy"],
               ["SolutionInputDP12.npy","SolutionInputDP09.npy"],
               ["SolutionInputDP11.npy","SolutionInputDP05.npy"],
               ["SolutionInputDP15.npy","SolutionInputDP00.npy"],
               ["SolutionInputDP16.npy","SolutionInputDP01.npy"],
               ["SolutionInputDP12.npy","SolutionInputDP07.npy"],
               ["SolutionInputDP06.npy","SolutionInputDP25.npy"],
               ["SolutionInputDP20.npy","SolutionInputDP37.npy"],
               ["SolutionInputDP18.npy","SolutionInputDP20.npy"],
               ["SolutionInputDP11.npy","SolutionInputDP28.npy"],
               ["SolutionInputDP29.npy","SolutionInputDP14.npy"],
               ["SolutionInputDP04.npy","SolutionInputDP23.npy"],
               ["SolutionInputDP05.npy","SolutionInputDP14.npy"],
               ["SolutionInputDP29.npy","SolutionInputDP06.npy"],
               ["SolutionInputDP32.npy","SolutionInputDP21.npy"],
               ["SolutionInputDP13.npy","SolutionInputDP24.npy"],
               ["SolutionInputDP17.npy","SolutionInputDP20.npy"],
               ["SolutionInputDP12.npy","SolutionInputDP04.npy"],
               ["SolutionInputDP14.npy","SolutionInputDP33.npy"],
               ["SolutionInputDP40.npy","SolutionInputDP08.npy"],
               ["SolutionInputDP18.npy","SolutionInputDP26.npy"],
               ["SolutionInputDP24.npy","SolutionInputDP06.npy"],
               ["SolutionInputDP30.npy","SolutionInputDP21.npy"],
               ["SolutionInputDP38.npy","SolutionInputDP13.npy"],
               ["SolutionInputDP16.npy","SolutionInputDP34.npy"],
               ["SolutionInputDP21.npy","SolutionInputDP38.npy"],
               ["SolutionInputDP21.npy","SolutionInputDP24.npy"],
               ["SolutionInputDP31.npy","SolutionInputDP25.npy"],
               ["SolutionInputDP05.npy","SolutionInputDP07.npy"],
               ["SolutionInputDP14.npy","SolutionInputDP03.npy"],
               ["SolutionInputDP13.npy","SolutionInputDP18.npy"],
               ["SolutionInputDP22.npy","SolutionInputDP06.npy"],
               ["SolutionInputDP40.npy","SolutionInputDP24.npy"],
               ["SolutionInputDP36.npy","SolutionInputDP28.npy"],
               ["SolutionInputDP38.npy","SolutionInputDP05.npy"],
               ["SolutionInputDP34.npy","SolutionInputDP08.npy"],
               ["SolutionInputDP36.npy","SolutionInputDP15.npy"],
               ["SolutionInputDP10.npy","SolutionInputDP29.npy"],
               ["SolutionInputDP31.npy","SolutionInputDP40.npy"],
               ["SolutionInputDP02.npy","SolutionInputDP38.npy"],
               ["SolutionInputDP30.npy","SolutionInputDP15.npy"],
               ["SolutionInputDP29.npy","SolutionInputDP27.npy"],
               ["SolutionInputDP40.npy","SolutionInputDP36.npy"],
               ["SolutionInputDP00.npy","SolutionInputDP18.npy"],
               ["SolutionInputDP24.npy","SolutionInputDP01.npy"],
               ["SolutionInputDP04.npy","SolutionInputDP15.npy"],
               ["SolutionInputDP36.npy","SolutionInputDP01.npy"],
               ["SolutionInputDP10.npy","SolutionInputDP21.npy"],
               ["SolutionInputDP37.npy","SolutionInputDP15.npy"],
               ["SolutionInputDP13.npy","SolutionInputDP40.npy"],
               ["SolutionInputDP09.npy","SolutionInputDP06.npy"],
               ["SolutionInputDP21.npy","SolutionInputDP27.npy"],
               ["SolutionInputDP29.npy","SolutionInputDP15.npy"],
               ["SolutionInputDP06.npy","SolutionInputDP18.npy"]]


test_files = [["SolutionInputDP54.npy","SolutionInputDP48.npy"],
               ["SolutionInputDP57.npy","SolutionInputDP43.npy"],
               ["SolutionInputDP55.npy","SolutionInputDP54.npy"],
               ["SolutionInputDP43.npy","SolutionInputDP48.npy"],
               ["SolutionInputDP58.npy","SolutionInputDP47.npy"],
               ["SolutionInputDP53.npy","SolutionInputDP52.npy"],
               ["SolutionInputDP43.npy","SolutionInputDP47.npy"],
               ["SolutionInputDP41.npy","SolutionInputDP44.npy"],
               ["SolutionInputDP43.npy","SolutionInputDP58.npy"],
               ["SolutionInputDP41.npy","SolutionInputDP46.npy"],
               ["SolutionInputDP45.npy","SolutionInputDP53.npy"],
               ["SolutionInputDP59.npy","SolutionInputDP42.npy"],
               ["SolutionInputDP48.npy","SolutionInputDP56.npy"],
               ["SolutionInputDP51.npy","SolutionInputDP53.npy"],
               ["SolutionInputDP59.npy","SolutionInputDP57.npy"],
               ["SolutionInputDP55.npy","SolutionInputDP60.npy"],
               ["SolutionInputDP42.npy","SolutionInputDP55.npy"],
               ["SolutionInputDP49.npy","SolutionInputDP55.npy"],
               ["SolutionInputDP49.npy","SolutionInputDP52.npy"],
               ["SolutionInputDP60.npy","SolutionInputDP59.npy"],
               ["SolutionInputDP47.npy","SolutionInputDP49.npy"],
               ["SolutionInputDP48.npy","SolutionInputDP56.npy"],
               ["SolutionInputDP54.npy","SolutionInputDP57.npy"],
               ["SolutionInputDP53.npy","SolutionInputDP55.npy"],
               ["SolutionInputDP53.npy","SolutionInputDP46.npy"],
               ["SolutionInputDP42.npy","SolutionInputDP54.npy"],
               ["SolutionInputDP41.npy","SolutionInputDP43.npy"],
               ["SolutionInputDP44.npy","SolutionInputDP46.npy"],
               ["SolutionInputDP49.npy","SolutionInputDP42.npy"],
               ["SolutionInputDP50.npy","SolutionInputDP60.npy"],]




from cvxopt import solvers, matrix
solvers.options['show_progress'] = False

surrogate_mdl = load(path.join(path.dirname(__file__), 'Surrogates', 'surrogate_likeDu_v22.joblib') )
# read U matrix
U_mtx = surrogate_mdl.coef_
intercept = surrogate_mdl.intercept_



def min_L_infinity_cvxopt(A, b):
    b = b.reshape(-1,1)
    f = np.zeros((A.shape[1] + 1,)).reshape(-1, 1)
    f[-1, :] = 1
    G = np.concatenate((np.concatenate((-A, -1 * np.ones((A.shape[0], 1))), axis=1),
                        np.concatenate((A, -1 * np.ones((A.shape[0], 1))), axis=1),
                        np.concatenate((np.identity(A.shape[1]), np.zeros((A.shape[1], 1))), axis=1),
                        np.concatenate((-np.identity(A.shape[1]), np.zeros((A.shape[1], 1))), axis=1)), axis=0)
    h = np.concatenate((-b, b, 1000 * np.ones((A.shape[1], 1)), 1000 * np.ones((A.shape[1], 1))), axis=0)
    f = matrix(f)
    G = matrix(G)
    h = matrix(h)
    ret = solvers.lp(f, G, h)
    x = np.array(ret['x'][:-1])
    t = np.array(ret['x'][-1])
    return x, t


class SequentialFuselageActuatorsEnv(gym.Env):
    """
    ### Description
    The goal of this environment is to minimize the shape error of a fuselage by adjusting the forces exerted
    by actuators at all of the 18 positions around the lower circumference of the part. At each time step, add a force to each actuator location
    
    ### Observation Space
    Observations consist of the current position deviation of nodes from their target positions along the edge of the fuselage

    ### Action Space
    At each time step, the force at one of the 18 actuator locations can be set to a target value.
    Forces are applied all at once for the n actuators with the largest magnitude of force specified.

    a[0-17]: continuous number between -1000 and 1000. 
    Note: Action space is normalized into the interval [-1,1] and scaled by the environment
    
    ### Rewards
    At the end of each episode, max(1-error_new/error_initial, -1) is returned. 
    
    ### Episode termination
    Episodes terminate after exactly one step (one-shot simulation)

    ```
    env = gym.make('FuselageActuators-v12')
    ```
    """
    metadata = {'render_modes': ['human'],
                'modes': ['Surrogate_Train', 'Surrogate_Test']
    }

    def __init__(self, render_mode: Optional[str] = None, env_no=0 ,n_actuators=10, max_req=0.04, mode='Surrogate_Train', end_mode = 'fix_num', port=50056, file1=None, file2=None, record=False, seed=0):
        super(SequentialFuselageActuatorsEnv, self).__init__()
        # Process keyword arguments
        self.end_mode = end_mode
        self.mode = mode
        self.render_mode = render_mode
        self.record = record
        self.port = port
        self.file1 = file1
        self.file2 = file2
        self.n_actuators = n_actuators
        self.max_req = max_req 
        random.seed(seed)
        self.initDev = None
        self.Actuator_list = []
        # self.state = np.zeros((19, 355))

        self.act_ind_set = set(range(18))
        self.env_no = env_no




        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(18)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(18+1, 354+1), dtype=np.float32)


        # Generate headers for recording
        self.h1= []
        self.h2= []
        self.h3= []
        for h in range(177):
            self.h1.append("initDev"+str(h+1))
            self.h3.append("finalDev"+str(h+1))
        for h in range(18):
            self.h2.append("Force"+str(h+1))

        return
        
        
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action[0]), err_msg

        # compute forces
        self.Actuator_list.append(action[0])
        state = np.zeros((19, 355))
        state[self.Actuator_list, -1] = 1

        _, pre_max = self._get_errors()
        # pre_max = np.max(np.abs(self.deviations))
        pre_l2 = np.linalg.norm(self.deviations)

        U_actuator = U_mtx[:, self.Actuator_list]

        forces, dev_max = min_L_infinity_cvxopt(U_actuator, -intercept-self.initDev)

        self.forces = np.zeros((18))
        self.forces[self.Actuator_list] = forces.flatten()

        self.reward = 0

        


        # Calculate y and z components of the forces from desired magnitudes
        self.displacements = (np.dot(U_actuator, forces)+intercept.reshape(-1,1)).reshape(-1,2)
        self.deviations = self._get_deviation()

        # prese_max = np.max(np.abs(self.deviations))
        _, prese_max = self._get_errors()
        # prese_l2 = np.linalg.norm(self.deviations)
        self.reward = (pre_max - prese_max)/pre_l2

        # update state
        U_sel = U_mtx[:, self.Actuator_list]
        unsel = list(self.act_ind_set - set(self.Actuator_list))
        U_unsel = U_mtx[:, unsel]
        X = np.concatenate([U_unsel, (-intercept-self.initDev).reshape(-1, 1)], axis=1)

        sol, r, rank, s = la.lstsq(U_sel, X)
        XspCml = X - np.dot(U_sel, sol)
        # sol, r, rank, s = la.lstsq(U_sel, (-intercept-self.initDev).reshape(-1, 1))
        # X[:,-1] = X[:,-1] - np.dot(U_sel, sol).reshape(-1,)
        # XspCml = X

        # self.state[:, :354] = np.zeros((19, 354))
        cml_norm = la.norm(XspCml,axis=0)
        XspCml = XspCml/(cml_norm.reshape(1,-1))

        state[-1, :354] = XspCml[:, -1]
        state[unsel, :354] = (XspCml[:, :-1]).transpose()
        info = {}

        if self.end_mode == 'fix_num':
            done = len(self.Actuator_list) == self.n_actuators 
        elif self.end_mode == 'max_req':
            done = prese_max <= self.max_req 


        if done:
            # Calculate the error
            self.error, self.maxDev = self._get_errors()
            # self.maxDev = prese_max
            # Output info
            info = {"initError": self.error_initial, "Error": self.error, "Forces": self.forces, "maxDev": self.maxDev,
                    "initMaxDev": self.maxDev_initial,'ActuatorNum':len(self.Actuator_list)}




        # self.render()

        # Record the interaction to csv file
        # if self.record and self.mode != 'Surrogate':
        #     self._record()

        return np.array(state, dtype=np.float32), self.reward, done, False, info

    def reset(self):

        # Set forces to zero
        self.forces = np.zeros(18, dtype=np.float32) 

        # Check mode
        if self.mode == 'Surrogate_Train':
            pair_id = self.env_no
            # Load precalculated nodal positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Train')
            file1 = train_files[pair_id][0]
            filepath = path.join(folder, file1)
            self.initPos = np.load(filepath)
            # Randomly select source for target positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Train')
            file2 = train_files[pair_id][1]
            filepath = path.join(folder, file2)
            self.targetPos = np.load(filepath)
        
        elif self.mode == 'Surrogate_Test':
            pair_id = self.env_no
            # Load precalculated nodal positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Test')
            file1 = test_files[pair_id][0]
            filepath = path.join(folder, file1)
            self.initPos = np.load(filepath)
            # Randomly select source for target positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Test')
            file2 = test_files[pair_id][1]
            filepath = path.join(folder, file2)
            self.targetPos = np.load(filepath)

        # Get deviations
        self.displacements = np.zeros((177, 2))
        self.deviations = self._get_deviation()
        self.initDev = self.deviations


        # Initialize the error
        self.error, self.maxDev = self._get_errors()
        self.error_best = self.error
        self.error_old = self.error
        self.error_initial = self.error
        self.maxDev_old = self.maxDev
        self.maxDev_initial = np.max(np.abs(self.initDev))

        # initial state
        U_unsel = U_mtx
        X = np.concatenate([U_unsel, (-intercept - self.initDev).reshape(-1, 1)], axis=1)
        XspCml = X
        state = np.zeros((19, 355))

        state[:,:354] = XspCml.transpose()
        self.Actuator_list = []
        return np.array(state, dtype=np.float32), {} # reward, done, info can't be included

    def render(self, mode='human'):
        print("Forces=", self.forces)
        print("Error=", self.error, " ==> Reward=", self.reward)

    def _record(self):
        # Build dataframes
        df1 = pd.DataFrame(self.initDev, self.h1).T
        df2 = pd.DataFrame(self.forces, self.h2).T
        df3 = pd.DataFrame(self.finalDev, self.h3).T
        # Join them together
        df = pd.concat([df1, df2, df3], axis=1)
        # Write csv file
        df.to_csv(self.recordPath, mode='a', header=not os.path.exists(self.recordPath))

    def _get_errors(self):
        # Needs to be called after getting observations so that data is up to date
        # Calculate error relative to perfect circle with r=288
        n = len(self.deviations)
        dev_total = np.sqrt(np.square(self.deviations[:177]) + np.square(self.deviations[177:]))
        max_e = max(dev_total) # maximum error
        mae = sum(np.abs(self.deviations))/n # mean absolute error
        rmse = np.sqrt(sum((self.deviations)**2)/n) # root mean squared error
        mse = sum((self.deviations)**2)/n # mean squared error
        se = sum((self.deviations)**2) # sum of squared errors
        return rmse, max_e

    def close (self):
        return

    def _get_obs(self):
        # Get displacements from simulation
        self.displacements = self._get_displacement()
        # Calculate deviations
        self.deviations = self._get_deviation()

        obs = self.deviations
        obs = obs.flatten() #np.expand_dims(obs, -1)
        return np.array(obs, dtype=np.float32)
        
    
    def _get_deviation(self):
        '''
        Calculate the distance of the current node positions from their ideal positions
        ''' 
        finalPos = self.initPos + self.displacements
        deviations = finalPos - self.targetPos[:,0:2]
        return deviations.flatten()


    def _monitor_process(self, processName):
        flag = False
        #Iterate over the all the running process
        for proc in psutil.process_iter():
            try:
                # Check if process name contains the given name string.
                if processName.lower() in proc.name().lower():
                    flag = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return flag


    """
    # function to compute optimal forces
    # minimize t
    # -Ax - t1 <= -b
    # Ax - t1 <= b
    # x <= 1000
    # -x <= -1000
    # def min_L_infinity_cvxopt(A,b):
    #     f = np.zeros((A.shape[1] + 1,)).reshape(-1, 1)
    #     f[-1, :] = 1
    #     G = np.concatenate((np.concatenate((-A, -1 * np.ones((A.shape[0], 1))), axis=1),
    #                         np.concatenate((A, -1 * np.ones((A.shape[0], 1))), axis=1),
    #                         np.concatenate((np.identity(A.shape[1]),np.zeros((A.shape[1], 1))),axis=1),
    #                         np.concatenate((-np.identity(A.shape[1]),np.zeros((A.shape[1], 1))),axis=1)), axis=0)
    #     h = np.concatenate((-b, b, 1000*np.ones((A.shape[1],1)), 1000*np.ones((A.shape[1],1))), axis=0)
    #     f = matrix(f)
    #     G = matrix(G)
    #     h = matrix(h)
    #     ret = solvers.lp(f, G, h)
    #     x = np.array(ret['x'][:-1])
    #     t = np.array(ret['x'][-1])
    #     return x, t
    """
