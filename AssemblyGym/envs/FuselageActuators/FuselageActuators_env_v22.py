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



class FuselageActuatorsEnv(gym.Env):
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

    def __init__(self, render_mode: Optional[str] = None, env_no=0, n_actuators=10, mode='Surrogate_Train', port=50056, file1=None, file2=None, record=False, seed=0):
        super(FuselageActuatorsEnv, self).__init__()
        # Process keyword arguments
        self.mode = mode
        self.render_mode = render_mode
        self.record = record
        self.port = port
        self.file1 = file1
        self.file2 = file2
        self.n_actuators = n_actuators
        self.surrogate = None
        random.seed(seed)
        # if self.mode=="File":
        #     assert is_instance(file1, str), "Must specify a file path for the starting positions"
        #     assert is_instance(file2, str), "Must specify a file path for the target positions"

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(-1, 1, shape = (18,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(177*2,), dtype=np.float32) # 177 deviations of nodal positions from their targets

        self.surrogate = load(path.join(path.dirname(__file__), 'Surrogates', 'surrogate_likeDu_v22.joblib') )

        # Generate headers for recording
        self.h1= []
        self.h2= []
        self.h3= []
        for h in range(177):
            self.h1.append("initDev"+str(h+1))
            self.h3.append("finalDev"+str(h+1))
        for h in range(18):
            self.h2.append("Force"+str(h+1))
        
        # Record file location
        timestamp = datetime.now()
        timestampStr = timestamp.strftime("%Y%m%d-%H%M")
        folder = path.join(path.dirname(__file__), 'Recordings', 'FuselageActuators-v22', self.mode) 
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.recordPath= path.join(folder, timestampStr+".csv")
        self.env_no = env_no

        self.initPos = None
        self.displacements = None
        
        
    def step(self, action):
        # Pick the ten largest forces, keep others at zero
        n = self.n_actuators
        # idx = (-abs(action)).argsort()[:n]
        # self.forces[idx] += action[idx]*1000 # Action space (-1,1) scaled to (-1000lb, 1000lb)

        self.forces += action*1000 # Action space (-1,1) scaled to (-1000lb, 1000lb)
        idx = (abs(action)).argsort()[:18-n]
        
        self.forces[idx] = 0
        # print(self.forces)
        
        # Calculate y and z components of the forces from desired magnitudes
        angles = np.linspace(12, -192, 18)
        self.forces_Y = self.forces*np.cos(np.deg2rad(angles))
        self.forces_Z = self.forces*np.sin(np.deg2rad(angles))
        # Predict deviations from surrogate model
        u = self.surrogate.predict(np.expand_dims(self.forces, axis=0)).flatten()
        self.displacements = u.reshape((-1,2))

        # Track 
        p_init = self.initPos[:,0:2].flatten()
        p_final = p_init + u
        p_target =self.targetPos[:,0:2].flatten()
        self.deviations = p_final - p_target
        # Assemble observation
        obs = self.deviations
        obs = obs.flatten()
        
        # Calculate the error
        self.error, self.maxDev = self._get_errors()
        
        # Terminate after one time step
        done = True

        # Calcualate the reward
        self.reward = max((1-self.error/self.error_old), -1)
        self.error_old=self.error
        self.maxDev_old=self.maxDev

        # Output info    
        info = {"initError":self.error_initial, "Error":self.error, "Forces":self.forces, "maxDev": self.maxDev, "initMaxDev": self.maxDev_initial}
        # self.render()

        # Record the interaction to csv file
        if self.record and self.mode != 'Surrogate':
            self._record()

        return np.array(obs, dtype=np.float32), self.reward, done, False, info

    def reset(self):
        # print("Resetting the environment")

        # Set forces to zero
        self.forces = np.zeros(18, dtype=np.float32) 


        # File mode
        if self.mode == 'Surrogate_Train':
            # Load precalculated nodal positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Train') 
            # file1 = random.choice(os.listdir(folder))
            file1 = train_files[self.env_no][0]
            filepath = path.join(folder, file1)
            self.initPos = np.load(filepath)
            self.displacements = np.zeros((177,2))
            # Randomly select source for target positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Train') 
            # file2 = random.choice(os.listdir(folder))
            file2 = train_files[self.env_no][1]
            while file1 == file2:   # make sure files are not the same
                file2 = random.choice(os.listdir(folder))
            filepath = path.join(folder, file2)
            self.targetPos = np.load(filepath)  
            # Print info
            # print("Initial shape from", file1.split('.')[0])
            # print("Target shape from", file2.split('.')[0])
        elif self.mode == 'Surrogate_Test':
            # Load precalculated nodal positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Test') 
            # file1 = random.choice(os.listdir(folder))
            file1 = test_files[self.env_no][0]
            filepath = path.join(folder, file1)
            self.initPos = np.load(filepath)
            self.displacements = np.zeros((177,2))
            # Randomly select source for target positions
            folder = path.join(path.dirname(__file__), 'Shapes', 'Test') 
            # file2 = random.choice(os.listdir(folder))
            file2 = test_files[self.env_no][1]
            filepath = path.join(folder, file2)
            self.targetPos = np.load(filepath)  
            # Print info
            # print("Initial shape from", file1.split('.')[0])
            # print("Target shape from", file2.split('.')[0])

        # Get deviations
        self.deviations = self._get_deviation()

        # Assemble observation
        obs = self.deviations
        obs = obs.flatten()

        # Initialize the error
        self.error, self.maxDev = self._get_errors()
        self.error_best = self.error
        self.error_old = self.error
        self.error_initial = self.error
        self.maxDev_old = self.maxDev
        self.maxDev_initial = self.maxDev

        # Zero reward
        self.reward = 0
        
        # Display initial error
        # print("Initial Error =", self.error)

        info = {"initError": self.error_initial, "initDev": self.deviations, "initMaxDev": self.maxDev_initial}

        return np.array(obs, dtype=np.float32), info # reward, done, info can't be included

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

