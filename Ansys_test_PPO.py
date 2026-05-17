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
import ansys
from ansys.mapdl.core import launch_mapdl
from ansys.mapdl import reader as mapdl_reader
import matplotlib.pyplot as plt
import torch
from AssemblyGym.envs import FuselageActuators
import gym
import argparse
from distutils.util import strtobool
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn

def Show_Deviations_Max(forces_,actuatorIds_,targetPosVis_,initPosVis_,finalPosVis_,forcesActive_,forcesInactive_,init_Max_Deviation_,final_Max_Deviation_,mag_):
    # Plot
    angles = np.linspace(12, -192, 18)
    forces_Y_ = forces_ * np.cos(np.deg2rad(angles))
    forces_Z_ = forces_ * np.sin(np.deg2rad(angles))
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(11, 5))
    fig.suptitle('Fuselage Edge Deviations (magnified x%i)' %(mag_, ))
    # Plot of initial positions
    axs[0].plot(targetPosVis_[:,0], targetPosVis_[:,1], '.')
    axs[0].plot(initPosVis_[:,0], initPosVis_[:,1], '.')
    # axs[0].plot(targetPos[actuatorIds_,0], targetPos[actuatorIds_,1], 'x')
    # C= np.sqrt(initDev[0]**2+initDev[1]**2) # magnitude of displacements - use for color in quivers
    # axs[0].quiver(initPosVis_[:,0], initPosVis_[:,1], targetPos[:,0]-initPosVis_[:,0], targetPos[:,1]-initPosVis_[:,1], C, angles='xy', scale=1, units='xy')
    axs[0].axes.set_aspect('equal')
    axs[0].set_xlabel('Y [in]')
    axs[0].set_ylabel('Z [in]')
    axs[0].set_title('Initial Shape (Max Dev = %.4fin)' %init_Max_Deviation_)
    axs[0].legend(['Target shape', 'Initial shape'], loc='center')
    axs[0].set_xlim(-150, 150)
    axs[0].set_ylim(-150, 150)

    # Plot of final positions
    axs[1].plot(targetPosVis_[:,0], targetPosVis_[:,1], '.')
    axs[1].plot(finalPosVis_[:,0], finalPosVis_[:,1], '.')
    #C= np.sqrt(finalDev) # magnitude of displacements - use for color in quivers
    #axs[1].quiver(finalPosVis_[:,0], finalPosVis_[:,1], targetPos[:,0]-finalPosVis_[:,0], targetPos[:,1]-finalPosVis_[:,1], C, angles='xy', scale=1, units='xy')
    axs[1].plot(finalPosVis_[actuatorIds_[forcesInactive_],0], finalPosVis_[actuatorIds_[forcesInactive_],1], 'x', color='grey', markersize=10, markeredgewidth=2)
    axs[1].plot(finalPosVis_[actuatorIds_[forcesActive_],0], finalPosVis_[actuatorIds_[forcesActive_],1], '*', color='cyan', markersize=12)
    axs[1].quiver(finalPosVis_[actuatorIds_[forcesActive_],0], finalPosVis_[actuatorIds_[forcesActive_],1], forces_Y_[forcesActive_], forces_Z_[forcesActive_], angles='xy', scale=5, units='xy')
    axs[1].axes.set_aspect('equal')
    axs[1].set_xlabel('Y [in]')
    axs[1].set_ylabel('Z [in]')
    axs[1].set_title('Adjusted Shape (Max Dev = %.4fin)' %final_Max_Deviation_)
    first_legend = axs[1].legend(['Target shape', 'Adjusted shape'], loc='center')


    # Add the legend manually to the current Axes.
    axs[1].add_artist(first_legend)
    # Create another legend for the second line.
    axs[1].legend(['_nolegend_','_nolegend_','Unused actuators', 'Selected actuators', 'Force vectors'], loc='lower right', bbox_to_anchor=(1.55, 0.0))

    axs[1].set_xlim(-150, 150)
    axs[1].set_ylim(-150, 150)

    plt.draw()
    # plt.savefig('./Plots/ResultViz'+str(i)+'.png', dpi=300, transparent=False, bbox_inches='tight')
    plt.show()

def Show_Deviations_Error(forces_,actuatorIds_,targetPosVis_,initPosVis_,finalPosVis_,forcesActive_,forcesInactive_,initError_,finalError_,mag_):
    # Plot
    angles = np.linspace(12, -192, 18)
    forces_Y_ = forces_ * np.cos(np.deg2rad(angles))
    forces_Z_ = forces_ * np.sin(np.deg2rad(angles))
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(11, 5))
    fig.suptitle('Fuselage Edge Deviations (magnified x%i)' %(mag_, ))
    # Plot of initial positions
    axs[0].plot(targetPosVis_[:,0], targetPosVis_[:,1], '.')
    axs[0].plot(initPosVis_[:,0], initPosVis_[:,1], '.')
    # axs[0].plot(targetPos[actuatorIds_,0], targetPos[actuatorIds_,1], 'x')
    # C= np.sqrt(initDev[0]**2+initDev[1]**2) # magnitude of displacements - use for color in quivers
    # axs[0].quiver(initPosVis_[:,0], initPosVis_[:,1], targetPos[:,0]-initPosVis_[:,0], targetPos[:,1]-initPosVis_[:,1], C, angles='xy', scale=1, units='xy')
    axs[0].axes.set_aspect('equal')
    axs[0].set_xlabel('Y [in]')
    axs[0].set_ylabel('Z [in]')
    axs[0].set_title('Initial Shape (RMSE = %.3fin)' %initError_)
    axs[0].legend(['Target shape', 'Initial shape'], loc='center')
    axs[0].set_xlim(-150, 150)
    axs[0].set_ylim(-150, 150)

    # Plot of final positions
    axs[1].plot(targetPosVis_[:,0], targetPosVis_[:,1], '.')
    axs[1].plot(finalPosVis_[:,0], finalPosVis_[:,1], '.')
    #C= np.sqrt(finalDev) # magnitude of displacements - use for color in quivers
    #axs[1].quiver(finalPosVis_[:,0], finalPosVis_[:,1], targetPos[:,0]-finalPosVis_[:,0], targetPos[:,1]-finalPosVis_[:,1], C, angles='xy', scale=1, units='xy')
    axs[1].plot(finalPosVis_[actuatorIds_[forcesInactive_],0], finalPosVis_[actuatorIds_[forcesInactive_],1], 'x', color='grey', markersize=10, markeredgewidth=2)
    axs[1].plot(finalPosVis_[actuatorIds_[forcesActive_],0], finalPosVis_[actuatorIds_[forcesActive_],1], '*', color='cyan', markersize=12)
    axs[1].quiver(finalPosVis_[actuatorIds_[forcesActive_],0], finalPosVis_[actuatorIds_[forcesActive_],1], forces_Y_[forcesActive_], forces_Z_[forcesActive_], angles='xy', scale=5, units='xy')
    axs[1].axes.set_aspect('equal')
    axs[1].set_xlabel('Y [in]')
    axs[1].set_ylabel('Z [in]')
    axs[1].set_title('Adjusted Shape (RMSE = %.3fin)' %finalError_)
    first_legend = axs[1].legend(['Target shape', 'Adjusted shape'], loc='center')


    # Add the legend manually to the current Axes.
    axs[1].add_artist(first_legend)
    # Create another legend for the second line.
    axs[1].legend(['_nolegend_','_nolegend_','Unused actuators', 'Selected actuators', 'Force vectors'], loc='lower right', bbox_to_anchor=(1.55, 0.0))

    axs[1].set_xlim(-150, 150)
    axs[1].set_ylim(-150, 150)

    plt.draw()
    # plt.savefig('./Plots/ResultViz'+str(i)+'.png', dpi=300, transparent=False, bbox_inches='tight')
    plt.show()

test_files = [["SolutionInputDP54.inp","SolutionInputDP48.inp"],
               ["SolutionInputDP57.inp","SolutionInputDP43.inp"],
               ["SolutionInputDP55.inp","SolutionInputDP54.inp"],
               ["SolutionInputDP43.inp","SolutionInputDP48.inp"],
               ["SolutionInputDP58.inp","SolutionInputDP47.inp"],
               ["SolutionInputDP53.inp","SolutionInputDP52.inp"],
               ["SolutionInputDP43.inp","SolutionInputDP47.inp"],
               ["SolutionInputDP41.inp","SolutionInputDP44.inp"],
               ["SolutionInputDP43.inp","SolutionInputDP58.inp"],
               ["SolutionInputDP41.inp","SolutionInputDP46.inp"],
               ["SolutionInputDP45.inp","SolutionInputDP53.inp"],
               ["SolutionInputDP59.inp","SolutionInputDP42.inp"],
               ["SolutionInputDP48.inp","SolutionInputDP56.inp"],
               ["SolutionInputDP51.inp","SolutionInputDP53.inp"],
               ["SolutionInputDP59.inp","SolutionInputDP57.inp"],
               ["SolutionInputDP55.inp","SolutionInputDP60.inp"],
               ["SolutionInputDP42.inp","SolutionInputDP55.inp"],
               ["SolutionInputDP49.inp","SolutionInputDP55.inp"],
               ["SolutionInputDP49.inp","SolutionInputDP52.inp"],
               ["SolutionInputDP60.inp","SolutionInputDP59.inp"],
               ["SolutionInputDP47.inp","SolutionInputDP49.inp"],
               ["SolutionInputDP48.inp","SolutionInputDP56.inp"],
               ["SolutionInputDP54.inp","SolutionInputDP57.inp"],
               ["SolutionInputDP53.inp","SolutionInputDP55.inp"],
               ["SolutionInputDP53.inp","SolutionInputDP46.inp"],
               ["SolutionInputDP42.inp","SolutionInputDP54.inp"],
               ["SolutionInputDP41.inp","SolutionInputDP43.inp"],
               ["SolutionInputDP44.inp","SolutionInputDP46.inp"],
               ["SolutionInputDP49.inp","SolutionInputDP42.inp"],
               ["SolutionInputDP50.inp","SolutionInputDP60.inp"],]

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    # parser.add_argument("--wandb-project-name", type=str, default="Sequential Fuselage Actuators",
    #     help="the wandb's project name")
    # parser.add_argument("--wandb-entity", type=str, default="",
    #     help="the entity (team) of wandb's project")
    parser.add_argument("--resume", type=any, default=False,
        help="resume from a previous run")
    parser.add_argument("--resume-run-id", type=str, default=None,
        help="resume from a previous run")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="FuselageActuators-v22",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=16384000,
        help="total timesteps of the experiments")
    parser.add_argument("--memory-size", type=int, default=100000,
                        help="replay momory size")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=50,
        help="the number of parallel train game environments")
    parser.add_argument("--num-test-envs", type=int, default=30,
        help="the number of parallel test game environments")
    parser.add_argument("--num-init", type=int, default=1,
                        help="the init number of each environment")
    parser.add_argument("--num-episode", type=int, default=1,
        help="the number of episodes to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=64,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--actor-logstd", type=float, default=-5,
        help="exponent for actor_std initialization")

    # Environment specific arguments
    parser.add_argument("--n_actions", type=int, default=10,
        help="the maximum number of non-zero actions")
    parser.add_argument("--mode", type=str, default="Surrogate_Train",
        help="instantiate environment for Surrogate_Train, Surrogate_Test, Surrogate_Validate")
    parser.add_argument("--end-mode", type=str, default="fix_num",
        help="instantiate environment for fix_num, max_req")
    parser.add_argument("--record", type=bool, default=False,
        help="record interaction with environment")
    parser.add_argument("--norm-reward", type=bool, default=False,
        help="enable wrapper to normalize reward")

    parser.add_argument('--eps_start', type=float, default=0.95)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=int, default=3000)  # change accroding to number of train episodes
    parser.add_argument('--tau', type=float, default=0.001)

    parser.add_argument('--memory_size', type=int, default=100000)

    args = parser.parse_args()
    # args.batch_size = int(args.num_envs * args.num_episode * args.n_actions)
    args.batch_size = 256
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # fmt: on
    return args
args = parse_args()
def make_env(env_id, seed, idx, capture_video, run_name, n_actions, mode, record):
    def thunk():
        env = gym.make(env_id, env_no=idx, n_actuators=n_actions, mode=mode, record=record, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)#, new_step_api=False)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        if args.norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return thunk

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent_OLD(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # layers for self.actor_mean
        self.fc1 = layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.fc3 = layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01)

        self.actor_logstd = nn.Parameter(1 * torch.ones(1, np.prod(envs.action_space.shape)),
                                         requires_grad=False)  # initial action_std = exp(actor_logstd)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None, scaleStd=1):
        # Start with standard MLP
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        action_mean = torch.tanh(self.fc3(x))
        # Build action distribution
        action_logstd = self.actor_logstd  # .expand_as(action_mean)
        action_std = torch.exp(action_logstd) * scaleStd
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if action == "deterministic":
            action = action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

class Agent(nn.Module):
    def __init__(self, envs, n_actions):
        self.n_actions = n_actions
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # layers for self.actor_mean
        self.fc1 = layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.fc3 = layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01)

        self.actor_logstd = nn.Parameter(args.actor_logstd * torch.ones(1, np.prod(
            envs.single_action_space.shape)))  # initial action_std = exp(actor_logstd)

    def get_value(self, obs):
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        # Start with standard MLP
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # Use hardshrink to enforce max number of nonzero outputs
        idx = torch.argsort(abs(x))
        lambd = abs(x[0][idx[0][-(self.n_actions-1)]]).item()
        action_mean = F.hardshrink(x, lambd=lambd) # sets outputs whose magnitudes are smaller than lambda to zero
        # Build action distribution
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if action == "deterministic":
            action = action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)



class Run_Ansys:

    def __init__(self,port=50057):
        self.port = port
        # if not self._monitor_process('ansys'):
        self._launch_ansys()
        self.forces = np.zeros(18, dtype=np.float32)
        file1 = './AssemblyGym/envs/FuselageActuators/AnsysFiles/Benchmark/SolutionInputUndeformed.inp'
        # file1 = './AssemblyGym/envs/FuselageActuators/AnsysFiles/Test/SolutionInputDP41.inp'
        # Set filepath to input from environment creation
        filepath = file1

        # Parse the Ansys file
        with open(filepath, 'r') as f:
            text = f.read()
            new_text = text.split('/com,******************* SOLVE FOR LS 1 OF 1 ****************')
            self.setup_text = new_text[0]
            new_text = text.split('! *********** WB SOLVE COMMAND ***********')
            self.finish_text = new_text[1]
        self.mapdl.finish()
        self.mapdl.clear()
        log1 = self.mapdl.input_strings(self.setup_text)  # run setup
        # log2 = self._set_actuator_forces(self.mapdl, self.forces)  # apply forces
        # self.mapdl.cmsel(name='CM_FUSELAGE_EDGE')  # select nodes on the edge of the fuselage
        # self.initPos = self.mapdl.mesh.nodes[:, 1:3]  # initial positions of nodes
        # nnum = self.mapdl.mesh.nnum  # corresponding node numbers
        # self.mapdl.allsel()

        # Get initial position from Ansys
        # self._run_ansys()
        self.initPos = self._get_initPos()
        # self.displacements = self._get_displacement()
        # self.deviations = self._get_deviation()
        # self.initDev = self.deviations  # for recording

        return


    def run(self,forces,ind):

        # Check if MAPDL server is active, and restart it if it's not
        # if not self._monitor_process('ansys'):
        # if not self.mapdl._is_alive():
        #     self._launch_ansys()

        # Select Ansys input file (randomly)
        folder = path.join(path.dirname(__file__), 'AssemblyGym/envs/FuselageActuators/AnsysFiles/Test')
        file1 = test_files[ind][0]
        filepath = path.join(folder, file1)
        print("Initial shape from", file1.split('.')[0])

        # Parse the Ansys file
        with open(filepath, 'r') as f:
            text = f.read()
            new_text = text.split('/com,******************* SOLVE FOR LS 1 OF 1 ****************')
            self.setup_text = new_text[0]
            new_text = text.split('! *********** WB SOLVE COMMAND ***********')
            self.finish_text = new_text[1]
            f.close()

        # Load precalculated nodal positions
        folder = path.join(path.dirname(__file__), 'AssemblyGym\\envs\\FuselageActuators\\Shapes\\Test')
        file1_npy = file1.split(".")[0] + ".npy"
        filepath = path.join(folder, file1_npy)
        self.initPos = np.load(filepath)
        # self.displacements = np.zeros((177, 2))

        # Randomly select source for target positions
        folder = path.join(path.dirname(__file__), 'AssemblyGym\\envs\\FuselageActuators\\Shapes\\Test')
        file2 = test_files[ind][1]
        file2_npy = file2.split(".")[0] + ".npy"
        # Load precalculated target positions
        filepath = path.join(folder, file2_npy)
        print("Target shape from", file2_npy.split('.')[0])
        self.targetPos = np.load(filepath)

        self.forces = np.zeros(18, dtype=np.float32)
        self._run_ansys()
        self.initPos = self._get_initPos()
        self.displacements = self._get_displacement()
        self.deviations = self._get_deviation()
        self.initDev = self.deviations  # for recording
        self.error, self.maxDev = self._get_errors()
        self.error_initial = self.error
        self.maxDev_initial = self.maxDev

        #####################################################
        self.forces = forces
        self._run_ansys()
        #####################################################
        self.displacements = self._get_displacement()
        u = self.displacements.flatten()
        self.deviations = self._get_deviation()

        # Track
        p_init = self.initPos[:, 0:2].flatten()
        p_final = p_init + u
        p_target = self.targetPos[:, 0:2].flatten()
        self.deviations = p_final - p_target
        self.Actuator_list = np.nonzero(self.forces)

        # Calculate the error
        self.error, self.maxDev = self._get_errors()
        # self.maxDev = prese_max
        # Output info
        info = {"initError": self.error_initial, "Error": self.error, "Forces": self.forces, "maxDev": self.maxDev,
                "initMaxDev": self.maxDev_initial,'ActuatorNum':len(self.Actuator_list)}

        return info

    def close(self):
        self.mapdl.exit()  # close ANSYS

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

    # Functions dealing with ANSYS and processing simulation results
    def _run_ansys(self):
        try:
            # if not self._monitor_process('ansys'):
            # if not self.mapdl._is_alive():
            #     self._launch_ansys()
                # Clear solver memory
            self.mapdl.finish()
            self.mapdl.clear()
            # print("Ready to run")
            # Setup and run the simulation
            log1 = self.mapdl.input_strings(self.setup_text)  # run setup
            log2 = self._set_actuator_forces(self.mapdl, self.forces)  # apply forces
            log3 = self.mapdl.input_strings(self.finish_text)  # complete solution
            self.result = self.mapdl.result  # store result
            # print("Results are available")

        except:
            print("Exit Ansys and try to reconnect")

            try:
                self.mapdl.exit()
                print("Remote exit")
                time.sleep(10)
            except:
                print("No active Ansys process found. Wait and try to reconnect")
                time.sleep(10)

            i = 0
            while True:
                i += 1
                try:
                    self._launch_ansys()
                    print("Sucessfully reconnected to Ansys on attempt", i)
                    print("Try running again")
                    log1 = self.mapdl.input_strings(self.setup_text)  # run setup
                    print("Simulation setup complete")
                    log2 = self._set_actuator_forces(self.mapdl, self.forces)  # apply forces
                    print("Applied forces")
                    log3 = self.mapdl.input_strings(self.finish_text)  # complete solution
                    print("Solve finished")
                    self.result = self.mapdl.result  # store result
                    print("Results ready")

                    break

                except:
                    try:
                        print("Reconnect failed - remote exit again")
                        self.mapdl.exit()
                        time.sleep(10)
                    except:
                        time.sleep(10)
                        if i <= 3:
                            print("Wait and try to reconnect again - attempt", i)
                        else:
                            print("Check Ansys license server connection")
                            # Create popup message
                            root = tk.Tk()
                            root.title('Warning')
                            root.geometry('300x150')
                            answer = showwarning(title='Warning',
                                                 message='Check Ansys license server connection!')
                            if answer:
                                root.destroy()
                            time.sleep(5)

    def _get_stress(self):
        return self.mapdl.post_processing.nodal_eqv_stress()

    def _get_displacement(self):
        '''
        Get the displacement of nodes on the fuselage edge after forces have been applied.
        The displacements are relative to the initial positions of the nodes.
        '''
        self.mapdl.cmsel(name='CM_FUSELAGE_EDGE')  # select nodes on the edge of the fuselage
        displacements = self.mapdl.post_processing.nodal_displacement('ALL')  # get displacements of nodes on the edge
        self.mapdl.allsel()
        return displacements[:, 1:3]

    def _get_initPos(self):
        '''
        Get the initial positions of nodes on the fuselage edge before any forces are applied
        '''
        self.mapdl.cmsel(name='CM_FUSELAGE_EDGE')  # select nodes on the edge of the fuselage
        initPos = self.mapdl.mesh.nodes  # initial positions of nodes
        nnum = self.mapdl.mesh.nnum  # corresponding node numbers
        self.mapdl.allsel()
        return initPos[:, 1:3]

    def _get_deviation(self):
        '''
        Calculate the distance of the current node positions from their ideal positions
        '''
        finalPos = self.initPos + self.displacements
        deviations = finalPos - self.targetPos[:, 0:2]
        return deviations.flatten()

    def _set_actuator_forces(self, mapdl, forces):
        # Calculate y and z components of the forces from desired magnitudes
        angles = np.linspace(12, -192, 18)
        self.forces_Y = forces * np.cos(np.deg2rad(angles))
        self.forces_Z = forces * np.sin(np.deg2rad(angles))

        # Apply the forces as surface force on selected elements
        for i in range(0, 18):
            # Set x component of force (practically zero)
            mapdl.esel("s", "real", "", 27 + 3 * i)
            mapdl.sfe("all", 1, "pres", 1, 2.24808943074769e-009)
            # Set y component of force
            mapdl.esel("s", "real", "", 28 + 3 * i)
            mapdl.sfe("all", 1, "pres", 1, self.forces_Y[i])
            # Set z component of force
            mapdl.esel("s", "real", "", 29 + 3 * i)
            mapdl.sfe("all", 1, "pres", 1, self.forces_Z[i])
        mapdl.esel("all")  # make sure everything is selected before running solve

        # Run the solution
        mapdl._run("/nopr")
        mapdl.run("/gopr")
        mapdl.run("nsub,1,1,1")
        mapdl.time(1.)
        mapdl.outres("erase")
        mapdl.outres("all", "none")
        mapdl.outres("nsol", "all")
        mapdl.outres("rsol", "all")
        mapdl.outres("eangl", "all")
        mapdl.outres("etmp", "all")
        mapdl.outres("veng", "all")
        mapdl.outres("strs", "all")
        mapdl.outres("epel", "all")
        mapdl.outres("eppl", "all")
        mapdl.outres("cont", "all")


    def _launch_ansys(self):
        # Launch ANSYS
        try:
            # n_cpu = psutil.cpu_count(logical=False)
            n_cpu = 4
            self.mapdl = launch_mapdl(loglevel='ERROR', port=self.port,nproc=n_cpu, cleanup_on_exit=True, override=True)
            print(self.mapdl)
            print("Running on", n_cpu, "processors")
        except:
            n_cpu = min(4, n_cpu)  # license sometimes won't let me use more than 4 processors?

            self.mapdl = launch_mapdl(loglevel='ERROR', port=self.port, nproc=n_cpu, cleanup_on_exit=True,
                                      override=True)
            print(self.mapdl)
            print("Running on", n_cpu, "processors")

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device



    run_name = "Test__"+f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    log_name = f"runs/{run_name}"
    writer = SummaryWriter(log_name)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic



    test_envs = gym.make(args.env_id, n_actuators=args.n_actions, mode="Surrogate_Test", record=args.record, seed=args.seed, port=50056)
    state_dict = torch.load('./Agents/agent_32767872steps.pt',
                            map_location=device)
    agent = Agent_OLD(test_envs).to(device)
    test_envs.close()
    agent.load_state_dict(state_dict)

    # test_envs = gym.vector.AsyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.n_actions, "Surrogate_Test",
    #               args.record) for i in range(2)]
    # )
    # state_dict = torch.load('./Agents/FuselageActuators-v22__ppo_FuselageActuators_v22_surrogate__1__1742904880.pth',
    #                         map_location=device)
    # agent = Agent(test_envs,args.n_actions).to(device)
    # test_envs.close()
    # agent.load_state_dict(state_dict)



    forces_list = []
    forcesActive_list = []

    RMSE_list = []
    MaxDev_list = []

    su_RMSE_list = []
    su_MaxDev_list = []

    ansys_run = Run_Ansys()
    perfectPos = ansys_run.initPos


    for i in range(30):
        final_test_env = gym.make(args.env_id, env_no=i, n_actuators=args.n_actions, mode="Surrogate_Test", record=False, seed=args.seed)
        test_next_obs_np, info = final_test_env.reset()
        test_next_obs = torch.Tensor(test_next_obs_np).to(device).unsqueeze(0)

        with torch.no_grad():
            test_action, logprob, _, value = agent.get_action_and_value(test_next_obs,action = "deterministic")
        test_next_obs, reward, done, _, info = final_test_env.step(test_action.squeeze(0).cpu().numpy())
        test_next_obs = torch.Tensor(test_next_obs).to(device)

        writer.add_scalar(f"charts/test_initialError", info["initError"], i)
        writer.add_scalar(f"charts/test_finalError", info["Error"], i)
        writer.add_scalar(f"charts/test_initMaxDev", info["initMaxDev"], i)
        writer.add_scalar(f"charts/test_maxDev", info["maxDev"], i)


        ansys_info = ansys_run.run(final_test_env.forces, i)

        writer.add_scalar(f"charts/Ansys_initialError", ansys_info["initError"], i)
        writer.add_scalar(f"charts/Ansys_finalError", ansys_info["Error"], i)
        writer.add_scalar(f"charts/Ansys_initMaxDev", ansys_info["initMaxDev"], i)
        writer.add_scalar(f"charts/Ansys_maxDev", ansys_info["maxDev"], i)


        initError = ansys_run.error_initial #get_attr("error_initial")#[0]#error_initial
        initPos = ansys_run.initPos #get_attr("initPos")[0]
        targetPos = ansys_run.targetPos #get_attr("targetPos")[0]
        initDev = ansys_run.deviations #get_attr("deviations")[0]


        forces_list.append(final_test_env.forces)

        finalError = ansys_run.error #get_attr("error")#[0] #env.error
        finalPos = ansys_run.initPos + ansys_run.displacements # envs.get_attr("initPos")[0] + envs.get_attr("displacements")[0]
        finalDev = ansys_run.deviations #get_attr("deviations")[0]
        forces = ansys_run.forces #get_attr("forces")[0]
        forcesActive = np.argwhere(forces!=0)
        forcesInactive = np.argwhere(forces==0)

        forcesActive_list.append(forcesActive)

        # Actuator locations
        angles = np.linspace(12, -192, 18)
        angles[17] = 168
        anglesTarget = np.rad2deg(np.arctan2(targetPos[:, 1], targetPos[:, 0]))
        anglesInit = np.rad2deg(np.arctan2(initPos[:, 1], initPos[:, 0]))
        actuatorIds = np.absolute(np.expand_dims(angles, 1) - np.expand_dims(anglesInit, 1).T).argmin(axis=1)

        mag = 25
        initPosVis = perfectPos + (initPos - perfectPos) * mag
        targetPosVis = perfectPos + (targetPos - perfectPos) * mag
        finalPosVis = targetPos + (finalPos - perfectPos) * mag

        if i <4:
            Show_Deviations_Max(forces, actuatorIds, targetPosVis, initPosVis, finalPosVis, forcesActive, forcesInactive,
                                  ansys_info["initMaxDev"], ansys_info["maxDev"], mag)
            Show_Deviations_Error(forces, actuatorIds, targetPosVis, initPosVis, finalPosVis, forcesActive, forcesInactive,
                            ansys_info["initError"], ansys_info["Error"], mag)

        RMSE_list.append(ansys_info["Error"])
        MaxDev_list.append(ansys_info["maxDev"])
        su_RMSE_list.append(info["Error"])
        su_MaxDev_list.append(info["maxDev"])
    MaxDev = np.mean(np.asarray(MaxDev_list))
    RMSE = np.mean(np.asarray(RMSE_list))
    print("MaxDev: ", MaxDev)
    print("RMSE: ", RMSE)

    MaxDev_std = np.std(np.asarray(MaxDev_list))
    RMSE_std = np.std(np.asarray(RMSE_list))
    print("MaxDev_std: ", MaxDev_std)
    print("RMSE_std: ", RMSE_std)

    su_RMSE = np.mean(np.asarray(su_RMSE_list))
    su_MaxDev = np.mean(np.asarray(su_MaxDev_list))
    print("su_RMSE: ", su_RMSE)
    print("su_MaxDev: ", su_MaxDev)

    writer.close()
    ansys_run.close()




