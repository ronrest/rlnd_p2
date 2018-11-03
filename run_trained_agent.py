"""
This file runs a trained agent on the Reacher Unity ENvironment
"""
import numpy as np
import torch
import time
#from collections import deque

from unityagents import UnityEnvironment
from ddpg_agent import MultiAgent

# ##############################################################################
#                                                  SETUP ENVIRONMENT
# ##############################################################################
# NOTE: Before running the code cell below, change the file_name parameter to
# match the location of the Unity environment that you downloaded.
#
# - Mac: "path/to/Reacher.app"
# - Windows (x86): "path/to/Reacher_Windows_x86/Reacher.exe"
# - Windows (x86_64): "path/to/Reacher_Windows_x86_64/Reacher.exe"
# - Linux (x86): "path/to/Reacher_Linux/Reacher.x86"
# - Linux (x86_64): "path/to/Reacher_Linux/Reacher.x86_64"
# - Linux (x86, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86"
# - Linux (x86_64, headless): "path/to/Reacher_Linux_NoVis/Reacher.x86_64"
env_file = "Reacher_Linux/Reacher.x86_64"
env = UnityEnvironment(file_name=env_file)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# Reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)

# size of each action
action_size = brain.vector_action_space_size

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

# Set environment to train mode
env_info = env.reset(train_mode=True)[brain_name]

print("{sep}\nENVIRONMENT INFO\n{sep}".format(sep="="*60))
print('Number of agents:', num_agents)
print('Size of each action:', action_size)
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))

# ##############################################################################
#                                                  SETUP AGENT
# ##############################################################################
# Create agent
agent = MultiAgent(state_size=state_size, action_size=action_size, n_agents=num_agents, seed=42)

# LOAD SAVED SNAPSHOTS OF AGENT
agent.actor_local.load_state_dict(torch.load("checkpoint_actor.pth", map_location=lambda storage, loc: storage))
agent.critic_local.load_state_dict(torch.load("checkpoint_critic.pth", map_location=lambda storage, loc: storage))

# ##############################################################################
#                                                  RUN AN EPISODE
# ##############################################################################
print("{sep}\nRUNNING TRAINED AGENT\n{sep}".format(sep="="*60))
print("Using Device: ", agent.device)

# RESET ENVIRONMENT, STATE AND SCORES
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
episode_scores = np.zeros(num_agents) # score for each agent within episode
agent.reset()

# RUN AN EPISODE
timesteps = 1000
for t in range(timesteps):
    # A single step of interaction with the environment for each agent
    actions = agent.act(states)
    env_info = env.step(actions)[brain_name]
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done

    # Sum up rewards separately for each agent
    episode_scores += np.array(rewards)

    # Prepare for next timestep of iteraction
    states = next_states  # new states become the current states

    # Check if any of the agents has finished. Finish to keep all
    # trajectories in this batch the same size.
    if np.any(dones):
        break
    time.sleep(1/100.)

# FINAL FEEDBACK
episode_score = np.mean(episode_scores) # Summary of scores for this episode
print("Episode Score (averaged across agents): {}".format(episode_score))
print("DONE")
