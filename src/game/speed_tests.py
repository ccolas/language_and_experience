import gym
import pygame
import pickle
import time
from src.utils import register_game
import numpy as np


def test_game(game_id, n_steps, with_reset=False, render=False):
    env = gym.make(game_id)
    env.unwrapped.reset()
    if render: env.render()
    n_actions = len(env.unwrapped._action_keys)
    hidden_state = None
    times = []
    print(f'eval {game_id}')
    for i in range(n_steps):
        if (i + 1) % 100 == 0:
            print('  ', i+1)
        action_id = np.random.randint(n_actions)
        is_noop = len(env.unwrapped._action_keys[action_id].keys) == 0
        t_init = time.time()
        if with_reset and hidden_state:
            env.unwrapped.set_state(hidden_state)
        n_steps_to_run = 4 if not is_noop else 1
        for i_mock in range(n_steps_to_run):
            if i_mock > 0:
                action_id = -1
            state, reward, isOver, truncated, info = env.unwrapped.step(action_id)
            if render: env.render()
            if isOver:
                break
        times.append(time.time() - t_init)
        if isOver:
            if with_reset:
                hidden_state = None
            env.reset()
        elif with_reset:
            hidden_state = env.unwrapped.get_state()
    return times


if __name__ == '__main__':
    jnrl_games = ["JRNL_avoidGeorge",   # ok
                  "JRNL_beesAndBirds",  # ok
                  "JRNL_plaqueAttack",  # ok
                  "JRNL_portals",       # ok
                  "JRNL_preconditions", # ok
                  "JRNL_pushBoulders",
                  "JRNL_relational",    # ok
                  "JRNL_watergame",
                  "JRNL_boulderDash"]
    jnrl_games = ['JRNL_tutorial']
    results = dict()
    for game in jnrl_games:
        results[game] = dict()
        for level in range(1):
            results[game][level] = dict()
            register_game(game, level=level, fast=False)
            results[game][level]['with_reset'] = test_game(game_id=game + '-v0', n_steps=1000, with_reset=False)
            print(f"  with reset: fps: {1000 / np.sum(results[game][level]['with_reset'])}")