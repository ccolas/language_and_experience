import gym
import pygame
import pickle
import time
import numpy as np
import os

from src.utils import register_game, get_repo_path

def play_game(game_id='sokoban-v0', lvl=0, step_by_step=True, fps=20, save_to=None, verbose=False, with_img=True):
    env = gym.make(game_id)
    env.render()
    obs_0, info = env.unwrapped.reset(with_img=with_img)
    env.render()
    score = 0
    step = 0
    transition = dict(step=step, action=None, reward=None, done=None, img=obs_0, state=info['state'], won=None, lose=None, events_triggered=None, lvl=lvl, reset=True)
    trajectory = [transition]
    actions_times = []
    transitions_times = []
    t_init = time.time()
    while True:
        t_init_2 = time.time()
        action_id = -1
        while action_id == -1:
            run, action_id = get_action_from_key(env, step_by_step=step_by_step, fps=fps)
        if not run:
            break
        if not step_by_step or action_id != -1:
            step += 1
            if verbose: print(step)
            if action_id != -1:
                actions_times.append(time.time() - t_init)
                t_init = time.time()

            state, reward, isOver, truncated, info = env.unwrapped.step(action_id, with_img=with_img)
            # if reward != 0:
            #     print(reward)
            #     stop = 1
            transition = dict(step=step, action=action_id, reward=reward, done=isOver, img=state, state=info['state'], won=info['won'], lose=info['lose'],
                              events_triggered=info['events_triggered'], lvl=lvl, reset=False)
            trajectory.append(transition)
            score += reward
            if verbose:
                print("Action " + str(action_id) + " played at game tick " + str(step+1) + ", reward=" + str(reward) + ", new score=" + str(score))
            env.render()

            if isOver:
                if info['won']:
                    print('You won!')
                else:
                    print('You lost!')
                break
        if not step_by_step:
            time_so_far = time.time() - t_init_2
            time_to_sleep = 1 / fps - time_so_far
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
        transitions_times.append(time.time() - t_init_2)

    env.close()
    print(np.mean(actions_times))
    print(1/np.min(actions_times))
    print(1/np.mean(transitions_times))

    # save trajectory
    if save_to is not None:
        with open(save_to, 'wb') as f:
            pickle.dump(trajectory, f)

#
def get_action_from_key(env, step_by_step, fps):
    run, action_id = True, -1
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, None
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                return False, None
            for i_a, a in enumerate(env.unwrapped._action_keys):
                if len(a.keys) > 0:
                    if a.keys[0] == event.key:
                        action_id = i_a
                        break
    return run, action_id

#
def find_action_id(env, key):
    for i_a, a in enumerate(env.unwrapped._action_keys):
        if a.keys[0] == key:
            return i_a
    return None


if __name__ == '__main__':
    jnrl_games = ["sokoban",
                  "bait",
                  "aliens",
                  "avoidGeorge",   # ok
                  "beesAndBirds",  # ok
                  "plaqueAttack",  # ok
                  "portals",       # ok
                  "preconditions", # ok
                  "pushBoulders",
                  "relational",    # ok
                  "jaws",
                "missile_command",
                  "watergame",
                  "boulderDash",
                  ]
    game = 'plaqueAttack'
    lvl = 0
    register_game(game, level=lvl, fast=False)
    # output = play_game(game_id=game + '-v0', lvl=lvl, step_by_step=False, verbose=True)

    general_path = get_repo_path() + f'data/trajectories/{game}/'
    os.makedirs(general_path, exist_ok=True)
    play_game(game_id=game + '-v0', step_by_step=False, lvl=lvl, verbose=True, with_img=False, save_to=general_path + game + '_complete_0.pkl')
