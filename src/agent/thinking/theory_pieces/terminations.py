import numpy as np

from src.utils import AVATAR_NAME, format_prompt, get_neighbors



class KillAll:
    def __init__(self, name, params):
        self.name = name
        self.rewards_when_killed = []  # rewards obtained when this object gets killed
        self.rewards_when_killed_per_obj = dict()
        self.counts_when_win = []  # counts of this object when we win
        self.prev_counts_when_win = []  # counts of this object before we win
        self.counts_when_not_win = []  # counts of this object when we don't win

        self.counts_when_loss = []  # counts of this object when we lose
        self.prev_counts_when_loss = []  # counts of this object right before we lost
        self.counts_when_not_loss = []  # counts of this object when we don't win
        colors = params['true_game_info']['colors']
        self.colors = dict(zip(colors.keys(), [v.lower() for v in colors.values()]))

    def get_str(self, win_or_lose):
        if win_or_lose == 'win':
            eval_prompt = format_prompt([self.name], 'win', None, self.colors)
        elif win_or_lose == 'lose':
            eval_prompt = format_prompt([self.name], 'lose', None, self.colors)
        else:
            raise NotImplementedError
        return eval_prompt


    def prob_reward(self, names, vgdl_lines):
        # whether killing this object gives a reward, positive or negative
        if len(self.rewards_when_killed) == 0:
            probs = [0, 0, 1]  # positive, negative, zero
        elif names not in self.rewards_when_killed_per_obj.keys():
            probs = [0, 0, 1]
        else:
            ratio_positive = np.sum(np.array(self.rewards_when_killed_per_obj[names]) > 0) / len(self.rewards_when_killed_per_obj[names])
            ratio_negative = np.sum(np.array(self.rewards_when_killed_per_obj[names]) < 0) / len(self.rewards_when_killed_per_obj[names])
            ratio_null = np.sum(np.array(self.rewards_when_killed_per_obj[names]) == 0) / len(self.rewards_when_killed_per_obj[names])
            probs = np.array([min(0.9, ratio_positive), min(0.9, ratio_negative), max(0.1, ratio_null)])
            probs /= probs.sum()
            # if probs.sum() != 1:
            #     print(probs)
        return probs




    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, traj_episode, obj_episode, nn_episode):
        if i_step == 0:
            self.counts_when_not_win.append(traj_episode['count_objs'][i_step][self.name])
        else:
            done = traj_episode['done'][i_step - 1]
            win = traj_episode['won'][i_step - 1]
            lost = traj_episode['lose'][i_step - 1]
            prev_pos, pos = obj['pos'][i_step - 1], obj['pos'][i_step]
            just_died = prev_pos is not None and pos is None
            if just_died:
                self.rewards_when_killed.append(traj_episode['reward'][i_step - 1])
                colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj['obj_id'], prev_pos, radius=2)
                for other_obj in colliding_objects:
                    key = (obj['name'], obj_episode[other_obj]['name'])
                    if key not in self.rewards_when_killed_per_obj.keys():
                        self.rewards_when_killed_per_obj[key] = []
                    self.rewards_when_killed_per_obj[key].append(traj_episode['reward'][i_step - 1])
            if win:
                self.counts_when_win.append(traj_episode['count_objs'][i_step][self.name])
                self.prev_counts_when_win.append(traj_episode['count_objs'][i_step - 1][self.name])
            else:
                self.counts_when_not_win.append(traj_episode['count_objs'][i_step][self.name])
            if lost:
                self.counts_when_loss.append(traj_episode['count_objs'][i_step][self.name])
                self.prev_counts_when_loss.append(traj_episode['count_objs'][i_step - 1][self.name])
            else:
                self.counts_when_not_loss.append(traj_episode['count_objs'][i_step][self.name])

    def get_scores(self):
        # wining requires to kill this object:
        # - if it gives positive rewards when we kill it
        # - if we won and the count just passed to 0
        # but not if:
        # - we won and the count was > 0
        # could be but not sufficient if:
        # - count was 0 before winning too (eg at start)

        win_score = 0.75
        if AVATAR_NAME in self.name or self.name == 'wall':
            win_score = 0
        elif len(self.counts_when_win) > 0:
            if np.any(np.array(self.counts_when_win) > 0):
                # - we won and the count was > 0
                win_score = 0
            elif np.all(np.array(self.prev_counts_when_win) > 0):
                # - if we won and the count just passed to 0
                win_score = 0.9
            elif len(self.rewards_when_killed) > 0:
                win_score = max(min(np.mean(np.array(self.rewards_when_killed) > 0), 0.9), 0.75)
        elif len(self.rewards_when_killed) > 0:
            # - if it gives positive rewards when we kill it
            win_score = max(min(np.mean(np.array(self.rewards_when_killed) > 0), 0.8), 0.75)

        # losing might be caused by killing objects:
        # - that give you negative rewards
        # - that are the agent
        # - whose count went to 0 exactly when losing
        # unless:
        # their count was 0 at some point without you losing
        if AVATAR_NAME in self.name:
            lose_score = 1
        elif self.name == 'wall':
            lose_score = 0
        elif len(self.counts_when_loss) > 0:
            lose_score = 0.25 # count is 0.25 by default
            if np.any(np.array(self.counts_when_not_loss) == 0):
                # count was 0 without losing, probably not making you lose, but maybe so in combination with others
                lose_score = 0.2
            # count went to 0 exactly when it lost
            inds_0 = np.argwhere(np.array(self.counts_when_loss) == 0).flatten()
            for i in inds_0:
                if self.prev_counts_when_loss[i] > 0:
                    if np.any(np.array(self.counts_when_not_loss) == 0):
                        lose_score = 0.5
                    else:
                        lose_score = 0.9
        elif len(self.rewards_when_killed) > 0:
            lose_score = max(min(0.9, np.mean(np.array(self.rewards_when_killed) < 0)), 0.25)
        else:
            # no data here
            lose_score = 0.25

        assert not (lose_score == 1 and win_score == 1)
        if lose_score == 1:
            win_score = 0
        elif win_score == 1:
            lose_score = 0
        return win_score, lose_score


class Timeout:
    def __init__(self, params):
        colors = params['true_game_info']['colors']
        self.colors = dict(zip(colors.keys(), [v.lower() for v in colors.values()]))

    def get_str(self, win_or_lose):
        if win_or_lose == 'win':
            eval_prompt = format_prompt(None, 'win', None, self.colors)
        elif win_or_lose == 'lose':
            eval_prompt = format_prompt(None, 'lose', None, self.colors)
        else:
            raise NotImplementedError
        return eval_prompt
