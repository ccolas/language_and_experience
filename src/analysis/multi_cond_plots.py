"""
Multi-condition Learning Curve Plots.

This script generates the main learning curve comparison figures for the paper,
showing performance across different experimental conditions (individual learning,
social learning with human/model messages, etc.).

Usage:
    python multi_cond_plots.py [--data-dir PATH] [--plot-dir PATH]
"""

import os
import sys
import pickle
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# Add repo to path
repo_path = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/'
sys.path.insert(0, repo_path)
from src.utils import get_repo_path

# Default paths - override via command line
DEFAULT_DATA_PATH = os.path.join(get_repo_path(), "data/inference_data/")
DEFAULT_PLOT_PATH = os.path.join(get_repo_path(), "results/plots/")

games = [
    "avoidGeorge",    # ok
    "beesAndBirds",   # ok
    "preconditions",  # ok
    "portals",        # ok
    "pushBoulders",
    "relational",     # ok
    "plaqueAttack",   # ok
    "aliens",
    "missile_command",
    "jaws"
]

# conditions = ['machine_no_msg',  'machine_gpt_msg', 'machine_human_msg', 'machine_humanbest_msg', 'machine_gptoracle_msg',
#               'human_no_msg', 'human_gpt_msg', 'human_human_msg'] # 'machine_hand_msg',
# labels = ['individual machine', 'social machine (machine GPT4 msg)', 'social machine (human msg)', 'social machine (human best msg)', 'social machine (oracle machine GPT$ msg)',
#           'individual human', 'social human (machine GPT4 msg)', 'social human (human msg)'] # 'social machine (hand-defined msg)',
# labels = ['individual', 'social', 'social', 'social', 'social',
#           'individual', 'social', 'social']
top = 5
top_percent = int(top / 20 * 100)

conditions = ['machine_no_feedback',
              'machine_machine_feedback', 'machine_human_feedback', 'machine_human_feedback_to_filter', 'machine_from_human_trajs_machine_feedback',
              'machine_no_feedback_no_explo', 'machine_no_feedback_no_goal', 'machine_oracle_model', 'llm_no_feedback', 'llm_machine_feedback', #"machine_no_feedback_down", "machine_no_feedback_down2",  "machine_no_feedback_down3",
              'human_no_msg', 'human_machine_msg', 'human_human_msg', 'dqn', 'machine_machine_feedback_no_proposals']
              # 'machine_no_feedback_no_goal', 'machine_oracle', 'machine_oracle_no_goal'] # 'machine_hand_msg',
labels =  ['model',
           'model (exp + model msg)', 'model (exp + human msg)', f"model (exp + top {top_percent}% human msg)", "model (exp + model msgs from human trajs)",
           'model (experience, no exploration)', 'model (experience, no goals)', 'oracle model', 'pure LLM', 'pure llm (exp + msg)', #'model (experience, planning errors)', 'model (experience, more planning errors)',  'model (experience, more planning errors2)',
           'human', 'human (exp + model msg)', 'human (exp + human msg)', 'deep RL', 'machine (exp + model msg, no proposals)'] # 'social machine (hand-defined msg)',
the_key = 'receiver_avg_game_rank'
# 'receiver_avg_game_rank'
# 'sender_prestige'
# 'sender_avg_game_rank'

comparison_conditions = [['machine_no_feedback', 'human_no_msg', 'dqn', 'llm_no_feedback', 'machine_oracle_model']]#['human_no_msg', 'human_machine_msg', 'human_human_msg']]#['machine_no_feedback', 'machine_machine_feedback', 'machine_human_feedback', 'machine_from_human_trajs_machine_feedback']]##,]

# comparison_conditions = [['machine_no_feedback', 'machine_human_feedback', 'machine_machine_feedback', 'human_no_msg', 'human_human_msg', 'human_machine_msg'],
                        # ['machine_no_feedback', 'machine_human_feedback', 'machine_machine_feedback'],#, 'llm_machine_feedback'],#, 'machine_from_human_trajs_machine_feedback'],
                        #  ['machine_no_feedback', 'machine_no_feedback_no_explo', 'machine_no_feedback_no_goal'],
                        #  ['human_no_msg', 'machine_no_feedback', 'machine_oracle_model', 'llm_no_feedback', 'dqn'],
                        #  ['machine_no_feedback', 'human_no_msg', ],#"machine_no_feedback_down", "machine_no_feedback_down2", "machine_no_feedback_down3",],
                        #  ['machine_machine_feedback', 'human_machine_msg'],
                         #['human_no_msg', 'human_human_msg', 'human_machine_msg']]

should_compute_stats = [True] * len(comparison_conditions)
# comparison_names = [f'comparing_comparisons/with_top{top_percent}_{the_key}',]
comparison_names = ['comparisons/individual_comp']

# comparison_names = ['comparisons/machine_social_learning_effect',
#                     # 'comparisons/machine_ablations',
#                     # 'comparisons/machine_llm_baselines',
#                     # 'comparisons/machine_human_individual',
#                     # 'comparisons/machine_human_social',
#                     'comparisons/human_social_learning_effect']

                    # 'comparison_individual']#, 'comparison_machine_oracle_ablations']

# msg input: no_msg gpt_msg, hand_msg, human_msg
overwrite = False

figsize = (9, 5)
linewidth = 6
colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556], [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184], 'deeppink', 'dimgray', 'teal']
# plt.rcParams['axes.linewidth'] = 2  # Change the linewidth value as desired
# plt.rcParams.update({'font.size': 20})  # Change the font size value as desired
# ticks_width=2
# ticks_length=5
plt.rcParams['axes.linewidth'] = 4  # Change the linewidth value as desired
plt.rcParams.update({'font.size': 25})  # Change the font size value as desired
ticks_width=5
ticks_length=12

with open(data_path + 'ranks.json', 'r') as f:
    sorted_inds = json.load(f)

max_lives = 15

def plot_learning_curves(results):
    games = list(results.keys())


    for comparison_name, comp_conds, compute_stats in zip(comparison_names, comparison_conditions, should_compute_stats):
        cond_path = plot_path + f'{comparison_name}/'
        os.makedirs(cond_path, exist_ok=True)

        if compute_stats:
            data = dict()
            for i_game, game in enumerate(games):
                for cond, cond_data in results[game].items():
                    if cond not in data.keys():
                        data[cond] = dict()
                    data[cond][i_game] = []
                    for seed, seed_data in cond_data.items():
                        if cond == 'machine_human_feedback_to_filter' and isinstance(the_key, str):
                            if the_key == 'parent_solved_game':
                                max_idx = len(sorted_inds[game][the_key])
                            else:
                                max_idx = top
                            if int(seed) not in sorted_inds[game][the_key][:max_idx]:
                                continue
                        y = seed_data['n_levels_solved']
                        if y[-1] == 4 and len(y) < max_lives:
                            y += [4] * (max_lives - len(y))
                        data[cond][i_game].append(y)
            # plot_relative_auc(data, games)
            # stats_res = auc_analysis(data[comp_conds[0]], data[comp_conds[1]])
            # print(stats_res)
            # stats_res = fixed_effect_analysis(data[comp_conds[0]], data[comp_conds[2]])
            # print(stats_res)
            # plt.show()


        for game in games:
            plt.figure(figsize=figsize)
            line_added = False
            for i_cond, cond in enumerate(comp_conds):
                linestyle = '-'
                i_msg = comp_conds.index(cond)
                data = []
                if cond in results[game].keys():
                    seeds = sorted(list(results[game][cond].keys()))
                    for seed in seeds:
                        seed_data = np.zeros(max_lives + 1)
                        seed_data.fill(np.nan)

                        metrics = results[game][cond][seed]
                        y = [0] + metrics['n_levels_solved']
                        if y[-1] == 4 and len(y) < max_lives + 1:
                            y += [4] * ((max_lives + 1) - len(y))
                        y = np.array(y).astype(float)
                        # if np.all(y==0):
                            # continue
                        seed_data[:len(y)] = y[:max_lives+1]
                        data.append(seed_data)
                    if len(data) > 0:
                        line = np.nanmedian(np.array(data), axis=0)
                        noises = np.random.rand(len(line)) * 0.1
                        line += noises
                        low = np.nanpercentile(np.array(data), 25, axis=0) + noises
                        high = np.nanpercentile(np.array(data), 75, axis=0) + noises
                        # line = np.nanmean(np.array(data), axis=0)
                        # low = line - np.nanstd(np.array(data),  axis=0)
                        # high = line + np.nanstd(np.array(data),  axis=0)
                        x = np.arange(max_lives + 1)
                        labels_idx = conditions.index(cond)
                        # linestyle = '--' if 'no_prop' in cond else '-'
                        # if cond in ['human_no_msg', 'human_human_msg']:
                        #     color = colors[0]
                        # elif cond in ['machine_no_feedback', 'machine_human_feedback']:
                        #     color = colors[1]
                        # else:
                        #     color = colors[3]
                        color = colors[i_cond]
                        plt.plot(x, line, color=color, linewidth=linewidth, label=labels[labels_idx], linestyle=linestyle)
                        plt.fill_between(x, low, high, color=color, alpha=0.25)
                        line_added = True
            if line_added:
                if game == 'missile_command':
                    plt.title('missileCommand', fontweight='bold')
                else:
                    plt.title(game, fontweight='bold')
                plt.xlabel('life', fontweight='bold')
                if 'aliens' in game or 'plaque' in game:
                    plt.ylabel('levels solved', fontweight='bold')
                    plt.yticks([-0.03, 1, 2, 3, 4], [0, 1, 2, 3, 4], fontweight='bold')
                else:
                    plt.yticks([-0.03, 1, 2, 3, 4], [''] * 5, fontweight='bold')
                plt.ylim([-0.04, 4.12])
                plt.yticks([-0.03, 1, 2, 3, 4], [0, 1, 2, 3, 4], fontweight='bold')
                xtickslabels = ['1', '', '', '', '5', '', '', '', '', '10', '', '', '', '', '15'][:max_lives]
                plt.xticks(range(1, max_lives+1), xtickslabels, fontweight='bold')
                # plt.xticks(range(1, max_lives+1), fontweight='bold')

                plt.xlim([-0.06, max_lives + .02])
                plt.yticks(fontweight='bold')
                plt.tick_params(axis='both', which='major', width=ticks_width, length=ticks_length)
                plt.tick_params(axis='both', which='minor', width=ticks_width, length=ticks_length)
                if 'aliens' in game:
                    font = FontProperties()
                    font.set_weight('bold')
                    font.set_size(22)
                    plt.legend(prop=font, loc='lower right')
                    stop = 1
                plt.tight_layout(pad=1)  # Reduce padding around entire figure
                fig_id = f"{game}.png"
                # plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
                plt.savefig(cond_path + fig_id, bbox_inches='tight', pad_inches=0.01, dpi=300)
                plt.savefig(cond_path + fig_id.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.01, dpi=300)
            # plt.show()
            plt.close('all')

    for comparison_name, comp_conds in zip(comparison_names, comparison_conditions):
        cond_path = plot_path + f'{comparison_name}/'
        os.makedirs(cond_path, exist_ok=True)
        plt.figure(figsize=figsize)
        line_added = False
        for i_cond, cond in enumerate(comp_conds):
            lines_game = []
            for game in games:
                if game == 'jaws':
                    stop = 1
                # if cond[:4] == 'mach':
                #     linestyle = '--'
                # else:
                linestyle = '-'
                # msg = cond.split('_')[1]
                # i_msg = ['no', 'gpt', 'human', 'hand', 'humanbest', 'gptoracle'].index(msg)
                # if msg == 'no':
                #     i_msg = 0
                # else:
                #     i_msg = 1
                i_msg = comp_conds.index(cond)

                data = []
                if cond in results[game].keys():
                    seeds = sorted(list(results[game][cond].keys()))
                    for seed in seeds:
                        seed_data = np.zeros(max_lives + 1)
                        seed_data.fill(np.nan)

                        metrics = results[game][cond][seed]
                        y = [0] + metrics['n_levels_solved']
                        if y[-1] == 4 and len(y) < max_lives + 1:
                            y += [4] * ((max_lives + 1) - len(y))
                        y = np.array(y).astype(float)
                        # if np.all(y==0):
                        # continue
                        seed_data[:len(y)] = y[:max_lives+1]
                        data.append(seed_data)
                    if len(data) > 0:
                        line = np.nanmedian(np.array(data), axis=0)
                        lines_game.append(line)

            if len(lines_game) > 0:
                line = np.nanmedian(np.array(lines_game), axis=0)
                line += np.random.rand(len(line)) * 0.1
                low = np.nanpercentile(np.array(lines_game), 25, axis=0)
                high = np.nanpercentile(np.array(lines_game), 75, axis=0)
                # line = np.nanmean(np.array(data), axis=0)
                # low = line - np.nanstd(np.array(data),  axis=0)
                # high = line + np.nanstd(np.array(data),  axis=0)
                x = np.arange(max_lives + 1)
                labels_idx = conditions.index(cond)
                plt.plot(x, line, color=colors[i_msg], linewidth=linewidth, label=labels[labels_idx], linestyle=linestyle)
                plt.fill_between(x, low, high, color=colors[i_msg], alpha=0.25)
                line_added = True
        if line_added:
            plt.xlabel('life', fontweight='bold')
            plt.ylabel('levels solved', fontweight='bold')
            plt.ylim([-0.04, 4.12])
            plt.yticks([0, 1, 2, 3, 4], fontweight='bold')
            xtickslabels = ['1', '', '', '', '5', '', '', '', '', '10', '', '', '', '', '15'][:max_lives]
            plt.xticks(range(1, max_lives + 1), xtickslabels, fontweight='bold')
            # plt.xticks(range(1, max_lives + 1), fontweight='bold')
            plt.xlim([-0.04, max_lives])
            plt.yticks(fontweight='bold')
            plt.tick_params(axis='both', which='major', width=ticks_width, length=ticks_length)
            plt.tick_params(axis='both', which='minor', width=ticks_width, length=ticks_length)
            # plt.title("human (median over games)", fontweight='bold')
            plt.legend(prop= {'weight':'bold'}, loc='lower right')
            plt.tight_layout()
            fig_id = f"human_all.png"
            plt.savefig(cond_path + fig_id, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.savefig(cond_path + fig_id.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close('all')
        stop = 1



    stop = 1

def get_perf_lvl(lvls_solved, lvl):
    if lvl in lvls_solved:
        return lvls_solved.index(lvl) + 1
    else:
        for above_lvl in range(lvl, 5):
            if above_lvl in lvls_solved:
                return lvls_solved.index(above_lvl) + 1
        return 15

def plot_bars():
    pass


def get_gen(filename):
    return int(filename.split('generation_')[1].split('_life')[0])

def get_life(filename):
    return int(filename.split('_life_')[1].split('_lvl')[0])

def extract_seed(seed_path):
    with open(seed_path + 'params.json', 'r') as f:
        params = json.load(f)
    n_lives_per_gen = params['exp_params']['n_lives_per_gen']
    file_names = os.listdir(seed_path + 'dumps/interaction_data/')
    n_gens = len(file_names)
    data = dict()
    for file_name in sorted(file_names):
        gen = get_gen(file_name)
        life = get_life(file_name)
        total_life = gen * n_lives_per_gen + life
        with open(seed_path + f'dumps/interaction_data/{file_name}', 'rb') as f:
            data[total_life] = pickle.load(f)

    keys_to_track = ['episodes', 'env_steps', 'actions', 'life', 'gen', 'n_levels_solved']
    episode_metrics = dict(zip(keys_to_track, [[] for _ in range(len(keys_to_track))]))
    episode_metrics['reward'] = []
    for i_life in sorted(list(data.keys())):
        episode = data[i_life]
        for k in keys_to_track:
            episode_metrics[k].append(episode[k])


    return episode_metrics

def extract_results(data_path, overwrite=True):
    if os.path.exists(data_path + 'extracted_res.pkl') and not overwrite:
        with open(data_path + 'extracted_res.pkl', 'rb') as f:
            results = pickle.load(f)
    else:
        results = dict()
    with open(human_no_msg_path, 'r') as f:
        human_no_msg_data = json.load(f)
    with open(human_machine_msg_path, 'r') as f:
        human_machine_msg_data = json.load(f)
    with open(human_human_msg_path, 'r') as f:
        human_human_msg_data = json.load(f)
    for game in games:
        game_str = game
        if game_str not in results.keys():
            results[game_str] = dict()
        print(f'> extracting results for game {game}')
        for cond in conditions:
            if cond in ['human_no_msg', 'human_machine_msg', 'human_human_msg']:
                continue
            print(f'  > extracting results for cond {cond}')
            study_path = data_path + cond + '/'
            game_path = study_path + game + '/'
            if cond == 'dqn':
                stop = 1
            if os.path.exists(game_path):
                seeds = os.listdir(game_path)
                if cond not in results[game_str].keys():
                    results[game_str][cond] = dict()
                for seed in seeds:
                    seed_path = game_path + seed + '/'
                    if cond == 'machine_human_feedback_to_filter' and isinstance(the_key, str):
                        if the_key == 'parent_solved_game':
                            max_idx = len(sorted_inds[game][the_key])
                        else:
                            max_idx = top
                        if int(seed) not in sorted_inds[game][the_key][:max_idx]:
                            if seed in results[game_str][cond].keys():
                                del results[game_str][cond][seed]
                            continue
                    if seed not in results[game_str][cond].keys():
                        try:
                            episode_metrics = extract_seed(seed_path)
                            if len(episode_metrics['life']) < 15 and not episode_metrics['n_levels_solved'][-1] == 4:
                                print(f'    > seed {seed}, incomplete')
                                pass
                            else:
                                # print(f'    > seed {seed}, success')
                                results[game_str][cond][seed] = episode_metrics
                                with open(data_path + 'extracted_res.pkl', 'wb') as f:
                                    pickle.dump(results, f)
                        except:
                            # print(f'    > seed {seed}, failure!!')
                            pass
                print(f"    > {len(results[game_str][cond])} valid seeds")


        human_game_id = f"JRNL_{game}_v0"
        human_conditions = ['human_no_msg', 'human_machine_msg', 'human_human_msg']#, 'human_gpt_msg', 'human_human_msg']
        human_data = [human_no_msg_data[human_game_id], human_machine_msg_data[human_game_id], human_human_msg_data[human_game_id]]#, human_gpt_msg_data[human_game_id], human_human_msg_data[human_game_id]]
        for cond, cond_data in zip(human_conditions, human_data):
            results[game_str][cond] = dict()
            print(f'  > extracting results for cond {cond}')
            for element in cond_data:
                seed = element['subject']
                if seed not in results[game_str][cond].keys():
                    print(f'  > extracting seed {seed}')
                    results[game_str][cond][seed] = dict(n_levels_solved=[None] * max_lives,
                                                   lvl_players=[None] * max_lives)
                if element['life'] <= max_lives:
                    if results[game_str][cond][seed]['n_levels_solved'][element['life'] - 1] is not None:
                        results[game_str][cond][seed]['n_levels_solved'][element['life'] - 1] = max(element['level'] + int(element['won']), results[game_str][cond][seed]['n_levels_solved'][element['life'] - 1])
                    else:
                        results[game_str][cond][seed]['n_levels_solved'][element['life'] - 1] = element['level'] + int(element['won'])
                    if element['won'] and element['level'] == 3:
                        results[game_str][cond][seed]['n_levels_solved'][element['life']:] = [4] * (max_lives - element['life'])
            print(f"    > {len(results[game_str][cond])} valid seeds")

    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate multi-condition learning curve plots')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_PATH,
                        help='Directory containing experiment results')
    parser.add_argument('--plot-dir', type=str, default=DEFAULT_PLOT_PATH,
                        help='Directory to save plots')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing extracted results')
    args = parser.parse_args()

    # Update global paths
    data_path = args.data_dir
    plot_path = args.plot_dir
    os.makedirs(plot_path, exist_ok=True)

    # Define human data paths
    human_no_msg_path = os.path.join(data_path, "human_no_msg.json")
    human_machine_msg_path = os.path.join(data_path, "human_machine_msg.json")
    human_human_msg_path = os.path.join(data_path, "human_human_msg.json")

    results = extract_results(data_path, args.overwrite)
    plot_learning_curves(results)