#!/usr/bin/env python3
"""
Run experiments for "Language and Experience" paper.

Main experimental conditions:
1. Individual Learning (experience only)
2. Social Learning with Human Advice
3. Social Learning with Model Advice
4. Generational/Chain Learning

Example usage:
    # Individual learning
    python run_experiment.py --game beesAndBirds --condition individual --n-lives 15

    # Social learning with human descriptions
    python run_experiment.py --game beesAndBirds --condition social --msg-source human --n-lives 15

    # Social learning with model descriptions
    python run_experiment.py --game beesAndBirds --condition social --msg-source model --n-lives 15

    # Generational/chain learning
    python run_experiment.py --game beesAndBirds --condition chain --n-generations 10 --n-lives-per-gen 2

For more options, run: python run_experiment.py --help
"""

import argparse
import os
import sys

# Add repository to path
repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_path)

from src.utils import get_repo_path


AVAILABLE_GAMES = [
    "avoidGeorge",
    "beesAndBirds",
    "preconditions",
    "portals",
    "pushBoulders",
    "relational",
    "plaqueAttack",
    "aliens",
    "missile_command",
    "jaws",]

GAME_MAX_STEPS = {
    "avoidGeorge": 510,
    "test": 350,
    "beesAndBirds": 1000,
    "preconditions": 500,
    "relational": 350,
    "portals": 1000,
    "pushBoulders": 500,
    "plaqueAttack": 1200,
    "aliens": 1300,
    "jaws": 510,
    "missile_command": 400
}


def get_models_path():
    """Get path to LLM models from environment variable or default locations."""
    if "MODELS_PATH" in os.environ:
        return os.environ["MODELS_PATH"]

    # Check common default locations
    default_paths = [
        os.path.join(get_repo_path(), "data/models/"),
        os.path.expanduser("~/models/"),
    ]

    for path in default_paths:
        if os.path.exists(path):
            return path

    return ""


def create_params(args):
    """Create parameter dictionary from command line arguments."""
    repo_path = get_repo_path()
    models_path = get_models_path()

    # Determine if language should be used
    use_language = args.condition in ["social", "chain"]

    params = dict(
        verbose=args.verbose,
        debug=args.debug,
        seed=args.seed,
        exp_params=dict(
            exp_id=args.exp_id,
            exp_path=os.path.join(repo_path, "data/inference_data/"),
            language_likelihood_type="inverted_proposal",
            msg_to_load=args.msg_source if use_language else None,
            data_to_load=None,
            use_interaction_likelihood=True,
            use_oracle_data=False,
            use_language_proposal=use_language,
            use_language_likelihood=use_language,
            use_data_proposal=True,
            stop_when_solved=args.condition != "chain",
            comparison_immortal=False,
            trial_id=args.trial,
            n_gens=args.n_generations if args.condition == "chain" else 1,
            n_lives_per_gen=args.n_lives_per_gen if args.condition == "chain" else args.n_lives,
            chain=args.condition == "chain",
            agent_reset=args.condition == "chain",
            n_cpus=args.n_cpus,
            render=args.render,
            noisy_action=False,
        ),
        agent=dict(
            thinker=dict(
                alg=args.algorithm,
                schedule="every_20",
                n_smc_steps=1,
                n_particles=args.n_particles,
                n_mcmc_steps=5,
                n_simulations_likelihood=10,
                n_transitions_likelihood=250,
                prior_prob_no_int=0.75,
                prior_prob_low=0.1,
                beta_softmax_lang_proposal=0.25,
                aggregation_strategy="standard",
                llm_params=dict(
                    llm_model=os.path.join(models_path, args.llm_model) if args.llm_model else ""
                )
            ),
            planner=dict(
                max_subgoals_to_plan_for=5,
                planner_type="evo",
                time_budget=2,
                stickiness=0.15,
                discount=0.99,
                mcts_exploration_param=1.5,
                planning_horizon_info=(4, 10, 5),
                safety_trials=20,
                safety_distance=3,
                invalid_dist_rew=-1000,
                beta_explore_exploit=2,
            ),
            reaction_time=4,
            warmup=6,
        ),
        game_params=dict(block_size=100),
        max_steps=GAME_MAX_STEPS.get(args.game, 500)
    )

    return params


def load_linguistic_data(args, params):
    """Load linguistic data if needed for social/chain conditions."""
    if not (params["exp_params"]["use_language_likelihood"] or
            params["exp_params"]["use_language_proposal"]):
        return None

    msg_source = params["exp_params"]["msg_to_load"]
    if msg_source is None:
        return None

    repo_path = get_repo_path()
    descriptions_path = os.path.join(repo_path, "data_input/descriptions/", msg_source)

    trial_id = params["exp_params"]["trial_id"] % 100
    description_file = os.path.join(descriptions_path, f"{args.game}_{trial_id}.txt")

    if os.path.exists(description_file):
        with open(description_file, "r") as f:
            return f.read().strip()
    else:
        print(f"Warning: No description found at {description_file}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for Language and Experience paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=AVAILABLE_GAMES,
        help="Game to play"
    )

    # Experiment condition
    parser.add_argument(
        "--condition",
        type=str,
        default="individual",
        choices=["individual", "social", "chain"],
        help="Experimental condition: individual (experience only), social (with language), chain (generational)"
    )

    # Language source
    parser.add_argument(
        "--msg-source",
        type=str,
        default="human_no_feedback",
        help="Source of language descriptions (e.g., 'human_no_feedback', 'machine_no_feedback')"
    )

    # Experiment configuration
    parser.add_argument("--exp-id", type=str, default="experiment", help="Experiment identifier")
    parser.add_argument("--trial", type=int, default=0, help="Trial number")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # Number of lives/generations
    parser.add_argument("--n-lives", type=int, default=15, help="Number of lives (for individual/social)")
    parser.add_argument("--n-generations", type=int, default=10, help="Number of generations (for chain)")
    parser.add_argument("--n-lives-per-gen", type=int, default=2, help="Lives per generation (for chain)")

    # Inference parameters
    parser.add_argument("--n-particles", type=int, default=20, help="Number of SMC particles")
    parser.add_argument("--n-cpus", type=int, default=8, help="Number of CPUs for parallel inference")
    parser.add_argument("--algorithm", type=str, default="smc", choices=["smc", "llm", "dqn"], help="Learning algorithm")

    # LLM configuration
    parser.add_argument("--llm-model", type=str, default="Meta-Llama-3.1-70B-Instruct", help="LLM model name")

    # Display options
    parser.add_argument("--render", action="store_true", help="Render game visually")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Validate game choice
    if args.game not in AVAILABLE_GAMES:
        print(f"Error: Unknown game '{args.game}'")
        print(f"Available games: {', '.join(AVAILABLE_GAMES)}")
        sys.exit(1)

    # Validate language requirements
    if args.condition in ["social", "chain"]:
        models_path = get_models_path()
        if not models_path or not os.path.exists(models_path):
            print("Warning: MODELS_PATH not set or invalid. LLM features may not work.")
            print("Set with: export MODELS_PATH=/path/to/your/models")

    # Create parameters
    params = create_params(args)

    # Load linguistic data if needed
    linguistic_data = load_linguistic_data(args, params)

    # Import and run main loop
    from src.main import MainLoop

    print(f"\n{'='*60}")
    print(f"Running experiment: {args.exp_id}")
    print(f"Game: {args.game}")
    print(f"Condition: {args.condition}")
    if args.condition in ["social", "chain"]:
        print(f"Message source: {args.msg_source}")
    print(f"{'='*60}\n")

    main_loop = MainLoop(params.copy(), args.game, linguistic_data)
    main_loop.run()

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
