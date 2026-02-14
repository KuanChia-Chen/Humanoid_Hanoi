import torch
import argparse
import sys
import pickle
import os

from util.evaluation_factory import interactive_eval
from util.nn_factory import load_checkpoint, nn_factory
from util.env_factory import env_factory, add_env_parser

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--offscreen', default=False, action='store_true')
    # Manually handle path argument
    try:
        path_idx = sys.argv.index("--path")
        model_path = sys.argv[path_idx + 1]
        if not isinstance(model_path, str):
            print(f"{__file__}: error: argument --path received non-string input.")
            sys.exit()
    except ValueError:
        print(f"No path input given. Usage is 'python eval.py simple --path /path/to/policy'")

    # model_path = args.path
    previous_args_dict = pickle.load(open(os.path.join(model_path, "experiment.pkl"), "rb"))
    actor_checkpoint = torch.load(os.path.join(model_path, 'actor.pt'), map_location='cpu')

    # Load environment
    previous_args_dict['env_args'].simulator_type = "box_tower_of_hanoi"
    previous_args_dict['algo_args'].env_name = "BoxTowerOfHanoiEnv"

    add_env_parser(previous_args_dict['all_args'].env_name, parser, is_eval=True)
    args = parser.parse_args()

    # Overwrite previous env args with current input
    for arg, val in vars(args).items():
        if hasattr(previous_args_dict['env_args'], arg):
            setattr(previous_args_dict['env_args'], arg, val)

    previous_args_dict['env_args'].simulator_type += "_mesh"      # Use mesh model
    print(previous_args_dict['env_args'])
    previous_args_dict['env_args'].reward_name = "pos_delta_target"
    env = env_factory(previous_args_dict['all_args'].env_name, previous_args_dict['env_args'])()

    interactive_eval(env=env, offscreen=args.offscreen)
