import os
import sys
import json
import shutil
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env_setup import CHECKPOINT_FREQUENCY

def check_reward(parser, args):
    # checking if the arguments are given and correct
    if args.reward is None:
        parser.error("--reward must be passed in training")
        
def check_rnd_strength(parser, args):
    # checking if the arguments are given and correct
    if args.rnd_strength is None:
        parser.error("--rnd_strength must be passed in training")

def check_glaucoma_level(parser, args):
    # checking if the arguments are given and correct
    if args.glaucoma_level is None:
        parser.error("--glaucoma_level must be passed in training")
    if args.glaucoma_level < 0:
        parser.error("--glaucoma_level must be >= 0")

def check_layout(parser, args):
    # checking if the arguments are given and correct
    if args.layout is None:
        parser.error("--layout must be passed")

def check_experiment(parser, args):
    # checking if the arguments are given and correct
    if args.experiment is None:
        parser.error("--experiment must be passed")

def check_experiment_on_play(parser, args, CHECKPOINT_DIR) -> str:
    # checking if the experiment exists
    dir_path_train = Path(CHECKPOINT_DIR)
    directory_train_exists = dir_path_train.exists() and dir_path_train.is_dir()
    if not directory_train_exists:
        print(f"The experiment called '{args.experiment}' does not exist.")
        exit(0)
    # checking if the arguments are given and correct
    if args.model is None:
        parser.error("--model must be passed in playing, because it's the policy")
    if args.model%CHECKPOINT_FREQUENCY != 0:
        parser.error(f"--model must be a multiple of {CHECKPOINT_FREQUENCY}")
    MODEL_NAME = f'{CHECKPOINT_DIR}/best_model_{args.model}'
    # checking if the file exists
    file_path = Path(f'{MODEL_NAME}.zip')
    if not file_path.is_file():
        print("The model 'best_model_{args.model}' does not exist in {CHECKPOINT_DIR}")
        exit(0)
    return MODEL_NAME

def save_experiment_info(CHECKPOINT_DIR, aux_dict):
    info_dict = aux_dict.copy()
    info_dict.pop("render_mode")
    info_dict.pop("eval_layout")
    with open(f"{CHECKPOINT_DIR}/config.json", "w") as f:
        json.dump(info_dict, f, indent=4)

def read_experiment_info(CHECKPOINT_DIR):
    with open(f"{CHECKPOINT_DIR}/config.json", "r") as f:
        data = json.load(f)
    return data