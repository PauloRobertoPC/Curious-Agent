import os
import json
import shutil
from pathlib import Path
from InquirerPy import inquirer

CHECKPOINT_FREQUENCY = 100000

def overwrite_experiment_on_train(train_dir:str, log_dir:str):
    dir_path_train = Path(train_dir)
    dir_path_log = Path(log_dir)

    directory_train_exists = dir_path_train.exists() and dir_path_train.is_dir()
    directory_log_exists = dir_path_log.exists() and dir_path_log.is_dir()

    if directory_train_exists:
        print("Overwriting last train...")
        print("New training starting...")
        shutil.rmtree(dir_path_train)
        if directory_log_exists:
            shutil.rmtree(dir_path_log)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)


def check_experiment_on_train(args, train_dir:str, log_dir:str):

    dir_path_train = Path(train_dir)
    dir_path_log = Path(log_dir)

    directory_train_exists = dir_path_train.exists() and dir_path_train.is_dir()
    directory_log_exists = dir_path_log.exists() and dir_path_log.is_dir()

    if directory_train_exists:
        print(f"A experiment called '{args.experiment}' already exists.")
        overwrite = inquirer.select(
            message="Would you like to overwrite the last experiment with a new one?",
            choices=["No", "Yes"],
        ).execute()
        print(overwrite)
        if overwrite == "No":
            print("Train aborted!")
            exit(0)
        else:
            print("Overwriting last train...")
            print("New training starting...")
            shutil.rmtree(dir_path_train)
            if directory_log_exists:
                shutil.rmtree(dir_path_log)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

def check_reward(parser, args):
    # checking if the arguments are given and correct
    if args.reward is None:
        parser.error("--reward must be passed in training")
        
def check_rnd_strength(parser, args):
    # checking if the arguments are given and correct
    if args.reward is None:
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

def save_experiment_info(CHECKPOINT_DIR, info_dict):
    with open(f"{CHECKPOINT_DIR}/config.json", "w") as f:
        json.dump(info_dict, f, indent=4)

def read_experiment_info(CHECKPOINT_DIR):
    with open(f"{CHECKPOINT_DIR}/config.json", "r") as f:
        data = json.load(f)
    return data