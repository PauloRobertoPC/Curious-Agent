# Curious Agent

## Stable Baselines 3

### Training an agent

```bash 
uv run sb3/main.py --action=train --experiment=<experiment_name> --reward=<reward type> --glaucoma_level=<glaucoma_intensity>
```

###  Agent playing
```bash 
uv run sb3/main.py --action=play --experiment=<experiment_name> --model=<policy_number>

```
###  Evaluating Agent
```bash 
uv run sb3/main.py --action=evaluate --experiment=<experiment_name> --model=<policy_number> --eval_episodes=<number_of_episodes>
```

###  Getting Help
```bash 
uv run sb3/main.py --help
```

## Sample Factory

### Training an agent

```bash 
uv run sf/train.py --env=health_gathering_glaucoma
```

### Running experiments(Grid Search)

Change the file sf/experiments/exps.py to set the experiments

```bash 
uv run sf/run.py --run=sf.experiments.exps --experiments_per_gpu=1 --num_gpus=1
```

###  Agent playing
```bash 
uv run sf/play.py --env=health_gathering_glaucoma
```

### To see metrics in tensorboard run
```bash 
uv run -m tensorboard.main --logdir=./train_dir
```

# Books to read
1. Autopoieses and Biology of Intentionality
2. Goal as Emergent Autopoietic Processes
3. Autopoieses and Perception
4. Autopoieses and Cognition

# Qualificação

## Capítulos

- Trabalhos Relacionados
- Fundamentação Teórica
