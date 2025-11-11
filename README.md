# Curious Agent

## Stable Baselines 3

```bash
uv run sb3/main.py
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
