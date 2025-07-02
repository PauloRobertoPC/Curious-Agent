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
