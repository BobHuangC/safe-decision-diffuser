# Safe Decision Diffuser

## Setup the environment

Create python environment with conda
```bash
conda create -f environment.yml
conda activate safediff
pip install 'shimmy[gym-v21]'
pip install -e .
```

Apart from this, you'll have to setup your MuJoCo environment and key as well. Please follow [D4RL](https://github.com/Farama-Foundation/D4RL) and [DSRL](https://github.com/liuzuxin/DSRL) repo and setup the environment accordingly.

### Run Experiments

You can run decision-diffuser experiments using the following command:

```bash
python train.py --config configs/cdbc_dsrl/cdbc_antrun_mlp.py
```

By default we use `ddpm` solver. To use `dpm`, set `--sample_method=dpm` and `-algo_cfg.num_timesteps=1000`.

### Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable.
Alternatively, you could simply run `wandb login`.

## Current Results on OSRL datasets

### CDBC(transformer)



## Credits
The project structure borrows from the [Jax CQL implementation](https://github.com/young-geng/JaxCQL).

We also refer to [the diffusion model implementation from OpenAI](https://github.com/openai/guided-diffusion/tree/main/guided_diffusion), [official CDT implementation](https://github.com/liuzuxin/OSRL) and the [official diffusion Q learning implementation](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL/).
