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
python train.py --config configs/tcdbc_dsrl/tcdbc_carrun.py
```

By default we use `ddpm` solver. To use `dpm`, set `--sample_method=dpm` and `-algo_cfg.num_timesteps=1000`.

### Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable.
Alternatively, you could simply run `wandb login`.

## Current Results on OSRL datasets

### CDBC(transformer)

#### comparence with CDT(SOTA)

Table 2. Evaluation results of the normalized reward and cost. The cost threshold is 1. Each value is averaged over 3 distinct cost thresholds, 20 evaluation episodes, and 3 random seeds. 

| Task      | CDT      |        | TCDBC                                                    |                                              | Mark                                                                                                                     |
| ----------- | -------- | ------ | -------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
|              | reward ↑ | cost ↓ | reward ↑                                                 | cost ↓                                       |                                                                                                                                 |
| Ballrun        | 0.39     | 1.16   | 0.37272396634321475                                      | 0.7641666666666668                           | TCDBC using no data augmentation |
| CarRun     | 0.99     | 0.65   | 0.989104                                                 | 0.9775                                       |                                                                                                                                 |
| DroneRun  | 0.63     | 0.79   | 0.5948128719252312 | 0.9675 |                            |
| AntRun  | 0.72     | 0.91   | 0.71514478                                               | 1.057778                                     |                                                                                                                                 |
| BallCircle | 0.77     | 1.07   | 0.7641009573775155                                       | 1.02375                                      |                                               |
| CarCircle  | 0.75     | 0.95   | 0.7282844129731193                                       | 0.9279999999999999                           |   |
| DroneCircle  | 0.63     | 0.98   | 0.6176245753202573         | 1.2                                          |       |
| AntCircle     | 0.54     | 1.78   |             0.4156316443576508                                             |                 2.690208333333333      |                                                              |

**Mark**: In the implementation of CDT, the author uses data augmentation s.t. the agent can learn to imitate the behavior of the most rewaring and safe trajectories when the desired return $(\rho, \kappa)$ is infeasible.
While in the trainning of TCDBC, this technique doesn't always work for TCDBC. Thus in some envs, we do not use the data augmentation technique.

#### training curves

In the following demo, we showed the training process of CDBC(transfomer-based) on showing the average cost return and reward during the training. 
Each evaluation, we test the model with 3 different target return pairs(target return, target cost return).

##### AntCircle

<p float="left">
  <img src="/assets/AntCircle-cost.png" width="500" />
  <img src="/assets/AntCircle-reward.png" width="500" /> 
</p>

##### AntRun

<p float="left">
  <img src="/assets/AntRun-cost.png" width="500" />
  <img src="/assets/AntRun-reward.png" width="500" /> 
</p>

##### BallCircle

<p float="left">
  <img src="/assets/BallCircle-cost.png" width="500" />
  <img src="/assets/BallCircle-reward.png" width="500" /> 
</p>

##### BallRun

<p float="left">
  <img src="/assets/BallRun-cost.png" width="500" />
  <img src="/assets/BallRun-reward.png" width="500" /> 
</p>

##### CarCircle

<p float="left">
  <img src="/assets/CarCircle-cost.png" width="500" />
  <img src="/assets/CarCircle-reward.png" width="500" /> 
</p>

##### CarRun

<p float="left">
  <img src="/assets/CarRun-cost.png" width="500" />
  <img src="/assets/CarRun-reward.png" width="500" /> 
</p>

##### DroneCircle

<p float="left">
  <img src="/assets/DroneCircle-cost.png" width="500" />
  <img src="/assets/DroneCircle-reward.png" width="500" /> 
</p>

##### DroneRun

<p float="left">
  <img src="/assets/DroneRun-cost.png" width="500" />
  <img src="/assets/DroneRun-reward.png" width="500" /> 
</p>

## Credits
The project structure borrows from the [Jax CQL implementation](https://github.com/young-geng/JaxCQL).

We also refer to [the diffusion model implementation from OpenAI](https://github.com/openai/guided-diffusion/tree/main/guided_diffusion), [official CDT implementation](https://github.com/liuzuxin/OSRL) and the [official diffusion Q learning implementation](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL/).
