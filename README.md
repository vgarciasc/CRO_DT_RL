# CRO-DT-RL

This repository contains the code for the paper "Evolving Interpretable Decision Trees for Reinforcement Learning". 
It uses the Coral Reef Optimization algorithm (implemented through [PyCRO-SL](https://github.com/jperezaracil/PyCROSL)) to evolve DTs for RL.

## How to install

The code was tested with Python 3.10. To install the dependencies, clone the repository, `cd` into the folder and run:

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## How to run

The entry point is the file `cro_dt_rl/cro_dt_rl.py`, which effectively runs the CRO algorithm. It has the following params:

- `--task`: the RL task to solve. Can be `cartpole`, `mountain-car`, `lunar-lander` or `dssat`.
- `--cro_config`: the file containing the CRO configuration. See `configs/simple_erl_test.json` for an example.
- `--depth`: the depth of the randomly generated trees.
- `--alpha`: the alpha parameter for the CRO algorithm (determines the weight of tree size).
- `--initial_pop`: file that contains the initial population of trees. If not provided, the population will be randomly generated.
- `--initial_pop_individual`: if True, each simulation will use a different individual from the initial population provided. If False, all individuals from the initial population will be used in each simulation.
- `--episodes`: number of episodes to run when evaluating each individual's fitness.
- `--simulations`: how many simulations to run the CRO-DT-RL algorithm for.
- `--output_prefix`: what should be the prefix of the output file.
- `--n_jobs`: number of parallel jobs to run.
- `--verbose`: if True, the algorithm will print information about the evolution process. True by default.

Two output files will be created: one with the best individual from each simulation, and one with all the statistics from every generation, which is updated constantly.

## Initial population

The initial population can be provided through the `--initial_pop` parameter. Examples of these files can be found in the `final_results` folder.
Note that to obtain these files, it is required to run Imitation Learning and Reward Pruning algorithms. The code for these algorithms can be found in the [erltrees](github.com/vgarciasc/erltrees) repository.