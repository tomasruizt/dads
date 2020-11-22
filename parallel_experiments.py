from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

import click
from itertools import product
from typeguard import typechecked


@click.command()
@click.option("--shell_file", default="run_point2d.sh")
@click.option("--num_seeds", default=10)
@click.option("--cpus", default=4)
def main(shell_file: str, num_seeds: int, cpus: int) -> None:
    seeds = range(num_seeds)
    use_resampling = (True, False)
    reduce_state_space = (True, False)
    configs = product(seeds, use_resampling, reduce_state_space)

    print("Started experiments!")
    with Pool(cpus) as pool:
        run = partial(run_experiment, shell_file=shell_file)
        pool.map(run, configs)
    print("Finished experiments!")


@typechecked
def make_command(seed: int, use_resampling: bool, reduce_state_space: bool, shell_file: str) -> str:
    experiment_name = get_experiment_name(use_resampling=use_resampling, reduce_state_space=reduce_state_space)
    return f"bash {shell_file} --seed={seed} " \
           f"--use_dynamics_uniform_resampling={use_resampling} " \
           f"--use_state_space_reduction={reduce_state_space} " \
           f"--experiment_name={experiment_name}"


@typechecked
def get_experiment_name(use_resampling: bool, reduce_state_space: bool) -> str:
    return f"resampling{use_resampling}_goalSpace{reduce_state_space}"


def run_experiment(config, shell_file):
    command = make_command(*config, shell_file=shell_file)
    seed, use_resampling, reduce_state_space = config
    experiment_name = get_experiment_name(use_resampling=use_resampling, reduce_state_space=reduce_state_space)
    log_fpath = f"./logs/out-{experiment_name}-seed{seed}.log"
    with open(log_fpath, "w") as file:
        call(command.split(), stdout=file, stderr=file)


if __name__ == '__main__':
    main()

