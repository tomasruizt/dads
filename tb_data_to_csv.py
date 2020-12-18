import logging
import os
import warnings
from typing import Any, Callable, Tuple, NamedTuple, List

import click
import pandas as pd
from tensorflow.python.framework.errors_impl import DataLossError
from tensorflow.python.summary.summary_iterator import summary_iterator
import re


def get_tag(el: Any) -> str:
    return el.summary.value[0].tag


def tag_eq(tag: str) -> Callable[[Any], bool]:
    def predicate(el: Any) -> bool:
        try:
            return tag == get_tag(el)
        except:
            return False
    return predicate


def value(el: Any) -> float:
    return el.summary.value[0].simple_value


def step(el: Any) -> int:
    return el.step


class Config(NamedTuple):
    use_resampling_scheme: bool
    use_state_space_reduction: bool
    seed: int


def get_config(filename: str) -> Config:
    match = re.match(r'resampling(.*)_goalSpace(.*)-seed-(.)', filename)
    return Config(use_resampling_scheme=match[1] == "True",
                  use_state_space_reduction=match[2] == "True",
                  seed=int(match[3]))


def get_tb_data(events_out_dir: str) -> List:
    events_fname = [f for f in os.listdir(events_out_dir) if f.startswith("events.out")][0]
    events_fpath = os.path.join(events_out_dir, events_fname)
    return list(summary_iterator(events_fpath))


def tag_to_dataframe(tb_data, tagname: str, colname) -> pd.DataFrame:
    tag_data = list(filter(tag_eq(tagname), tb_data))
    return pd.DataFrame(dict(
        ITERATION=map(step, tag_data),
        MEASUREMENT=colname,
        VALUE=map(value, tag_data)
    ))


def to_dataframe(tb_data, config: Config) -> pd.DataFrame:
    tag_to_colname = [
        ("dads/reward", "PSEUDOREWARD"),
        ("DADS-MPC/rewards", "MPC_REWARD"),
        ("dads/buffer-goal-changing-transitions[%]", "BUFFER-TRANSITIONS_MOVING_GOAL[%]"),
        ("dads/dyn-train-goal-changing-transitions[%]", "DYNTRAIN-TRANSITIONS_MOVING_GOAL[%]"),
        ("dads/dynamics-l2-error-moving-goal", "DYN_L2_ERROR_MOVING_GOAL"),
        ("dads/dynamics-l2-error-nonmoving", "DYN_L2_ERROR_NONMOVING_GOAL"),
        ("DADS-MPC/is-success", "IS_SUCCESS"),
        ("dads/reward-moving", "PSEUDOREWARD_MOVING"),
        ("dads/reward-nonmoving", "PSEUDOREWARD_NONMOVING"),
        ("rollout/ep_rew_reward", "NEW-PSEUDOREWARD")
    ]
    dfs = [tag_to_dataframe(tb_data, tag, colname) for tag, colname in tag_to_colname]
    df = pd.concat(dfs)
    df["SEED"] = config.seed
    df["CONTROL_GOAL_SPACE"] = config.use_state_space_reduction
    df["USE_RESAMPLING"] = config.use_resampling_scheme
    return df


def dump_dads_csv(basedir: str, subdir_name="train"):
    csv_filename = os.path.join(basedir, "full-results.csv")
    if os.path.isfile(csv_filename):
        os.remove(csv_filename)
    dfs = []
    for seed_dir in next(os.walk(basedir))[1]:
        try:
            events_out_dir = os.path.join(basedir, seed_dir, subdir_name)
            config = get_config(filename=seed_dir)
            df = to_dataframe(tb_data=get_tb_data(events_out_dir=events_out_dir), config=config)
            dfs.append(df)
        except DataLossError:
            logging.error(f"file {seed_dir} had an error.")
    full_df = pd.concat(dfs)
    full_df.to_csv(csv_filename, index=False)


def dump_command_skills_csv(dirname: str):
    events_out_dir = os.path.join(dirname, "SAC_1")
    df = to_dataframe(tb_data=get_tb_data(events_out_dir=events_out_dir),
                      config=Config(use_resampling_scheme=False, use_state_space_reduction=True, seed=0))
    csv_fname = os.path.join(dirname, "gsc-results.csv")
    df.to_csv(csv_fname, index=False)


@click.command()
@click.option("--basedir", help="Root of the common tensorboard directories")
@click.option("--subdir", default="train", help="Name of the tb subdirectory")
def cli(basedir: str, subdir):
    dump_dads_csv(basedir=basedir, subdir_name=subdir)


if __name__ == '__main__':
    cli()

