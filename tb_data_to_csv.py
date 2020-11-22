import os
from typing import Any, Callable, Tuple, NamedTuple, List
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
import re


def get_tag(el: Any) -> str:
    return el.summary.value[0].tag


def tag_contains(tag: str) -> Callable[[Any], bool]:
    def predicate(el: Any) -> bool:
        try:
            return tag in get_tag(el)
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


def get_tb_data(folder: str) -> List:
    train_folder = os.path.join(folder, "train")
    events_fname = [f for f in os.listdir(train_folder) if f.startswith("events.out")][0]
    events_fpath = os.path.join(train_folder, events_fname)
    return list(summary_iterator(events_fpath))


def to_dataframe(tb_data, config: Config) -> pd.DataFrame:
    pseudoreward_data = list(filter(tag_contains("dads/reward"), tb_data))
    pseudoreward_df = pd.DataFrame(dict(
        REWARD_TYPE="PSEUDOREWARD",
        STEP=map(step, pseudoreward_data),
        REWARD=map(value, pseudoreward_data)
    ))

    mpc_data = list(filter(tag_contains("DADS-MPC/rewards"), tb_data))
    mpc_df = pd.DataFrame(dict(
        REWARD_TYPE="MPC_REWARD",
        STEP=map(step, mpc_data),
        REWARD=map(value, mpc_data)
    ))

    df = pd.concat((pseudoreward_df, mpc_df))
    df["SEED"] = config.seed
    df["CONTROL_GOAL_SPACE"] = config.use_state_space_reduction
    df["USE_RESAMPLING"] = config.use_resampling_scheme
    return df


if __name__ == '__main__':
    basedir = "results/reach"
    dfs = []
    for run in os.listdir(basedir):
        df = to_dataframe(tb_data=get_tb_data(folder=os.path.join(basedir, run)),
                          config=get_config(filename=run))
        dfs.append(df)
    full_df = pd.concat(dfs)
    full_df.to_csv(os.path.join(basedir, "full-results.csv"), index=False)
