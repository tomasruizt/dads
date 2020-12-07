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


def tag_to_dataframe(tb_data, tagname: str, colname) -> pd.DataFrame:
    tag_data = list(filter(tag_contains(tagname), tb_data))
    return pd.DataFrame(dict(
        ITERATION=map(step, tag_data),
        MEASUREMENT=colname,
        VALUE=map(value, tag_data)
    ))


def to_dataframe(tb_data, config: Config) -> pd.DataFrame:
    tag_to_colname = [
        ("dads/reward", "PSEUDOREWARD"),
        ("DADS-MPC/rewards", "MPC_REWARD"),
        ("dyn-train-goal-changing-transitions[%]", "TRANSITIONS_MOVING_GOAL[%]"),
        ("dynamics-l2-error-moving-goal", "DYN_L2_ERROR_MOVING_GOAL"),
        ("dynamics-l2-error-nonmoving", "DYN_L2_ERROR_NONMOVING_GOAL"),
        ("DADS-MPC/is-success", "IS_SUCCESS")
    ]
    dfs = [tag_to_dataframe(tb_data, tag, colname) for tag, colname in tag_to_colname]
    df = pd.concat(dfs)
    df["SEED"] = config.seed
    df["CONTROL_GOAL_SPACE"] = config.use_state_space_reduction
    df["USE_RESAMPLING"] = config.use_resampling_scheme
    return df


if __name__ == '__main__':
    basedir = "results/reach"
    csv_filename = os.path.join(basedir, "full-results.csv")
    if os.path.isfile(csv_filename):
        os.remove(csv_filename)

    dfs = []
    for run in os.listdir(basedir):
        df = to_dataframe(tb_data=get_tb_data(folder=os.path.join(basedir, run)),
                          config=get_config(filename=run))
        dfs.append(df)
    full_df = pd.concat(dfs)
    full_df.to_csv(csv_filename, index=False)
