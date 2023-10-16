import numpy as np
import pandas as pd
import evo_v6 as evo
from typing import List, Tuple, Set
from profiler import profile, Profiler
import random
import math

TARGET_TRAIN = 0.6
TARGET_VAL = 0.2
TARGET_TEST = 0.2
rows = 60
# NUM_PREV_SPLIT = 20
DATA = pd.DataFrame({  # 10 letters
    "source": [i for i in range(1, rows+1)],
    "metadata_a": np.random.rand(rows),
    "metadata_b": np.random.normal(10, 3, size=rows),
    "metadata_c": np.random.normal(1_000, 100, size=rows),
    "metadata_d": np.random.zipf(1.2, size=rows),
    "metadata_e": [i for i in range(1, rows + 1)],
    "metadata_f": [i if i % 2 == 0 else 0 for i in range(1, rows + 1)],
    "metadata_g": [i if i % rows == 0 else 0 for i in range(1, rows + 1)],
    "metadata_h": np.random.triangular(0, 500, 600, size=rows),
    "metadata_i": np.random.uniform(100, 10_000, size=rows),
    "metadata_j": np.random.poisson(10, size=rows),
}).set_index("source")

PREV_SPLIT = [
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    {13, 14, 15, 16},
    {17, 18, 19, 20},
]


def calculate_l2_norm(data, train: List, val: List, test: List, target_train: float = TARGET_TRAIN,
                      target_val: float = TARGET_VAL, target_test: float = TARGET_TEST) -> Tuple[float, float, float]:
    """
    Calculate the L2 norm of the data split
    :param data: Dataframe containing raw data
    :param train: rows to include in training set
    :param val: rows to include in validation set
    :param test: rows to include in test set
    :param target_train: target train split percentage
    :param target_val: target validation split percentage
    :param target_test: target test split percentage
    :return: tuple of train, val, and test L2 norm
    """
    assert abs((target_test + target_val + target_train) - 1) < 0.000000001
    assert len(train) + len(val) + len(test) == len(data)
    assert len(set(train).intersection(set(val))) == 0
    assert len(set(train).intersection(set(test))) == 0
    assert len(set(val).intersection(set(test))) == 0

    data_total = data.sum(axis=0)

    train_l2 = np.linalg.norm((data.loc[train].sum(axis=0) / data_total) - target_train)
    val_l2 = np.linalg.norm((data.loc[val].sum(axis=0) / data_total) - target_val)
    test_l2 = np.linalg.norm((data.loc[test].sum(axis=0) / data_total) - target_test)

    return train_l2, val_l2, test_l2


@profile
def calculate_train_l2_norm(split: List, data: pd.DataFrame = DATA, target_train: float = TARGET_TRAIN) -> float:
    """
    Calculate the L2 norm of the data split
    :param data: Dataframe containing raw data
    :param train: rows to include in training set
    :param target_train: target train split percentage
    :return: tuple of train, val, and test L2 norm
    """
    train = split[0]

    train_l2 = np.linalg.norm((data.loc[train].sum(axis=0) / data.sum(axis=0)) - target_train)

    return train_l2


@profile
def calculate_val_l2_norm(split: List, data: pd.DataFrame = DATA, target_val: float = TARGET_VAL) -> float:
    """
    Calculate the L2 norm of the data split
    :param data: Dataframe containing raw data
    :param val: rows to include in validation set
    :param target_val: target validation split percentage
    :return: tuple of train, val, and test L2 norm
    """
    val = split[1]

    val_l2 = np.linalg.norm((data.loc[val].sum(axis=0) / data.sum(axis=0)) - target_val)

    return val_l2


@profile
def calculate_test_l2_norm(split: List, data: pd.DataFrame = DATA, target_test: float = TARGET_TEST) -> float:
    """
    Calculate the L2 norm of the data split
    :param data: Dataframe containing raw data
    :param test: rows to include in test set
    :param target_test: target test split percentage
    :return: tuple of train, val, and test L2 norm
    """
    test = split[2]

    test_l2 = np.linalg.norm((data.loc[test].sum(axis=0) / data.sum(axis=0)) - target_test)

    return test_l2


@profile
def check_illegal_demotion(split: List, prev_split: List[Set] = PREV_SPLIT) -> [0, math.inf]:
    """
    Checks if a row from the training set is in the test or validation set and if a row from the validation set is in
    the test set.
    :param split: The solution of the newest split
    :param prev_split: The previously defined split
    :return: A float, infinite if it's illegal, 0 if legal
    """
    current_train = split[0]
    current_val = split[1]
    current_test = split[2]

    prev_train = prev_split[0]
    prev_val = prev_split[1]
    prev_test = prev_split[2]

    if len(set(current_test).intersection(prev_val)) > 0:
        return math.inf
    if len(set(current_test).intersection(prev_train)) > 0:
        return math.inf
    if len(set(current_val).intersection(prev_train)) > 0:
        return math.inf
    return 0


@profile
def swap_between_splits(sol):
    """
    Swap a random row from one split to another
    :param sol: solution to mutate
    :return: mutated solution
    """
    sol = sol[0]
    split1, split2 = random.sample(range(len(sol)), 2)
    # print(split1, split2)
    if len(sol[split1]) == 0 or len(sol[split2]) == 0:
        return sol
    row1_idx = random.randint(0, len(sol[split1]) - 1)
    row2_idx = random.randint(0, len(sol[split2]) - 1)
    sol[split1][row1_idx], sol[split2][row2_idx] = sol[split2][row2_idx], sol[split1][row1_idx]

    return sol


def move_row(sol):
    """
    Move a random row from one split to another
    :param sol: solution to mutate
    :return: mutated solution
    """
    sol = sol[0]
    split1, split2 = random.sample(range(len(sol)), 2)
    # print(split1, split2)
    if len(sol[split1]) == 0:
        return sol
    row_idx = random.randint(0, len(sol[split1]) - 1)
    sol[split2].append(sol[split1].pop(row_idx))

    return sol


def main():
    E = evo.Environment()

    # Add objectives
    E.add_fitness_criteria("train_l2", calculate_train_l2_norm)
    E.add_fitness_criteria("val_l2", calculate_val_l2_norm)
    E.add_fitness_criteria("test_l2", calculate_test_l2_norm)
    E.add_fitness_criteria("illegal_demotion", check_illegal_demotion)

    # another idea would be to have the solution be a np array where each item is 0, 1, or 2 for train, val, test

    # register agents
    E.add_agent("swap_between_splits", swap_between_splits, 1)
    E.add_agent("move_row", move_row, 1)

    # create initial splits
    all_train = [[i for i in range(1, rows+1)], [], []]
    all_val = [[], [i for i in range(1, rows+1)], []]
    all_test = [[], [], [i for i in range(1, rows+1)]]

    E.add_solution(all_train)
    E.add_solution(all_val)
    E.add_solution(all_test)

    for _ in range(10):
        base = list(range(1, rows+1))
        random.shuffle(base)
        while True:
            a = np.random.dirichlet(np.ones(3), size=1)
            a = np.round(a * rows).astype(int)
            if sum(a[0]) == rows:
                break
        x, y, z = base[0: int(a[0][0])], base[int(a[0][0]): int(a[0][1]) + int(a[0][0])], \
            base[int(a[0][1]) + int(a[0][0]): int(a[0][1]) + int(a[0][0]) + int(a[0][2])]
        E.add_solution([x, y, z])

    E.evolve(10*(10**3), 100, viol=math.inf, status=1_000, sync=1_000, time_limit=200, reset=True)

    E.summarize(source='test_results', with_details=True)


if __name__ == "__main__":
    main()
