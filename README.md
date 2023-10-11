# contextual-alpha-fair-bandits
Source code for the paper "Contextual $\alpha$-fair Bandits"

# Installing dependencies

The dependencies are listed in `requirements.txt`. To install them, use your favorite Python package manager. For example, with `pip` you can do:

```
python3 -m pip install -r requirements.txt
```

# Running experiments on the [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) dataset

## Preparing the data
The `ml25m` directory contains experiments for the [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) dataset. `cd` into the `ml25m` folder. You may use the `get_data.sh` script to download and extract the dataset. Before running the experiments, use the `data_exploration.py` script to prepare the data needed to run the experiments. You may use the option `--ROWS` to specify the number of rows of the dataset to be used in the experiments; for the experiments in the paper, the first 5000 rows of the dataset were used. Also, the `data_exploration.py` script shuffles the specified number of rows randomly (this is done to create an interesting arrival sequence). For reproducability, use the `--SEED` option to specify the random seed to do the shuffling. For the paper, the following values for the options were used:

```
python3 -m data_exploration --SEED=1 --ROWS=5000
```

## Computing the offline benchmark

To compute the approximate-regret for any policy, the $\alpha$-fair offline benchmark must also be computed. To do this, the `offline_optimal.py` script is provided. The script needs two parameters to run: the fairness parameter (which is named $\alpha$ in the paper), which is specified using the `--ALPHA` option, and a reward parameter, specifying rewards for "bad" arms, which is specified using the `--SMALLREWARD` option (recall that in our experiments, we assign a reward of $1$ to "good" arms, and a small positive reward to "bad" arms). For our paper, we used the following values for the options:

```
python3 -m offline_optimal --ALPHA=0.9 --SMALLREWARD=0.2
```

## Running the full-information and bandit-information feedback experiments

Experiments for the full-information feedback setting are given in the `ml25m.py` scipt. The script takes in values for the options `--ALPHA`, `--SMALLREWARD` and `--SEED`. The `--ALPHA` and `--SMALLREWARD` options are the same as before; the `--SEED` option is used to specify a random seed to be used to get reproducible results, since the policies which we are running are random. For the paper, the script was run as follows:

```
python3 -m ml25m --ALPHA=0.9 --SMALLREWARD=0.2
```

The experiments for the bandit-information feedback setting are given in the `ml25m_bandit.py` script, and is similar to the `ml25m.py` script for the full-information setting. For the paper, it was run as follows:

```
python3 -m ml25m --ALPHA=0.9 --SMALLREWARD=0.2
```

<!-- For Hedge algorithm, see this link: http://www.columbia.edu/~cs2035/courses/ieor6614.S16/mw.pdf. -->

