# contextual-alpha-fair-bandits
Source code for the paper "Contextual $\alpha$-fair Bandits"

# Installing dependencies

The dependencies are listed in `requirements.txt`. To install them, use your favorite Python package manager. For example, with `pip` you can do:

```
python3 -m pip install -r requirements.txt
```

To generate plots, we also use the `texlive` distribution along with a few additional packages. On Linux machines, this could be installed via the following: 

```
sudo apt install texlive-fonts-recommended texlive-fonts-extra cm-super dvipng
```

If you don't want to use `texlive` in plots, just leave out the `--USETEXLIVE` option in the config below.


# Running experiments on the [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) dataset

Throughout this section, we assume that we're working in the `ml25m` folder.

## Preparing the data

Use the `get_data.sh` script to download the dataset:

```shell
./get_data.sh
```

This will download and extract the dataset in a folder called `data`. Also, create a folder called `pickled_files`:

```
mkdir -p pickled_files
```

## Set the config.

As part of the repository, we've already included the config file we used to run our code (`config.json`). For completeness, here are all the config parameters we used to run our code (leave out `--USETEXLIVE` if you don't have a `texlive` distribution):

```shell
python3 -m config --ROWS=50000 --SEED=1 --ALPHA=0.9 --SMALLREWARD=0.001 --USETIMESTAMPS --FREQUENCY=1000 --FREQUENCY_MAX=1000 --FAIRCBFAIRNESS=0.98 --USETEXLIVE --HIGHFREQUENCY --PLOTREGRET --NUMNUS=50 --NUMALPHAS=50 --VARYING_NU_ROUNDS=10
```

Details about the config parameters are given in the `config.py` file.

## Prepare the dataset

Just use the `data_exploration` script for this:

```
python -m data_exploration
```

## Computing the offline benchmark

To compute the offline optimal benchmark (used to plot the approximate and standard regrets), use this:

```shell
python3 -m offline_optimal_optimized
```

## The full information setting experiments

Just run the `ml25m` and `ml25m_plots` scripts: 

```shell
python3 -m ml25m
python3 -m ml25m_plots
```

## Experiments for varying values of $\alpha$

For the experiments with varying values of $\alpha$, use the following scripts:

```shell
python3 -m dependence_on_alpha
python3 -m dependence_on_alpha_plots
```

## Experiments for varying values of $\nu$

We also compared the performance of our policy with the `FairCB` policy by Chen et. al 2020 (by trying out different values of $\nu$, the fairness parameter of `FairCB`, against our policy with $\alpha=0.9$). To run these experiments and get the corresponding plot, just run the following:

```shell
python3 -m dependence_on_nu
python3 -m dependence_on_nu_plots
```


## The bandit information setting experiments

For the bandit setting, just use the `ml25m_bandit` and `ml25m_bandit_plots` scripts:

```shell
python3 -m ml25m_bandit
python3 -m ml25m_bandit_plots
```

For the bandit setting, we also used a larger number of datapoints to plot Jain's Fairness Index (i.e around 50k points). For this, from the old config, remove the `--HIGHFREQUENCY` and `--PLOTREGRET` options, and then run the experiments:

```
python3 -m config --ROWS=50000 --SEED=1 --ALPHA=0.9 --SMALLREWARD=0.001 --USETIMESTAMPS --FREQUENCY=1000 --FREQUENCY_MAX=1000 --FAIRCBFAIRNESS=0.98 --USETEXLIVE --NUMNUS=50 --NUMALPHAS=50 --VARYING_NU_ROUNDS=10
python3 -m data_exploration
python3 -m ml25m_bandit
python3 -m ml25m_bandit_plots
```

<!-- For Hedge algorithm, see this link: http://www.columbia.edu/~cs2035/courses/ieor6614.S16/mw.pdf. -->


