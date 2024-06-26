import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--ROWS', dest='ROWS', default=50000, help='Number of rows of the dataset to use.')
parser.add_argument('--SEED', dest='SEED', default=42, help='Random seed to have reproducible results.')
parser.add_argument('--ALPHA', dest='ALPHA', default=0.5, help='Fairness level')
parser.add_argument('--SMALLREWARD', dest='SMALL_REWARD', default=0.001, help='Very small reward for the bad arm.')
parser.add_argument('--USETIMESTAMPS', dest='USETIMESTAMPS', action='store_true', help='Boolean determining whether the timestamps given in the dataset will be used to shuffle the rows.')
parser.add_argument('--FREQUENCY', dest='FREQUENCY', default=5000, help='Minimum frequency of a context in the resultant dataset.')
parser.add_argument('--FREQUENCY_MAX', dest='FREQUENCY_MAX', default=10000, help='Maximum frequency of a context in the resultant dataset.')
parser.add_argument('--FAIRCBFAIRNESS', dest='FAIRCBFAIRNESS', default=0.5, help='Fairness parameter to be used for the FairCB algorithm.')
parser.add_argument('--USETEXLIVE', dest='USETEXLIVE', action='store_true', help='Whether to use TexLive during plot generation.')
parser.add_argument('--HIGHFREQUENCY', dest='HIGHFREQUENCY', action='store_true', help='Whether to use only those users whose frequency is atleast FREQUENCY and atmost FREQUENCY_MAX.')
parser.add_argument('--PLOTREGRET', dest='PLOTREGRET', action='store_true', help='True if standard/approximate regret needs to be plotted. False otherswise. This saves time if offline optimal values dont have to be computed.')
parser.add_argument('--NUMNUS', dest='NUM_NUS', default=50, help='Number of values of nu (the fairness parameter for FairCB) required for the experiments for varying values of nu.')
parser.add_argument('--NUMALPHAS', dest='NUM_ALPHAS', default=50, help='Number of values of alpha (the fairness parameter for our policy) required for the experiments for varying values of alpha.')
parser.add_argument('--VARYING_NU_ROUNDS', dest='VARYING_NU_ROUNDS', default=50, help='Number of rounds for the experiments for varying values of nu (the fairness parameter for the FairCB algorithm).')

args = parser.parse_args()
config_dict = {
    "ROWS": int(args.ROWS),
    "SEED": int(args.SEED),
    "ALPHA": float(args.ALPHA),
    "SMALLREWARD": float(args.SMALL_REWARD),
    "USETIMESTAMPS": args.USETIMESTAMPS,
    "FREQUENCY": int(args.FREQUENCY),
    "FREQUENCY_MAX": int(args.FREQUENCY_MAX),
    "FAIRCBFAIRNESS": float(args.FAIRCBFAIRNESS),
    "NUM_NUS": int(args.NUM_NUS),
    "NUM_ALPHAS": int(args.NUM_ALPHAS),
    "VARYING_NU_ROUNDS": int(args.VARYING_NU_ROUNDS),
    "USETEXLIVE": args.USETEXLIVE,
    "HIGHFREQUENCY": args.HIGHFREQUENCY,
    "PLOTREGRET": args.PLOTREGRET,
    "DATAPATH": "pickled_files/data.pickle",
    "CATEGORIES_PATH": "pickled_files/categories.pickle",
    "USER_TO_INDEX_PATH": "pickled_files/user_to_index.pickle",
    "INDEX_TO_USER_PATH": "pickled_files/index_to_user.pickle",
}

## save the config
with open('config.json', 'w') as f:
    json.dump(config_dict, f)

