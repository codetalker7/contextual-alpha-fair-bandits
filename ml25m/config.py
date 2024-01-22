import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--ROWS', dest='ROWS', default=50000, help='Number of rows of the dataset to use.')
parser.add_argument('--SEED', dest='SEED', default=42, help='Random seed to have reproducible results.')
parser.add_argument('--ALPHA', dest='ALPHA', default=0.5, help='Fairness level')
parser.add_argument('--SMALLREWARD', dest='SMALL_REWARD', default=0.001, help='Very small reward for the bad arm.')
parser.add_argument('--USETIMESTAMPS', dest='USETIMESTAMPS', action='store_true', help='Boolean determining whether the timestamps given in the dataset will be used to shuffle the rows.')
parser.add_argument('--FREQUENCY', dest='FREQUENCY', default=5000, help='Minimum frequency of a context in the resultant dataset.')
parser.add_argument('--FAIRCBFAIRNESS', dest='FAIRCBFAIRNESS', default=0.5, help='Fairness parameter to be used for the FairCB algorithm.')
parser.add_argument('--USETEXLIVE', dest='USETEXLIVE', action='store_true', help='Whether to use TexLive during plot generation.')

args = parser.parse_args()
config_dict = {
    "ROWS": int(args.ROWS),
    "SEED": int(args.SEED),
    "ALPHA": float(args.ALPHA),
    "SMALLREWARD": float(args.SMALL_REWARD),
    "USETIMESTAMPS": args.USETIMESTAMPS,
    "FREQUENCY": int(args.FREQUENCY),
    "FAIRCBFAIRNESS": float(args.FAIRCBFAIRNESS),
    "USETEXLIVE": args.USETEXLIVE,
    "DATAPATH": "pickled_files/data.pickle",
    "CATEGORIES_PATH": "pickled_files/categories.pickle",
    "USER_TO_INDEX_PATH": "pickled_files/user_to_index.pickle",
    "INDEX_TO_USER_PATH": "pickled_files/index_to_user.pickle",
}

## save the config
with open('config.json', 'w') as f:
    json.dump(config_dict, f)

