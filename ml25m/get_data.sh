rm -rf data
mkdir data

wget https://files.grouplens.org/datasets/movielens/ml-25m-README.html -O data/ml-25m-README.html
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip -O data/ml-25m.zip

cd data
unzip ml-25m.zip

mkdir -p pickled_files
