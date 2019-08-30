from environments import ENVIRONMENTS
from environments.random_search import HyperparameterSearch
import os
import shutil
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-o',
                        '--override',
                        action="store_true",
                        help='remove the specified serialization dir before training')
    parser.add_argument('-s', '--serialization_dir', type=str, help='model serialization directory', required=True)
    parser.add_argument('-e', '--environment', type=str, help='hyperparameter environment', required=True)
    parser.add_argument('-i', '--input', type=str, help='input file', required=False)
    parser.add_argument('-d', '--dev_file', type=str, help='dev file', required=False)
    parser.add_argument('-n', '--num_assignments', type=int, help='number of assignments', required=True)

    args = parser.parse_args()

    if os.path.exists(args.serialization_dir) and args.override:
        print(f"overriding {args.serialization_dir}")
        shutil.rmtree(args.serialization_dir)
    else:
        pathlib.Path(args.serialization_dir).mkdir(parents=True, exist_ok=True)

    dev = pd.read_json(args.dev_file, lines=True)
    train = pd.read_json(args.input, lines=True)
    df = pd.DataFrame()
    print("creating feature matrix...")
    master = pd.concat([train, dev], 0)    
    print("done!")
    pbar = tqdm(range(args.num_assignments))
    for i in pbar:
        pathlib.Path(args.serialization_dir + f"/run_{i}/").mkdir(parents=True, exist_ok=True)
        env = ENVIRONMENTS[args.environment.upper()]
        space = HyperparameterSearch(**env)
        sample = space.sample()
        if sample.pop('stopwords') == 1:
            stop_words = 'english'
        else:
            stop_words = None
        weight = sample.pop('weight')
        if weight == 'binary':
            binary = True
        else:
            binary = False
        ngram_range = sample.pop('ngram_range')
        ngram_range = sorted([int(x) for x in ngram_range.split()])
        if weight == 'tf-idf':
            vect = TfidfVectorizer(stop_words=stop_words, lowercase=True, ngram_range=ngram_range)
        else:
            vect = CountVectorizer(binary=binary, stop_words=stop_words,lowercase=True, ngram_range=ngram_range)
        start = time.time()
        vect.fit(master.text)
        X_train = vect.transform(train.text)
        X_dev = vect.transform(dev.text)
        model_file = args.serialization_dir + f"/run_{i}/model"
        sample['C'] = float(sample['C'])
        sample['tol'] = float(sample['tol'])
        classifier = LogisticRegression(**sample)
        classifier.fit(X_train, train.label)
        end = time.time()
        for k, v in sample.items():
            sample[k] = [v]
        sub_df = pd.DataFrame(sample)
        sub_df['accuracy'] = classifier.score(X_dev, dev.label)
        sub_df['training_duration'] = end - start
        sub_df['ngram_range'] = str(ngram_range)
        sub_df['weight'] = weight
        sub_df['stopwords'] = stop_words
        df = pd.concat([df, sub_df], 0)
        best_trial = df.reset_index().iloc[df.reset_index().accuracy.idxmax()]
        pbar.set_description(f"best accuracy: {best_trial['accuracy']}")
    df['dataset_reader.sample'] = train.shape[0]
    df['model.encoder.architecture.type'] = 'logistic regression'
    df.to_csv(args.serialization_dir + "/results.tsv", sep='\t', index=False)
    print(f"best accuracy: {df.accuracy.max()}")