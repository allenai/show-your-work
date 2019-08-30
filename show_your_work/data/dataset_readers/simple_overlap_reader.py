import json
import logging
from typing import Dict

import tqdm
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, ArrayField
from allennlp.data.fields.metadata_field import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from nltk.util import skipgrams, ngrams
from overrides import overrides
import numpy as np

from show_your_work.data.fields.features_field import FeaturesField

logger = logging.getLogger(__name__)


@DatasetReader.register("simple_overlap")
class SimpleOverlapReader(DatasetReader):
    """
    A reader that converts an entailment dataset into simple overlap statistics that can be used
    as input features by a neural network. It reads the JSONL or TSV format to create three
    features: trigram overlap, bigram overlap, unigram overlap. The trigrams and bigrams allow
    for one skip word. The overlap is normalized by the number of corresponding n-grams in the
    hypothesis.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, 'r') as snli_file:
            logger.info("Reading instances from tsv/jsonl dataset at: %s", file_path)
            for line in snli_file:
                if file_path.endswith(".txt"):
                    # SNLI format
                    example = json.loads(line)
                    label = example["gold_label"]
                    premise = example["sentence1"]
                    hypothesis = example["sentence2"]
                    features = example["features"]
                else:
                    # DGEM/TSV format
                    fields = line.split("\t")
                    premise = fields[0]
                    hypothesis = fields[1]
                    label = fields[2]
                if label == '-':
                    # ignore unknown examples
                    continue
                instance = self.text_to_instance(np.array(features), premise, hypothesis, label)
                yield instance

    @overrides
    def text_to_instance(self,
                         features: np.ndarray,
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        fields['features'] = ArrayField(features)
        metadata = {
            'premise': premise,
            'hypothesis': hypothesis,
        }
        fields['metadata'] = MetadataField(metadata)
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)
