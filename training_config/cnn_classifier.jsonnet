local USE_LAZY_DATASET_READER = std.parseInt(std.extVar("LAZY_DATASET_READER")) == 1;

// GPU to use. Setting this to -1 will mean that we'll use the CPU.
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

// Throttle the training data to a random subset of this length.
local THROTTLE = std.extVar("THROTTLE");

// Use the SpaCy tokenizer when reading in the data. If this is false, we'll use the just_spaces tokenizer.
local USE_SPACY_TOKENIZER = std.parseInt(std.extVar("USE_SPACY_TOKENIZER"));

// learning rate of overall model.
local LEARNING_RATE = std.extVar("LEARNING_RATE");

// dropout applied after pooling
local DROPOUT = std.extVar("DROPOUT");

local BATCH_SIZE = std.parseInt(std.extVar("BATCH_SIZE"));

local EMBEDDINGS = std.split(std.extVar("EMBEDDINGS"), " ");
local FREEZE_EMBEDDINGS = std.split(std.extVar("FREEZE_EMBEDDINGS"), " ");

local EVALUATE_ON_TEST = std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1;

local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));


local CNN_FIELDS(max_filter_size, embedding_dim, hidden_size, num_filters) = {
      "architecture": {
          "type": "cnn",
          "ngram_filter_sizes": std.range(1, max_filter_size),
          "num_filters": num_filters,
          "embedding_dim": embedding_dim,
          "output_dim": hidden_size, 
      }
};


local GLOVE_FIELDS(trainable) = {
  "glove_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "glove_embedder": {
    "tokens": {
        "embedding_dim": 50,
        "trainable": trainable,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
    }
  },
  "embedding_dim": 50
};


local GLOVE_TRAINABLE = if std.count(FREEZE_EMBEDDINGS, "GLOVE") > 0 == false then true  else false;

local TOKEN_INDEXERS = GLOVE_FIELDS(GLOVE_TRAINABLE)['glove_indexer'];

local TOKEN_EMBEDDERS =  GLOVE_FIELDS(GLOVE_TRAINABLE)['glove_embedder'];


local EMBEDDING_DIM =  GLOVE_FIELDS(GLOVE_TRAINABLE)['embedding_dim'];

local ENCODER = CNN_FIELDS(std.parseInt(std.extVar("MAX_FILTER_SIZE")), EMBEDDING_DIM, std.parseInt(std.extVar("HIDDEN_SIZE")), std.extVar("NUM_FILTERS"));


local SST_READER(TOKEN_INDEXERS, THROTTLE, USE_SPACY_TOKENIZER, USE_SUBTREES, USE_LAZY_DATASET_READER) = {
  "lazy": USE_LAZY_DATASET_READER,
  "type": "sst_tokens",
  "granularity": "5-class",
  "use_subtrees": USE_SUBTREES,
  "token_indexers": TOKEN_INDEXERS,
};

{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
   "dataset_reader": SST_READER(TOKEN_INDEXERS, THROTTLE, USE_SPACY_TOKENIZER, false, USE_LAZY_DATASET_READER),
   "validation_dataset_reader": SST_READER(TOKEN_INDEXERS, null, USE_SPACY_TOKENIZER, false, USE_LAZY_DATASET_READER),
   "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/train.txt",
   "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/dev.txt",
   "test_data_path": if EVALUATE_ON_TEST then "https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/test.txt" else null,
   "evaluate_on_test": EVALUATE_ON_TEST,
   "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
                "token_embedders": TOKEN_EMBEDDERS
      },
      "seq2vec_encoder": ENCODER['architecture'],
      "dropout": DROPOUT
   },	
    "iterator": {
      "batch_size": BATCH_SIZE,
      "type": "basic"
   },
   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_epochs": NUM_EPOCHS,
      "optimizer": {
         "lr": LEARNING_RATE,
         "type": "adam"
      },
      "learning_rate_scheduler": {
        "type": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 2
      },
      "patience": 10,
      "validation_metric": "+accuracy",
      "num_serialized_models_to_keep": 1
   }
}
