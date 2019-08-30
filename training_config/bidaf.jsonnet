// Configuration for an Elmo-augmented machine comprehension model based on:
//   Seo, Min Joon et al. "Bidirectional Attention Flow for Machine Comprehension."
//   ArXiv/1611.01603 (2016)
{
  "numpy_seed": std.extVar("SEED"),
  "pytorch_seed": std.extVar("SEED"),
  "random_seed": std.extVar("SEED"),
  "dataset_reader": {
    "type": "squad",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "character_tokenizer": {
          "byte_encoding": "utf-8",
          "start_tokens": [259],
          "end_tokens": [260]
        },
        "min_padding_length": 5
      }
    }
  },
  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-train-v1.1.json",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/squad/squad-dev-v1.1.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                "embedding_dim": 100,
                "trainable": false
            },
            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                "num_embeddings": 262,
                "embedding_dim": std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM"))
                },
                "encoder": {
                "type": "cnn",
                "embedding_dim": std.parseInt(std.extVar("CHARACTER_EMBEDDING_DIM")),
                "num_filters": std.parseInt(std.extVar("NUM_FILTERS")),
                "ngram_filter_sizes": std.range(1, std.parseInt(std.extVar("MAX_FILTER_SIZE"))),
                },
                "dropout": std.extVar("CHARACTER_DROPOUT")
            }
        }
    },
    "num_highway_layers": std.parseInt(std.extVar("NUM_HIGHWAY_LAYERS")),
    "phrase_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100 + std.parseInt(std.extVar("NUM_FILTERS")) * std.parseInt(std.extVar("MAX_FILTER_SIZE")),
      "hidden_size": std.parseInt(std.extVar("PHRASE_LAYER_HIDDEN_SIZE")),
      "num_layers": std.parseInt(std.extVar("NUM_PHRASE_LAYERS")),
      "dropout": std.extVar("PHRASE_LAYER_DROPOUT")
    },
    "similarity_function": {
      "type": "linear",
      "combination": "x,y,x*y",
      "tensor_1_dim": std.parseInt(std.extVar("PHRASE_LAYER_HIDDEN_SIZE")) * 2,
      "tensor_2_dim": std.parseInt(std.extVar("PHRASE_LAYER_HIDDEN_SIZE")) * 2
    },
    "modeling_layer": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": std.parseInt(std.extVar("PHRASE_LAYER_HIDDEN_SIZE")) * 2 * 4,
      "hidden_size": std.parseInt(std.extVar("MODELING_LAYER_HIDDEN_SIZE")),
      "num_layers": std.parseInt(std.extVar("NUM_MODELING_LAYERS")),
      "dropout": std.extVar("MODELING_LAYER_DROPOUT")
    },
    "span_end_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 4 * 2 * std.parseInt(std.extVar("PHRASE_LAYER_HIDDEN_SIZE")) + 3 * std.parseInt(std.extVar("MODELING_LAYER_HIDDEN_SIZE")) * 2,
      "hidden_size": std.parseInt(std.extVar("SPAN_END_ENCODER_HIDDEN_SIZE")),
      "num_layers": std.parseInt(std.extVar("SPAN_END_ENCODER_NUM_LAYERS")),
      "dropout": std.extVar("SPAN_END_ENCODER_DROPOUT")
    },
    "dropout": std.extVar("DROPOUT")
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["passage", "num_tokens"], ["question", "num_tokens"]],
    "batch_size": 16
  },

  "trainer": {
    "num_epochs": 20,
    "grad_norm": std.extVar("GRAD_NORM"),
    "patience": 10,
    "validation_metric": "+em",
    "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "optimizer": {
      "type": "adam",
      "betas": [std.extVar("BETA_1"), std.extVar("BETA_2")],
      "lr": std.extVar("LEARNING_RATE")
    }
  }
}