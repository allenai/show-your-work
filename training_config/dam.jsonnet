// Configuration for a textual entailment model based on:
//  Parikh, Ankur P. et al. “A Decomposable Attention Model for Natural Language Inference.” EMNLP (2016).
{
  "numpy_seed": std.extVar("SEED"),
  "pytorch_seed": std.extVar("SEED"),
  "random_seed": std.extVar("SEED"),
  "dataset_reader": {
    "type": "snli",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    },
    "tokenizer": {
      "end_tokens": ["@@NULL@@"]
    }
  },
  "train_data_path": "/home/suching/scitail/SciTailV1.1/snli_format/scitail_1.0_train.txt",
  "validation_data_path": "/home/suching/scitail/SciTailV1.1/snli_format/scitail_1.0_dev.txt",
  "test_data_path": "/home/suching/scitail/SciTailV1.1/snli_format/scitail_1.0_test.txt",
  "evaluate_on_test": std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1,
  "model": {
    "type": "decomposable_attention_modified",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "projection_dim": std.parseInt(std.extVar("PROJECTION_DIM")),
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "attend_feedforward": {
      "input_dim": std.parseInt(std.extVar("PROJECTION_DIM")),
      "num_layers": std.parseInt(std.extVar("ATTEND_FEEDFORWARD_NUM_LAYERS")),
      "hidden_dims": std.parseInt(std.extVar("ATTEND_FEEDFORWARD_HIDDEN_DIMS")),
      "activations": std.extVar("ATTEND_FEEDFORWARD_ACTIVATION"),
      "dropout": std.extVar("ATTEND_FEEDFORWARD_DROPOUT")
    },
    "similarity_function": {"type": "dot_product"},
    "compare_feedforward": {
      "input_dim": std.parseInt(std.extVar("PROJECTION_DIM")) * 2,
      "num_layers": std.parseInt(std.extVar("COMPARE_FEEDFORWARD_NUM_LAYERS")),
      "hidden_dims": std.parseInt(std.extVar("COMPARE_FEEDFORWARD_HIDDEN_DIMS")),
      "activations": std.extVar("COMPARE_FEEDFORWARD_ACTIVATION"),
      "dropout": std.extVar("COMPARE_FEEDFORWARD_DROPOUT")
    },
    "aggregate_feedforward": {
      "input_dim": std.parseInt(std.extVar("COMPARE_FEEDFORWARD_HIDDEN_DIMS")) * 2,
      "num_layers": std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_NUM_LAYERS")),
      "hidden_dims": std.makeArray(std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_NUM_LAYERS")), function(i) std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_HIDDEN_DIMS"))),
      "activations": std.makeArray(std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_NUM_LAYERS")), function(i) std.extVar("AGGREGATE_FEEDFORWARD_ACTIVATION")),
      "dropout": std.extVar("AGGREGATE_FEEDFORWARD_DROPOUT")
    },
    "output_layer": {
      "input_dim": std.parseInt(std.extVar("AGGREGATE_FEEDFORWARD_HIDDEN_DIMS")),
      "num_layers": 1,
      "hidden_dims": 2,
      "activations": "linear",
      "dropout": 0.0
    },
     "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens\\._projection.*weight", {"type": "xavier_normal"}]
     ]
   },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
    "batch_size": std.parseInt(std.extVar("BATCH_SIZE"))
  },

  "trainer": {
    "num_epochs": std.parseInt(std.extVar("NUM_EPOCHS")),
    "patience": 20,
    "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
    "grad_clipping": std.extVar("GRAD_CLIP"),
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 1,
    "optimizer": {
      "type": "adagrad",
      "lr": std.extVar("LEARNING_RATE")
    }
  }
}