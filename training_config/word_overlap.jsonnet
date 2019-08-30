{
  "numpy_seed": std.extVar("SEED"),
  "pytorch_seed": std.extVar("SEED"),
  "random_seed": std.extVar("SEED"),
  "dataset_reader": {
    "type": "simple_overlap",
    "tokenizer": {
      "end_tokens": [],
    }
  },
  "train_data_path": "/home/suching/scitail/SciTailV1.1/snli_format/scitail_1.0_train_pretokenized.txt",
  "validation_data_path": "/home/suching/scitail/SciTailV1.1/snli_format/scitail_1.0_dev_pretokenized.txt",
  "test_data_path": "/home/suching/scitail/SciTailV1.1/snli_format/scitail_1.0_test_pretokenized.txt",
  "evaluate_on_test": std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1,
  "model": {
    "type": "simple_overlap",
    "classifier": {
      "input_dim": 3,
      "num_layers": std.parseInt(std.extVar("NUM_LAYERS")),
      "hidden_dims": std.makeArray(std.parseInt(std.extVar("NUM_LAYERS")), function(i) std.parseInt(std.extVar("HIDDEN_DIM"))),
      "activations": std.makeArray(std.parseInt(std.extVar("NUM_LAYERS")), function(i) std.extVar("ACTIVATION")),
      "dropout": std.extVar("DROPOUT"),
    },
    "output_layer": {
      "input_dim": std.parseInt(std.extVar("HIDDEN_DIM")),
      "num_layers": 1,
      "hidden_dims": 2,
      "activations": "linear",
      "dropout": 0.0,
    },
    "initializer": [
      [".*linear_layers.*weight", {"type": "xavier_normal"}],
      [".*token_embedder_tokens._projection.*weight", {"type": "xavier_normal"}]
    ]
  },
  "iterator": {
    "type": "basic",
    "batch_size": std.parseInt(std.extVar("BATCH_SIZE"))
  },

  "trainer": {
    "num_epochs": 140,
    "grad_norm": std.extVar("GRAD_NORM"),
    "patience": 20,
    "cuda_device": 0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
      "lr": std.extVar("LEARNING_RATE")
    },
    "learning_rate_scheduler": {
      "type": "exponential",
      "gamma": 0.5
    }
  }
}