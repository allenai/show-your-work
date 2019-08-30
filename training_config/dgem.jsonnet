{
   "numpy_seed": std.extVar("SEED"),
   "pytorch_seed": std.extVar("SEED"),
   "random_seed": std.extVar("SEED"),
    "dataset_reader": {
        "type": "entailment_tuple",
        "max_tuples": 30,
        "max_tokens": 200,
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },
        "tokenizer": {
            "end_tokens": []
        }
    },
  "train_data_path": "/home/suching/reproducibility/scitail/SciTailV1.1/dgem_format/scitail_1.0_structure_train.tsv",
  "validation_data_path": "/home/suching/reproducibility/scitail/SciTailV1.1/dgem_format/scitail_1.0_structure_dev.tsv",
  "test_data_path": "/home/suching/reproducibility/scitail/SciTailV1.1/dgem_format/scitail_1.0_structure_test.tsv",
    "model": {
        "type": "tree_attention",
        "use_encoding_for_node": false,
        "ignore_edges": false,
        "text_field_embedder": {
            "tokens": {
                "type": "embedding",
                "projection_dim": 100,
                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
                "embedding_dim": 300,
                "trainable": false
            }
        },
        "attention_similarity": {
            "type": "dot_product"
        },
        "premise_encoder": {
            "type": "lstm",
            "bidirectional": false,
            "num_layers": 1,
            "input_size": 100,
            "hidden_size": 100
        },
        "phrase_probability": {
            "input_dim": 400,
            "num_layers": 1,
            "hidden_dims": [2],
            "activations":['relu'],
            "dropout": 0.5,
        },
        "edge_probability": {
            "input_dim": 210,
            "num_layers": 1,
            "hidden_dims": 2,
            "activations": ['relu'],
            "dropout": 0.5,
        },
        "edge_embedding": {
            "vocab_namespace": "edges",
            "embedding_dim": 10
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_normal"}],
            [".*token_embedder_tokens._projection.*weight", {"type": "xavier_normal"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"], ["hypothesis", "num_tokens"]],
        "batch_size": 32
    },

    "trainer": {
        "num_epochs": 140,
        "grad_norm": 5,
        "patience": 20,
        "cuda_device": std.parseInt(std.extVar("CUDA_DEVICE")),
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adam",
        },
        "learning_rate_scheduler": {
            "type": "exponential",
            "gamma": 0.5
        }
    }
}
