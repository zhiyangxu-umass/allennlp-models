local bert_model = "bert-base-uncased";

{
    "dataset_reader": {
      "type": "srl_self_training",
      "bert_model_name": bert_model,
    },

    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 16
      }
    },

    "train_data_path": "dataset/train.10.pt",
    "aux_data_path": "dataset/train.aux-10.pt",
    "validation_data_path": "dataset/dev.all.pt",

    "model": {
        "type": "srl_self_training_baseline",
        "embedding_dropout": 0.05,
        "bert_model": bert_model,
    },

    "trainer": {
        "type": "self_training",
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
        },
        "checkpointer": {
            "keep_most_recent_by_count": 2,
        },
        "grad_norm": 1.0,
        "num_epochs": 40,
        "patience": 15,
        "validation_metric": "+f1-measure-overall",
        "mix_ratio": 0.1,
        "warm_up_epoch": -1,
        "weighted_self_training": true,
        "no_gold": false
    },
}
