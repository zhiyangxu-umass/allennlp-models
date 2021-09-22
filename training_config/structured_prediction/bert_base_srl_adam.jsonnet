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

    "train_data_path": "dataset/train.3.filter.pt",
    "aux_data_path": "dataset/train.all.filter.pt",
    "validation_data_path": "dataset/dev.all.pt",

    "model": {
        "type": "srl_self_training_baseline",
        "embedding_dropout": 0.05,
        "bert_model": bert_model,
    },

    "trainer": {
        "type": "self_training",
        "optimizer": {
            "type": "adam",
            "lr": 0.005,
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
        "mix_ratio": 1.0,
        "warm_up_epoch": -1,
        "weighted_self_training": true,
        "no_gold": false
    },
}
