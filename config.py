class Config:
    def __init__(self):
        self.hidden_size = 700
        self.learning_rate = 5e-4
        self.lambda_g=0.1
        self.gamma_decay=0.5
        
        self.model = {
            "dim_c": 200,
            "dim_z": 500,
            "embedder": {
            "dim": 100,
            },
            "encoder": {
            "rnn_cell": {
                "type": "GRUCell",
                "kwargs": {
                    "num_units": 700
                },
                "dropout": {
                    "input_keep_prob": 0.5
                }
            }
            },
            "decoder": {
            "rnn_cell": {
                "type": "GRUCell",
                "kwargs": {
                    "num_units": 700,
                },
                "dropout": {
                    "input_keep_prob": 0.5,
                    "output_keep_prob": 0.5
                },
            },
            "attention": {
                "type": "BahdanauAttention",
                "kwargs": {
                    "num_units": 700,
                },
                "attention_layer_size": 700,
            },
            "max_decoding_length_train": 21,
            "max_decoding_length_infer": 20,
            },
            "classifier": {
            "kernel_size": [3, 4, 5],
            "filters": 128,
            "other_conv_kwargs": {"padding": "same"},
            "dropout_conv": [1],
            "dropout_rate": 0.5,
            "num_dense_layers": 0,
            "num_classes": 1
            },
            "opt": {
            "optimizer": {
                "type":  "AdamOptimizer",
                "kwargs": {
                    "learning_rate": 5e-4,
                },
            },
            },
            }