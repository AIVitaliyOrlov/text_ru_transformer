from easydict import EasyDict

model_cfg = EasyDict()
model_cfg.name = 'TextDecoderTransformer'

model_cfg.d_model = 512
model_cfg.d_ff = 2048
model_cfg.max_sequence_length = 4048
model_cfg.heads_num = 16
model_cfg.layers_num = 6
model_cfg.eps = 1e-5
model_cfg.dropout_rate = 0.1
model_cfg.pre_normalization = False
model_cfg.activation = 'GELU'  # from (ReLU, GELU)
model_cfg.attention_bias = False

