def custom_lr_schedule(cur_step, emb_size=512, warmup_steps=500):
    """Constructs custom learning rate schedule.

    Learning rate changes during the training process should follow the next formula:
        learning_rate = d_model^(-0.5) * min((cur_step + 1)^(-0.5), (cur_step + 1) * warmup_steps^(-1.5)
    """
    learning_rate = emb_size ** (-0.5) * min((cur_step + 1) ** (-0.5), (cur_step + 1) * warmup_steps ** (-1.5))
    return learning_rate
