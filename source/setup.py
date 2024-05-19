def setup_config(lr=0.005, batch_size=64, epochs=30, seed=2024, image_size=(256, 256)):
    config ={
        'LR': lr,
        'BATCH_SIZE': batch_size,
        'EPOCHS': epochs,
        'SEED':seed,
        'IMAGE_SIZE':image_size
    }
    return config