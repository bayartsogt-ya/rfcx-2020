"""
PANNsDense121Att | 32k | Fold 0
"""
import os


# Arguments:
class Config:
    root = "./data/rfcx-species-audio-detection-sr-32k"
    output_dir = "./output"
    train_csv_path = os.path.join(root, "train_folds.csv")
    test_csv_path = os.path.join(root, "test.csv")
    train_period = 10
    test_period = 30
    num_epochs = 20
    batch_size = 64
    lr = 1e-2
    eta_min = 1e-4  # min learning rate

    transforms = []
    use_fold = 0
    num_classes = 24
    model_name = "PANNsDense121Att"
    model_config = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
        "classes_num": num_classes,
        "apply_aug": True,
    }
