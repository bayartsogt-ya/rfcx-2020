import os
from src.train_sed import main


if __name__ == "__main__":

    # Arguments:
    class Config:
        root = "/Users/bayartsogtyadamsuren" + \
            "/Projects/kaggle/rfcx-2020/data/sample_data"
        train_csv_path = os.path.join(root, "train_tp.csv")
        test_csv_path = os.path.join(root, "test.csv")
        train_period = 10
        test_period = 30
        num_epochs = 1
        batch_size = 2
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

    main(Config)
