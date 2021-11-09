import os
import glob


def get_dataset_paths(path):
    return {
        subset: [
            os.path.join(song_folder, stem + ".wav")
            for stem in ["mixture", "bass", "drums", "other", "vocals"]
            for song_folder in sorted(glob.glob(os.path.join(path, subset, "*")))
        ]
        for subset in ["train", "test"]
    }
