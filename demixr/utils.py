import os
import glob

import librosa


def get_dataset_paths(path):
    return {
        subset: [
            {
                stem: os.path.join(song_folder, stem + ".wav")
                for stem in ["mixture", "bass", "drums", "other", "vocals"]
            }
            for song_folder in sorted(glob.glob(os.path.join(path, subset, "*")))
        ]
        for subset in ["train", "test"]
    }


def load_audio(path, sample_rate=22050, mono=True):
    audio, _ = librosa.load(path, sr=sample_rate, mono=mono, res_type="kaiser_fast")
    return torch.tensor(audio)
