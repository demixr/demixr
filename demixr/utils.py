import os
import glob
import librosa
import torchaudio

from config import STEMS, MIX, EXT


def get_dataset_paths(path):
    return {
        subset: [
            {
                stem: {
                    "path": song_folder / f"{stem}{EXT}",
                    "info": get_file_info(song_folder / f"{stem}{EXT}"),
                }
                for stem in STEMS + [MIX]
            }
            for song_folder in sorted((path / subset).glob("*"))
        ]
        for subset in ["train", "test"]
    }


def get_file_info(filepath):
    info = torchaudio.info(str(filepath))
    return {"length": info.num_frames, "samplerate": info.sample_rate}


def load_audio(path, sample_rate=22050, mono=True):
    audio, _ = librosa.load(path, sr=sample_rate, mono=mono, res_type="kaiser_fast")
    return torch.tensor(audio)
