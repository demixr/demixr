import os
import glob

def create_musdbhq_dict(dataset):
    subsets = list()

    for subset in ["train", "test"]:
        tracks = glob.glob(os.path.join(dataset, subset, "*"))
        samples = list()

        # Go through tracks
        for track_folder in sorted(tracks):
            # Skip track if mixture is already written, assuming this track is done already
            song = dict()
            for stem in ["mix", "bass", "drums", "other", "vocals"]:
                filename = stem if stem != "mix" else "mixture"
                audio_path = os.path.join(track_folder, filename + ".wav")
                song[stem] = audio_path

            samples.append(song)

        subsets.append(samples)

    return subsets
