from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio
import csv
from hparams import hparams
import traceback


def build_from_path(in_dir, out_dir, speakers, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []

    index = 1
    for speaker_id, speaker in enumerate(speakers):
        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        with open(os.path.join(in_dir, speaker, 'sentences_train.csv'), encoding='utf-8', errors='replace') as f:
            for row in csv.reader(f, delimiter='|', escapechar=None, quotechar=None):
                wav_path = os.path.join(in_dir, speaker, row[0])
                text = row[1]
                futures.append(executor.submit(partial(_process_utterance, out_dir, index, speaker, speaker_id,
                                                       wav_path,
                                                       text)))
                index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, speaker, speaker_id, wav_path, text):
    # Load the audio to a numpy array:
    try:
        wav = audio.load_wav(wav_path)

        if hparams.rescaling:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max

        # Compute the linear-scale spectrogram from the wav:
        spectrogram = audio.spectrogram(wav).astype(np.float32)
        n_frames = spectrogram.shape[1]

        # Compute a mel-scale spectrogram from the wav:
        mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

        # Write the spectrograms to disk:
        spectrogram_filename = '%s/er_m-spec-%05d.npy' % (speaker, index)
        mel_filename = '%s/er_m-mel-%05d.npy' % (speaker, index)
        np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
        np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

        # Return a tuple describing this training example:
        return spectrogram_filename, mel_filename, n_frames, text, speaker_id
    except Exception:
        print("Error with the following file:", wav_path)
        print("Error:", traceback.format_exc())
        print(text)
        return "", "", 0, text, speaker_id
