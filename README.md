# Deep Voice 3 adaptation for Estonian

This is a modified version of [Ryuichi Yamamoto's implementation](https://github.com/r9y9/deepvoice3_pytorch) of Deep Voice 3 to support Estonian text-to-speech. The original detailed readme is available in the README_original.md file.
 
 A Flask API implementation of this code is available [here](https://koodivaramu.eesti.ee/tartunlp/text-to-speech ) and the TTS can be tested with our [web demo](https://www.neurokone.ee).
 
The code contains a submodule for [Estonian TTS preprocessing](https://github.com/TartuNLP/tts_preprocess_et), therefore cloning with the `--recurse-submodules` flag is recommended.

# Pretrained models
Pretrained public model files are available in the [releases section](https://github.com/TartuNLP/deepvoice3_pytorch/releases). It is recommended using the same version of code, as other versions may not be compatible.

# Requirements:

- Python >= 3.5
- CUDA >= 8.0
- PyTorch >= v1.0.0
- [nnmnkwii](https://github.com/r9y9/nnmnkwii) >= v0.0.11
- [MeCab](http://taku910.github.io/mecab/) (Japanese only)
- EstNLTK (>= 1.6.0) (Estonian only)

## Installation

Please install packages listed above first, and then

```
git clone https://github.com/r9y9/deepvoice3_pytorch && cd deepvoice3_pytorch
pip install -e ".[bin]"
```

## Training
 
To train a multispeaker Estonian TTS model:

```
python preprocess.py eesti_konekorpus $data .data/eesti_konekorpus --speakers Mari,Kalev,Albert,Vesta,Külli,Meelis --preset=presets/eesti_konekorpus.json
python train.py --preset=presets/eesti_konekorpus.json --data-root=./data/eesti_konekorpus --checkpoint-dir=checkpoints/$modelname --log-event-path=log/$modelname
```

For detailed info about parameters, training, speaker adaptation and everything else, please refer to the original
 readme.