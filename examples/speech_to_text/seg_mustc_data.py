#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import soundfile as sf
from itertools import groupby
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from audio_utils import get_waveform_part
log = logging.getLogger(__name__)

class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            if _lang == "en":
                filename = txt_root / f"{split}.{_lang}.clean"
            else:
                filename = txt_root / f"{split}.{_lang}"
            with open(filename) as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform_part(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)

def main(args):
    root = Path(args.data_root).absolute()
    lang = args.lang
    split = args.split
    cur_root = root / f"en-{lang}"
    assert cur_root.is_dir(), (
        f"{cur_root.as_posix()} does not exist. Skipped."
    )

    dataset = MUSTC(root.as_posix(), lang, split)
    output = Path(args.output).absolute()
    output.mkdir(exist_ok=True)
    f_text = open(output / f"{split}.{lang}", "w")
    f_wav_list = open(output / f"{split}.wav_list", "w")
    for waveform, sample_rate, trans, text, _, utt_id in tqdm(dataset):
        sf.write(
            output / f"{utt_id}.wav",
            waveform.squeeze(0).numpy(),
            samplerate=int(sample_rate)
        )
        with open(output / f"{utt_id}.lab", "w") as f:
            f.write(trans.upper())
        f_text.write(text + "\n")
        f_wav_list.write(str(output / f"{utt_id}.wav") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--task", required=True, type=str, choices=["asr", "st"])
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", required=True, choices=MUSTC.SPLITS)
    args = parser.parse_args()

    main(args)
