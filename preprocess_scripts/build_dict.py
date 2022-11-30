import os
import csv
import tqdm
import argparse
import pandas as pd
import string
import torch
import soundfile as sf
from typing import BinaryIO, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
SF_AUDIO_FILE_EXTENSIONS = {".wav", ".flac", ".ogg"}
FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}


def get_waveform(
    path_or_fp: Union[str, BinaryIO],
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always_2d: bool = True,
    output_sample_rate: Optional[int] = None,
    normalize_volume: bool = False,
    ) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
        output_sample_rate (Optional[int]): output sample rate
        normalize_volume (bool): normalize volume
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    if isinstance(path_or_fp, str):
        ext = Path(path_or_fp).suffix
        if ext not in SF_AUDIO_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Please install soundfile: pip install soundfile")

    waveform, sample_rate = sf.read(
        path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
    )
    waveform = waveform.T  # T x C -> C x T
    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=normalize_volume,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )

    if not normalization:
        waveform *= 2**15  # denormalized to 16-bit signed integers
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform, sample_rate

def convert_waveform(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    normalize_volume: bool = False,
    to_mono: bool = False,
    to_sample_rate: Optional[int] = None,
) -> Tuple[Union[np.ndarray, torch.Tensor], int]:
    """convert a waveform:
    - to a target sample rate
    - from multi-channel to mono channel
    - volume normalization

    Args:
        waveform (numpy.ndarray or torch.Tensor): 2D original waveform
            (channels x length)
        sample_rate (int): original sample rate
        normalize_volume (bool): perform volume normalization
        to_mono (bool): convert to mono channel if having multiple channels
        to_sample_rate (Optional[int]): target sample rate
    Returns:
        waveform (numpy.ndarray): converted 2D waveform (channels x length)
        sample_rate (float): target sample rate
    """
    try:
        import torchaudio.sox_effects as ta_sox
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    effects = []
    if normalize_volume:
        effects.append(["gain", "-n"])
    if to_sample_rate is not None and to_sample_rate != sample_rate:
        effects.append(["rate", f"{to_sample_rate}"])
    if to_mono and waveform.shape[0] > 1:
        effects.append(["channels", "1"])
    if len(effects) > 0:
        is_np_input = isinstance(waveform, np.ndarray)
        _waveform = torch.from_numpy(waveform) if is_np_input else waveform
        converted, converted_sample_rate = ta_sox.apply_effects_tensor(
            _waveform, sample_rate, effects
        )
        if is_np_input:
            converted = converted.numpy()
        return converted, converted_sample_rate
    return waveform, sample_rate

def replace(output, origin):
    output_list = output.split(',')
    origin_list = origin.split(' ')
    output_index = [idx for idx, word in enumerate(output_list) if word != '']
    if len(output_index) != len(origin_list):
        return None
    for idx, word in enumerate(origin_list):
        output_list[output_index[idx]] = word
    return ','.join(output_list)
    
def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        index=False, 
    )

def load_df_from_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

def main(args):
    root = args.data_root
    lang = args.lang
    split = args.split
    dict_save_path = args.dict_save_path
    output_file = os.path.join(root+'align_ende_test.tsv')
    output_table = load_df_from_tsv(output_file)
    # print(output_table.columns)
    if dict_save_path is not None:
        dict_save_path = root + dict_save_path
        if not  os.path.exists(dict_save_path):
            os.mkdir(dict_save_path)
    add1 = []
    add2 = []
    add3 = []
    add4 = []
    add5 = []
    add6 = []

    ids ={}
    audios = {}
    nframess ={}
    speakers ={}
    MANIFEST_COLUMNS = ["src_text","id", "audio", "n_frames","speaker"]
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for i in range(len(output_table)):
        print(i)
        sword = []
        tword = []
        
        _,id,word_time,word_text,audio,n_frames,src_text,tgt_text,speaker,audio_align,text_align,prob,word,align_num = output_table.iloc[i]
        # data/mustc/en-de/data/train/wav/ted_1.wav:779200:367200
        wavpath, offset, n_frames = audio.rsplit(':',2)
        wavpath2 = wavpath
        
        lprob = prob.rstrip().split(' ')
        lword = word.rstrip().split(' ')
        for tmp in lword:
            st,tt=tmp.split('<sep>')
            sword.append(st)
            tword.append(tt)

        lword_text = word_text.split(',')
        lword_time = word_time.split(',')

        wavlist = []
        src_wavlist = []
        framelist = []
        wavlist2 = []
        trg_word_list = []
        time_list = []
        if  len(lword_text)!=len(lword_time) or len(lprob)!=len(lword):
            continue
        for wo in sword:
            if '/' in wo:
                continue
            if sword.count(wo)==1:
                idx=sword.index(wo)
                trg = tword[idx]
                if '/' in trg:
                    continue
                if tword.count(trg)==1 and float(lprob[idx])>0.99:
                    idx2 = lword_text.index(wo)
                    stime = float(lword_time[idx2-1] if idx2>0 else 0)
                    # if i > 0:
                    #     stime = lword_time[i-1]
                    # else:
                    #     stime = 0
                    etime = float(lword_time[idx2])
                    alltime = float(etime - stime)
                    
                    sample_rate = sf.info(wavpath2).samplerate
                    # print(sample_rate)
                    try:

                        iframe = 0
                        start = int(stime*int(sample_rate))+int(offset)
                        frames = int(alltime*int(sample_rate))
                        waveform, _ = get_waveform(wavpath2, frames=frames,start=start)
                        waveform = torch.from_numpy(waveform)
                        iframe = int(alltime*int(sample_rate))
                        if dict_save_path is not None:
                            sf.write(
                                dict_save_path + f"/{id}_{wo}.wav",
                                waveform.squeeze(0).numpy(),
                                samplerate=int(sample_rate)
                            )
                            wavlist.append(dict_save_path + f"/{id}_{wo}.wav")
                        else:
                            wavlist.append(None)
                        src_wavlist.append(wo)
                        framelist.append(str(iframe))
                        wavlist2.append(wavpath2+':'+str(start)+':'+str(frames))
                        trg_word_list.append(trg)
                        time_list.append(str(round(alltime,2)))


                       
                        for ss in string.punctuation:
                            wo = wo.replace(ss, "")
                        
                        wo = str.lower(wo)
                        # print(src_text)
                        if wo not in ids:
                            ids[wo] = []
                            audios[wo] = []
                            nframess[wo] = []
                            speakers[wo] = []
                            ids[wo].append(f"/{id}_{wo}.wav")
                            audios[wo].append(wavpath2+':'+str(start)+':'+str(frames))
                            nframess[wo].append(str(frames))
                            speakers[wo].append(speaker)
                        else:
                            ids[wo].append(f"/{id}_{wo}.wav")
                            audios[wo].append(wavpath2+':'+str(start)+':'+str(frames))
                            nframess[wo].append(str(frames))
                            speakers[wo].append(speaker)

                    except:
                        print('以下文件存在错误：')
                        print(wavpath2)

    #     if dict_save_path is not None:
    #         add1.append("###".join(wavlist))
    #     else:
    #         add1.append(wavlist)
    #     add2.append("###".join(src_wavlist))
    #     add3.append("###".join(framelist))
    #     add4.append("###".join(wavlist2))
    #     add5.append("###".join(trg_word_list))
    #     add6.append("###".join(time_list))
    # output_table['path_dict_list']=add1
    # output_table['src_dict_list']=add2
    # output_table['frame_list']=add3
    # output_table['path_dict_list2']=add4
    # output_table['trg_word_list']=add5
    # output_table['time_list']=add6

    
    for key,value in ids.items():
        manifest["src_text"].append(key)
        manifest["id"].append('###'.join(value))
        manifest["audio"].append('###'.join(audios[key]))
        manifest["n_frames"].append('###'.join(nframess[key]))
        manifest["speaker"].append('###'.join(speakers[key]))
    df = pd.DataFrame.from_dict(manifest)
    # print(len(df))
    save_df_to_tsv(df, root+"word_dict.tsv")
    # save_df_to_tsv(output_table,'/data/zxh/st/STEMM/data/mustc/en-de/dict_align_ende.tsv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--task", required=True, type=str, choices=["asr", "st"])
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dict-save-path", default=None, type=str,)
    args = parser.parse_args()

    main(args)

