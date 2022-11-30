import os
import csv
import tqdm
import argparse
import pandas as pd
import string


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
    output_file = os.path.join(root, split + '_raw_seg_plus.tsv')
    output_table = load_df_from_tsv(output_file)
    # prob = open('/data/zxh/st/STEMM/data/mustc/en-de/output/'+ f'outprob-en{args.lang}.txt','r')
    # word = open('/data/zxh/st/STEMM/data/mustc/en-de/output/'+ f'outword-en{args.lang}.txt','r')
    prob = open(root+ f'outprob-en{args.lang}.txt','r')
    word = open(root+ f'outword-en{args.lang}.txt','r')
    align_num = open(root+ f'{args.lang}out.txt','r')
    # print(prob.readlines())
    output_table['prob'] = prob.readlines()
    output_table['word'] = word.readlines()
    output_table['align_num'] = align_num.readlines()
    # print(type(output_table))
    # print(output_table)
    save_df_to_tsv(output_table,root+f'align_en{args.lang}.tsv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--task", required=True, type=str, choices=["asr", "st"])
    parser.add_argument("--split", required=True)
    parser.add_argument("--lang", required=True, type=str)
    args = parser.parse_args()

    main(args)

