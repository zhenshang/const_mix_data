import os
import shutil
import argparse

splits = ['dev', 'tst-COMMON', 'tst-HE', 'train']
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main(args):
    seg_path = args.data_root
    for split in splits:
        split_path = os.path.join(seg_path, split)
        for f in os.listdir(split_path):
            if f.startswith('ted'):
                speaker = f.split('_')[1]
                speaker_dir = os.path.join(split_path, speaker)
                os.makedirs(speaker_dir, exist_ok=True)
                shutil.move(os.path.join(split_path, f), speaker_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    args = parser.parse_args()
    main(args)