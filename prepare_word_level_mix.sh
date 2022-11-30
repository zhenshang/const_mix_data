# 以下代码基于mustc的数据实现,step1为语音和转录的对齐，step2为转录和目标文本的对齐，step3为构建字典
# 
# step 1:语音和转录的对齐,基于fairseq实现，使用mfa工具，安装方法见：https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
# 代码基于stemm，可见https://github.com/ictnlp/STEMM
echo "1. clean the source sentences in the corpus" 
#操作可选，仅用于英文，将阿拉伯数字等转换为英文字符以及去除标点符号等,在train.en文件目录下生成train.en.clean
python3 preprocess_scripts/clean_mustc.py --data-root data/mustc/en-${lang}/data/ --split train
echo "2. convert raw data into tsv manifest" 
#将mustc数据转换为tsv格式,data.tsv
python3 examples/speech_to_text/prep_mustc_data_raw.py --data-root data/mustc/ --tgt-lang ${lang} 
echo "3. split audio files"
#分割数据文件以用于做mfa，将每句话分割为一个语音数据wav文件以及对应的转录文本数据文件lab
mkdir -p data/mustc/en-${lang}/segment/
python3 examples/speech_to_text/seg_mustc_data.py --data-root data/mustc/ --task st --lang ${lang} --output data/mustc/en-${lang}/segment/train --split train
python3 examples/speech_to_text/seg_mustc_data.py --data-root data/mustc/ --task st --lang ${lang} --output data/mustc/en-${lang}/segment/dev --split dev
python3 examples/speech_to_text/seg_mustc_data.py --data-root data/mustc/ --task st --lang ${lang} --output data/mustc/en-${lang}/segment/tst-COMMON --split tst-COMMON
python3 examples/speech_to_text/seg_mustc_data.py --data-root data/mustc/ --task st --lang ${lang} --output data/mustc/en-${lang}/segment/tst-HE --split tst-HE
echo "4. group by speaker"
#将上述生成的wav数据以及lab数据按照speaker进行分组
python3 preprocess_scripts/group.py --data-root data/mustc/en-${lang}/segment/
echo "5. forced alignment"
#使用mfa工具对齐
cd data/mustc/en-${lang}/segment/ 
mfa align train english_mfa english_mfa train_align
mfa align dev english_mfa english_mfa dev_align
mfa align tst-COMMON english_mfa english_mfa tst-COMMON_align
mfa align tst-HE english_mfa english_mfa tst-HE_align
cd ../../../../
echo "6. convert textgrid format into tsv"
#将mfa生成的TextGrid格式的文件转换为tsv格式的文件用于后续操作，align.tsv
python3 preprocess_scripts/convert_format.py --data-root data/mustc/en-${lang}
echo "7. concatenate origin table and align table"
#将语音数据文件data.tsv和对齐信息文件align.tsv合并为一个tsv文件
python3 preprocess_scripts/postprocess_raw.py --data-root data/mustc/en-${lang}
echo "8. learn dictionary"
#生成字典，spm等等
python3 examples/speech_to_text/learn_dict_raw.py --data-root data/mustc/ --vocab-type unigram --vocab-size 10000 --tgt-lang ${lang} --split train
echo "9. calculate the start and end indexs of aligned word sequence"
#使用tokenizer等计算每个单词的语音、文本在输入当中的位置和结束位置
python3 preprocess_scripts/word_align_info_raw.py --data-root data/mustc/en-${lang}
# 最终生成train_raw_seg_plus.tsv,为模型输入




# step 2:转录和目标翻译的对齐，这里使用awesome-align工具：https://github.com/neulab/awesome-align
# 以下例子为ende，具体操作见上述网址，得到对齐概率outprob-ende.txt和对齐单词outword-ende.txt和对齐位置deout.txt
DATA_FILE=/home/zxh/align/awesome-align/mustc/ende.src-tgt
MODEL_NAME_OR_PATH=bert-base-multilingual-cased
OUTPUT_FILE=/home/zxh/align/awesome-align/mustc/deout.txt
CUDA_VISIBLE_DEVICES=3 awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 128 \
    --output_word_file /home/zxh/align/awesome-align/mustc/outword-ende.txt \
    --output_prob_file /home/zxh/align/awesome-align/mustc/outprob-ende.txt




# step3：生成单词级语音字典
echo "1. merge the step1 and step2 data" 
# 合并第一步得到的语音对齐和第二部得到的文本对齐数据为一个tsv数据,align_ende.tsv
# 要将outprob-ende.txt和outword-ende.txt和对齐位置deout.txt移到data-root
python3 -u preprocess_scripts/merge_data.py --data-root data/mustc/en-de/ --task st --lang ${lang} --split train
echo "2. build word level audio dict" 
# 对每一句语音数据，对每个单词三元组（语音单词，转录单词，目标单词），如果转录-目标对齐概率大于0.99，存储单词级位置信息
# dict-save-path 为存放单词语音的位置，可选，如果设置，会提取并存放word-level的语音数据，如果不设置，只会提取每个单词在sentence-level数据的起始frame和结束frame等信息
python3 -u preprocess_scripts/build_dict.py --data-root data/mustc/en-de/ --task st --lang ${lang} --split train --dict-save-path word_audio_dict
# 最终生成word_dict.tsv,为单词级语音字典

# 最终使用step1生成的train_raw_seg_plus.tsv作为模型训练集数据,step3生成的word_dict.tsv作为模型数据集进行单词替换的可查字典