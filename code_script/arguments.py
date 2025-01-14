import argparse
from typing import List
import torch
import transformers
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, \
                         AlbertTokenizer, AlbertConfig, AlbertModel

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertModel,
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': BertTokenizer,
        # 'tokenizer': RobertaTokenizer, # original
        'model':RobertaModel,
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model':AlbertModel,
    }
}

def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Relation Extraction.")

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default="albert", type=str, required=True, choices=_MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default="albert-xxlarge-v2", type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    parser.add_argument("--new_tokens", default=5, type=int, 
                        help="The output directory where the model predictions and checkpoints will be written")

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_for_new_token", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--temps", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")



    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    # args = argparse.Namespace(data_dir="/opt/ml/PTR/datasets/klue-re", \
    #     output_dir="/opt/ml/PTR/results/klue-re", \
    #     model_type="roberta", \
    #     model_name_or_path="jinmang2/roberta-large-re-tapt-20300", \
    #     per_gpu_train_batch_size=64, \
    #     per_gpu_eval_batch_size=128, \
    #     gradient_accumulation_steps=1, \
    #     max_seq_length=139, \
    #     warmup_steps=0, \
    #     learning_rate=2e-5, \
    #     learning_rate_for_new_token=1e-5, \
    #     num_train_epochs=4, \
    #     weight_decay=1e-2, \
    #     adam_epsilon=1e-6, \
    #     max_steps=-1,
    #     cache_dir="",
    #     max_grad_norm=1.0,
    #     seed=42,
    #     dropout_prob=0.1,
    #     new_tokens=5, 
    #     temps="temp.txt")
    # args.n_gpu = torch.cuda.device_count()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return _GLOBAL_ARGS

def get_model_classes():
    return _MODEL_CLASSES