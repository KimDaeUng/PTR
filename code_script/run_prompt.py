from numpy.core.numeric import full
from arguments import get_args_parser
from templating import get_temps
from modeling import get_model, get_tokenizer
from data_prompt import REPromptDataset
from optimizing import get_optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torchsampler import ImbalancedDatasetSampler

from tqdm import tqdm, trange
import numpy as np
import pandas as pd
from collections import Counter
import random
import wandb
import os

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import auc, accuracy_score, recall_score, precision_score

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold


def kfold_split(dataset, n_splits=5, fold=1, random_state=42):
    full_df = dataset
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.arange(len(full_df))
    for fold_idx, (train_indices, valid_indices) in enumerate(kfold.split(indices, full_df[:]['label']), 1): # fold: [1, n_splits]
        if fold_idx == fold:
            print("="*20 + f" fold: {fold_idx} "+ "="*20)
            train_dataset = Dataset.from_pandas(full_df.iloc[train_indices])
            eval_dataset = Dataset.from_pandas(full_df.iloc[valid_indices])
            break
    return train_dataset, eval_dataset


RELATION_CLASS = [
    'no_relation', 
    'org:top_members/employees',
    'org:members',
    'org:product',
    'per:title',
    'org:alternate_names',
    'per:employee_of',
    'org:place_of_headquarters',
    'per:product',
    'org:number_of_employees/members',
    'per:children',
    'per:place_of_residence', 
    'per:alternate_names',
    'per:other_family',
    'per:colleagues',
    'per:origin', 
    'per:siblings',
    'per:spouse',
    'org:founded',
    'org:political/religious_affiliation',
    'org:member_of',
    'per:parents',
    'org:dissolved',
    'per:schools_attended',
    'per:date_of_death', 
    'per:date_of_birth',
    'per:place_of_birth',
    'per:place_of_death',
    'org:founded_by',
    'per:religion'
]
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = None, gamma: float = 0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def get_confusion_matrix(logits, labels):
    preds = np.argmax(logits, axis=1).ravel()
    cm = confusion_matrix(labels, preds)
    norm_cm = cm / np.sum(cm, axis=1)[:,None]
    cm = pd.DataFrame(norm_cm, index=RELATION_CLASS, columns=RELATION_CLASS)
    fig = plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True)
    return fig

def f1_score(output, label, rel_num, na_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    
    # f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        # if recall + precision > 0 :
        #     f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)

    return micro_f1 # , f1_by_relation

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score)

def evaluate(model, dataset, dataloader):
    model.eval()
    scores = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            logits = model(**batch)
            res = []
            for i in dataset.prompt_id_2_label:
                _res = 0.0
                for j in range(len(i)):
                    _res += logits[j][:, i[j]]                
                _res = _res.detach().cpu()
                res.append(_res)
            logits = torch.stack(res, 0).transpose(1,0)
            labels = batch['labels'].detach().cpu().tolist()
            all_labels+=labels
            scores.append(logits.cpu().detach())
        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy() 
        all_labels = np.array(all_labels)
        np.save("scores.npy", scores)
        np.save("all_labels.npy", all_labels)

        # Compute confusion matrix
        cm_fig = get_confusion_matrix(scores, np.array(all_labels))
        wandb.log({'confusion matrix': wandb.Image(cm_fig)})

        pred = np.argmax(scores, axis = -1)
        mi_f1 = f1_score(pred, all_labels, dataset.num_class, dataset.NA_NUM)
        # print(pred)
        # print(all_labels)
        # auprc = klue_re_auprc(pred, all_labels)
    return mi_f1 #, auprc


args = get_args_parser()
os.environ["WANDB_PROJECT"] = 'klue_ptr'
wandb.init(config=args,entity='kiyoung2',name='TAPT_Aug2_roberta-large_tapt_bs64_maxlen149_focal_ws300_lr5e-05_lrt3e-05')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)
tokenizer = get_tokenizer(special=[])
temps = get_temps(tokenizer)

# If the dataset has been saved, 
# the code ''dataset = REPromptDataset(...)'' is not necessary.
dataset = REPromptDataset(
    path  = args.data_dir, 
    name = 'train.txt', 
    rel2id = args.data_dir + "/" + "rel2id.json", 
    temps = temps,
    tokenizer = tokenizer,)
dataset.save(path = args.output_dir, name = "train")

# If the dataset has been saved, 
# the code ''dataset = REPromptDataset(...)'' is not necessary.
dataset = REPromptDataset(
    path  = args.data_dir, 
    name = 'val.txt', 
    rel2id = args.data_dir + "/" + "rel2id.json", 
    temps = temps,
    tokenizer = tokenizer)
dataset.save(path = args.output_dir, name = "val")

# If the dataset has been saved, 
# the code ''dataset = REPromptDataset(...)'' is not necessary.
dataset = REPromptDataset(
    path  = args.data_dir, 
    name = 'test.txt', 
    rel2id = args.data_dir + "/" + "rel2id.json", 
    temps = temps,
    tokenizer = tokenizer)
dataset.save(path = args.output_dir, name = "test")

train_dataset = REPromptDataset.load(
    path = args.output_dir, 
    name = "train", 
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")

val_dataset = REPromptDataset.load(
    path = args.output_dir, 
    name = "val", 
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")

test_dataset = REPromptDataset.load(
    path = args.output_dir, 
    name = "test", 
    temps = temps,
    tokenizer = tokenizer,
    rel2id = args.data_dir + "/" + "rel2id.json")

train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

train_dataset.cuda()

# print(type(train_dataset[:]['labels'].values))
def get_label(dataset):
    return dataset[:]['labels'].cpu().numpy()
# train_sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label=get_label)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

val_dataset.cuda()
val_sampler = SequentialSampler(val_dataset)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=train_batch_size)

test_dataset.cuda()
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=train_batch_size)

model = get_model(tokenizer, train_dataset.prompt_label_idx)
optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(gamma=0.5)

wandb.watch(model, log_freq=1000)
mx_res = 0.0
hist_mi_f1 = []
# hist_auprc = []
mx_epoch = None
last_epoch = None

for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
    model.train()
    model.zero_grad()
    tr_loss = 0.0
    global_step = 0 
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        logits = model(**batch)
        labels = train_dataset.prompt_id_2_label[batch['labels']] # (Batch, N_MASK=5)
        
        loss = 0.0
        
        for index, i in enumerate(logits):
            # i : (Batch, N_MLM_Head_label=(N_subj_entity_type, N_Relation_prompt_tokens, N_obj_entity_type))
            loss += criterion(i, labels[:,index])
        loss /= len(logits)
        # wandb.log({'train/EntityTypeLoss':loss.detach()})

        # Relation Label Loss
        res = []
        for i in train_dataset.prompt_id_2_label:
            _res = 0.0
            for j in range(len(i)):
                _res += logits[j][:, i[j]]
            res.append(_res)
        final_logits = torch.stack(res, 0).transpose(1,0) # [Batch, N_Relation_Label]

        loss += criterion(final_logits, batch['labels'])

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer_new_token.step()
            scheduler_new_token.step()
            model.zero_grad()
            # print (args)
            global_step += 1
            print (tr_loss/global_step, mx_res)

    mi_f1 = evaluate(model, val_dataset, val_dataloader)
    hist_mi_f1.append(mi_f1)
    # hist_auprc.append(auprc)
    
    wandb.log({'train/loss':tr_loss/global_step})
    wandb.log({'eval/micro_f1':mi_f1})
    
    if mi_f1 > mx_res:
        mx_res = mi_f1
        mx_epoch = epoch
        torch.save(model.state_dict(), args.output_dir+"/"+'parameter'+str(epoch)+".pkl")
    last_epoch = epoch

print(hist_mi_f1)

model.load_state_dict(torch.load(args.output_dir+"/"+'parameter'+str(last_epoch)+".pkl"))
mi_f1 = evaluate(model, test_dataset, test_dataloader)

print(mi_f1)