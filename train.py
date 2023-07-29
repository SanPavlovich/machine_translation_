import numpy as np
import itertools
import torch
from tqdm.notebook import tqdm
from datasets import load_metric
bleu_metric = load_metric("bleu")

import os

SOS_TOKEN = "<SOS>" # start of sentence token
EOS_TOKEN = "<EOS>" # end of sentence token
UNK_TOKEN = "<UNK>" # unknown token
PAD_TOKEN = "<PAD>" # padding token

WRITE_CNT_STEPS = 100
SCHEDULER_CNT_STEPS = 300

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def show_random_examples(input_data, out_answers, targets, dataloader, is_val):
    lst = [1, 3, -1] if is_val else [-1]
    # show random texts with three ids
    for i in lst:
        x_data = input_data[i]
        y_data = out_answers[i]
        target_data = targets[i]

        # get tokenizers
        x_tokenizer = dataloader.dataset.x_tokenizer
        y_tokenizer = dataloader.dataset.y_tokenizer

        # convert answers idx
        x_answers = x_tokenizer.convert_idx_to_tokens(x_data.cpu().detach().numpy().tolist())
        y_answers = y_tokenizer.convert_idx_to_tokens(y_data.cpu().detach().numpy().tolist())
        target_tokens = y_tokenizer.convert_idx_to_tokens(target_data.cpu().detach().numpy().tolist())

        print("X: ", ' '.join([x for x in x_answers]))
        print("Answer: ", ' '.join([y for y in y_answers]))
        print("Target: ", ' '.join([trg for trg in target_tokens]))


def correct_tokens_for_bleu(tokens_list):
    res_tokens_list = []
    for tok in tokens_list:
      if tok == EOS_TOKEN:
        break
      res_tokens_list.append(tok)

    return [res_tokens_list]


def bleu_calculation(targets_ids, all_labels, tokenizer, bleu_metric):
    targets_ids = np.array(list(itertools.chain(*targets_ids))) # concat list by axis=0
    all_labels = np.array(list(itertools.chain(*all_labels)))

    targets_ids = targets_ids.reshape(-1, targets_ids.shape[-1])
    all_labels = all_labels.reshape(-1, all_labels.shape[-1])

    targets_list = [
        correct_tokens_for_bleu(
            tokenizer.convert_idx_to_tokens(trg_idx_list)
        )[0] for trg_idx_list in targets_ids
    ]

    answers_list = [
        correct_tokens_for_bleu(
            tokenizer.convert_idx_to_tokens(label_list)
        ) for label_list in all_labels
    ]

    return bleu_metric.compute(predictions=targets_list, references=answers_list)['bleu']


def train_one_epoch(model, epoch_index, train_dataloader, scheduler, optimizer, loss_fn, #tb_writer, 
                    clip, cnt_classes, teacher_forcing_ratio):
    # train one epoch with scheduler, writer and clip
    running_loss = 0.
    last_loss = 0.
    train_loss = 0.
    i = 0

    all_labels = []
    all_answers = []

    for i, data in tqdm(enumerate(train_dataloader), desc='train loader'):
        input_data, labels = data

        optimizer.zero_grad()
        # move data to device
        input_data = input_data.to(device)
        labels = labels.to(device).long()

        # model inference with teacher_forcing
        outputs = model(input_data, labels, teacher_forcing_ratio=teacher_forcing_ratio)
        answers = outputs.argmax(axis=-1).squeeze(-1)
        loss = loss_fn(outputs.reshape(-1, cnt_classes), labels[:, :outputs.shape[1]].reshape(-1))

        # backward
        loss.backward()
        #grad clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        running_loss += loss.item()
        train_loss += loss.item()

        all_answers += [answers.cpu().detach().numpy().tolist()]
        all_labels += [labels.cpu().detach().numpy().tolist()]
        if i % WRITE_CNT_STEPS == WRITE_CNT_STEPS - 1 and epoch_index != -1:
            print("i =", i)
            # get answers idx by argmax
            show_random_examples(input_data, answers, labels, train_dataloader, is_val=False)

            # logging
            last_loss = running_loss / 100 # loss per batch
            tb_x = epoch_index * len(train_dataloader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

            print("train losses =", last_loss)
            if i % SCHEDULER_CNT_STEPS == 0 and scheduler is not None:
                #tb_writer.add_scalar('Scheduler LR', optimizer.param_groups[0]["lr"], tb_x)
                scheduler.step()

        i += 1
    bleu_score = bleu_calculation(all_answers, all_labels, train_dataloader.dataset.y_tokenizer, bleu_metric)
    print("train bleu:", bleu_score)
    train_loss = train_loss / len(train_dataloader)

    return last_loss, bleu_score


def validation(model, optimizer, loss_fn, epoch_number, val_dataloader, cnt_classes):
    val_loss = 0.0
    all_answers = []
    all_labels = []

    # eval model mode
    model.eval()
    with torch.no_grad():
        for i, vdata in tqdm(enumerate(val_dataloader)):
            input_data, labels = vdata

            optimizer.zero_grad()

            # move data to device
            input_data = input_data.to(device)
            labels = labels.to(device).long()

            # model inference
            outputs = model(input_data, None)

            outputs_reshape = outputs.reshape(-1, cnt_classes)

            # calc loss
            vloss = loss_fn(outputs_reshape, labels[:, :outputs.shape[1]].reshape(-1))
            val_loss += vloss.item()

            answers = outputs.argmax(axis=-1).squeeze(-1)
            all_answers += [answers.cpu().detach().numpy().tolist()]
            all_labels += [labels.cpu().detach().numpy().tolist()]
            # logging
            if i == 0:
                show_random_examples(input_data, answers, labels, val_dataloader, is_val=True)

    val_loss = val_loss / len(val_dataloader)

    bleu_score = bleu_calculation(all_answers, all_labels, val_dataloader.dataset.y_tokenizer, bleu_metric)

    print("val loss:", val_loss)
    print(f"val belu: {bleu_score}")
    return val_loss, bleu_score


def train(model, model_name, optimizer, train_dataloader, val_dataloader, scheduler, loss_fn, #writer, 
          clip, teacher_forcing_ratio, epochs, timestamp, cnt_classes):
    best_vloss = 1_000_000.

    for epoch_number in tqdm(range(epochs)):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        train_loss, train_bleu = train_one_epoch(model, epoch_number, train_dataloader, scheduler, optimizer, loss_fn, #tb_writer, 
                                                 clip, cnt_classes, teacher_forcing_ratio)

        model.train(False)
        val_loss, val_bleu = validation(model, optimizer, loss_fn, epoch_number, val_dataloader, cnt_classes)

        # logging
        #writer.add_scalar('Bleu/train', train_bleu, epoch_number)
        #writer.add_scalar('Loss/valid', val_loss, epoch_number)
        #writer.add_scalar('Bleu/valid', val_bleu, epoch_number)

        # save models if val_loss is better than best_vloss
        if val_loss < best_vloss:
            best_vloss = val_loss
            model_path = os.path.join(model_name, 'model_{}_{}.pkl'.format(epoch_number + 1, timestamp))
            torch.save(model.state_dict(), model_path)

    return model