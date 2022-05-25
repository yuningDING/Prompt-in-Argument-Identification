import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from transformers import LongformerTokenizerFast, AutoTokenizer, LongformerForTokenClassification, AutoModel

pd.options.mode.chained_assignment = None


class Constants:
    OUTPUT_LABELS = ['0', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                     'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                     'I-Concluding Statement']
    LABELS_TO_IDS = {v: k for k, v in enumerate(OUTPUT_LABELS)}
    IDS_TO_LABELS = {k: v for k, v in enumerate(OUTPUT_LABELS)}

    MIN_THRESH = {
        "I-Lead": 9,
        "I-Position": 5,
        "I-Evidence": 14,
        "I-Claim": 3,
        "I-Concluding Statement": 11,
        "I-Counterclaim": 6,
        "I-Rebuttal": 4,
    }

    PROB_THRESH = {
        "I-Lead": 0.7,
        "I-Position": 0.55,
        "I-Evidence": 0.65,
        "I-Claim": 0.55,
        "I-Concluding Statement": 0.7,
        "I-Counterclaim": 0.5,
        "I-Rebuttal": 0.55,
    }


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def agg_essays(folder):
    names, texts = [], []
    for f in tqdm(list(os.listdir(folder))):
        names.append(f.replace('.txt', ''))
        texts.append(open(folder + '/' + f, 'r', encoding='utf-8').read())
    df_texts = pd.DataFrame({'id': names, 'text': texts})

    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts


def ner(df_texts, df_train):
    all_entities = []
    for _, row in tqdm(df_texts.iterrows(), total=len(df_texts)):
        total = len(row.text_split)
        entities = ['0'] * total

        for _, row2 in df_train[df_train['id'] == row['id']].iterrows():
            discourse = row2['discourse_type']
            list_ix = [int(x) for x in row2['predictionstring'].split(' ')]
            entities[list_ix[0]] = f'B-{discourse}'
            for k in list_ix[1:]:
                try:
                    entities[k] = f'I-{discourse}'
                except IndexError:
                    print(row['id'])
                    print(row2['discourse_text'])
                    print('predictionstring index:', k)
                    print('max length of text:', total)
        all_entities.append(entities)

    df_texts['entities'] = all_entities
    print('Completed mapping discourse to each token.')
    return df_texts


def preprocess(folder, withNER=False):
    df_texts = agg_essays(folder)
    df_gold = ""
    if withNER:
        if 'Train' in folder:
            df_gold = pd.read_csv(folder + '/train.csv')
        elif 'Validation' in folder:
            df_gold = pd.read_csv(folder + '/validate.csv')
        elif 'Test' in folder:
            df_gold = pd.read_csv(folder + '/test.csv')
        else:
            print("No gold standard found for:" + folder + ".\nText DF without IOB-Tags will be returned.")
            return df_texts, None
    df_texts = ner(df_texts, df_gold)
    return df_texts, df_gold


def build_model_tokenizer(model, model_path=None):
    # Tokenizer
    if 'longformer' in model:
        tokenizer = LongformerTokenizerFast.from_pretrained(model, add_prefix_space=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)

    # Modell
    if 'longformer' in model:
        model = LongformerForTokenClassification.from_pretrained(model, num_labels=15)
        # ADD PAD TOKEN TO MODEL
        model.resize_token_embeddings(len(tokenizer))
        model.longformer.embeddings.word_embeddings.padding_idx = 1
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
    else:
        model = AutoModel.from_pretrained(model, num_labels=15)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))

    return model, tokenizer


class FeedbackPrizeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.text_split[index]
        word_labels = self.data.entities[index]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  padding='max_length',
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [Constants.LABELS_TO_IDS[label] for label in word_labels]

        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                try:
                    encoded_labels[idx] = labels[i]
                    i += 1
                except IndexError:
                    # print("Length of labels:" + str(len(labels)))
                    # print("Length of encoded_labels:" + str(len(encoded_labels)))
                    # print("IndexError with idx [" + str(idx) + "] and label number [" + str(i) + "]")
                    continue

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len


def load_data(dataframe, batch_size):
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }
    return DataLoader(dataframe, **train_params)


def model_train(training_loader, model, optimizer, device, max_norm):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.to(device)
    model.train()

    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        tr_logits = outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_norm
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def get_sentence_predictions(device, model, max_len, tokenizer, sentence):
    inputs = tokenizer(sentence,
                       is_split_into_words=True,
                       return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True,
                       max_length=max_len,
                       return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits,
                                         axis=1)  # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [Constants.OUTPUT_LABELS[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
        else:
            continue
    return prediction


def get_final_prediction(test_text_df, y_pred):
    final_preds = []
    for i in tqdm(range(len(test_text_df))):
        idx = test_text_df.id.values[i]
        pred = [x.replace('B-', '').replace('I-', '') for x in y_pred[i]]
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == '0':
                j += 1
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1

            if cls != '0' and cls != '' and end - j > 10:
                final_preds.append((idx, cls, ' '.join(map(str, list(range(j, end))))))

            j = end
    return final_preds


def model_predict(device, model, max_len, tokenizer, dataframe):
    pred = []
    for i, t in enumerate(dataframe['text_split'].tolist()):
        o = get_sentence_predictions(device, model, max_len, tokenizer, t)
        pred.append(o)
    pred = pd.DataFrame(get_final_prediction(dataframe, pred))
    return pred


def calc_overlap2(set_pred, set_gt):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)
        len_pred = len(set_pred)
        inter = len(set_gt & set_pred)
        overlap_1 = inter / len_gt
        overlap_2 = inter / len_pred
        return (overlap_1, overlap_2)
    except:  # at least one of the input is NaN
        return (0, 0)


def score_feedback_comp_micro(pred_df, gt_df, discourse_type):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    scores = {}
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type,['id', 'predictionstring']].reset_index(drop=True)
    pred_df = pred_df.loc[pred_df['class'] == discourse_type,['id', 'predictionstring']].reset_index(drop=True)
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [set(pred.split(' ')) for pred in pred_df['predictionstring']]
    gt_df['predictionstring'] = [set(pred.split(' ')) for pred in gt_df['predictionstring']]

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    overlaps = [calc_overlap2(*args) for args in zip(joined.predictionstring_pred, joined.predictionstring_gt)]

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['potential_TP'] = [(overlap[0] >= 0.5 and overlap[1] >= 0.5) for overlap in overlaps]
    joined['max_overlap'] = [max(*overlap) for overlap in overlaps]
    joined_tp = joined.query('potential_TP').reset_index(drop=True)
    tp_pred_ids = joined_tp.sort_values('max_overlap', ascending=False).groupby(['id', 'gt_id'])['pred_id'].first()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = set(joined['pred_id'].unique()) - set(tp_pred_ids)

    matched_gt_ids = joined_tp['gt_id'].unique()
    unmatched_gt_ids = set(joined['gt_id'].unique()) - set(matched_gt_ids)

    # Get numbers of each type
    TP = len(tp_pred_ids)
    scores['TP'] = TP
    FP = len(fp_pred_ids)
    scores['FP'] = FP
    FN = len(unmatched_gt_ids)
    scores['FN'] = FN

    if (TP+FN) != 0 and (TP+FP) != 0:
        scores['Precision'] = TP / (TP+FN)
        scores['Recall'] = TP / (TP+FP)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    scores['F1'] = my_f1_score
    return scores


def score_feedback_comp(gt_df, pred_df):
    scores = {}
    for discourse_type in gt_df.discourse_type.unique():
        class_scores = score_feedback_comp_micro(pred_df, gt_df, discourse_type)
        scores[discourse_type] = class_scores

    overall_scores = {'TP':0,'FP':0,'FN':0}
    f1_array = []
    for label in scores.keys():
        overall_scores['TP'] = overall_scores.get('TP') + scores.get(label).get('TP')
        overall_scores['FP'] = overall_scores.get('FP') + scores.get(label).get('FP')
        overall_scores['FN'] = overall_scores.get('FN') + scores.get(label).get('FN')
        f1_array.append(scores.get(label).get('F1'))
    overall_scores['Precision'] = overall_scores.get('TP') / (overall_scores.get('TP')+overall_scores.get('FN'))
    overall_scores['Recall'] = overall_scores.get('TP') / (overall_scores.get('TP') + overall_scores.get('FP'))
    overall_scores['F1'] = np.mean(f1_array)
    scores['overall'] = overall_scores

    f1 = scores.get('overall').get('F1')
    return f1, scores


def model_evaluate(data_pred, data_gold):
    col_list_gold = ["id", "discourse_type", "predictionstring"]
    data_gold = data_gold[col_list_gold]

    col_list_pred = ["id", "class", "predictionstring"]
    data_pred.reset_index(drop=True)
    data_pred.columns = col_list_pred

    overall_f1, scores = score_feedback_comp(data_gold, data_pred)
    print("Overall evaluation:", overall_f1)

    return overall_f1, scores


def write_evaluation(scores, output_path):
    df = pd.concat({k: pd.DataFrame(v) for k, v in scores.items()}, axis=1).stack(0).T
    df.to_csv(output_path)


def write_prediction(pred, output_path):
    pred.to_csv(output_path, index=False)
