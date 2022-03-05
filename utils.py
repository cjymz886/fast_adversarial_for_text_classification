from sklearn import metrics
import torch
import numpy as np
from loader import seq_padding



def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)




def metric(cfg, model, eval_data, word_to_id, cat_to_id,mode='train'):
    correct_num = 0
    total_num = 0

    y_test_cls=[]
    y_pred_cls=[]

    for line in eval_data:
        content = line.tokens
        label = line.label
        input_ids = [word_to_id[x] if x in word_to_id else 0 for x in content]
        input_ids=torch.LongTensor([seq_padding(input_ids,cfg.seq_length)]).to(cfg.device)
        with torch.no_grad():
            logits = model(input_ids,None,is_training=False)
            logits = logits.cpu().detach().numpy()
        pred_label = np.argmax(logits, axis=1)[0]

        y_test_cls.append(cat_to_id[label])
        y_pred_cls.append(pred_label)

        if pred_label == cat_to_id[label]:
            correct_num += 1
        total_num += 1

    acc = correct_num / total_num

    if mode=='test':
        # evaluate
        y_test_cls = np.array(y_test_cls)
        y_pred_cls = np.array(y_pred_cls)
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=cat_to_id,digits=4))

        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)

    return acc


