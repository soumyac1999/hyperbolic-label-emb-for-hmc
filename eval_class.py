import pickle
import torch
from tqdm import tqdm
from models import LabelEmbedModel, TextRCNN, TextCNN
from torch.utils.data import DataLoader
import torch.nn as nn


testset = pickle.load(open("rcv1/test.pkl", "rb"))

checkpoints = ['rcv1_exp/flat_textcnn/27', 'rcv1_exp/poin_0.1_textcnn/26']

for t in checkpoints:
    print(t)

    checkpoint = torch.load(t)

    test_data_loader = DataLoader(
        testset, batch_size=1024, shuffle=False, num_workers=16, pin_memory=True
    )

    label_model = LabelEmbedModel(testset.n_labels, emb_dim=300, dropout_p=0.1)
    doc_model = TextCNN(
                testset.text_dataset.vocab,
                emb_dim=300,
                dropout_p=0.1,
                word_embed_dim=300,
            )

    label_model = nn.DataParallel(label_model)
    doc_model = nn.DataParallel(doc_model)

    label_model.load_state_dict(checkpoint['label_model'])
    doc_model.load_state_dict(checkpoint['doc_model'])

    label_model = label_model.cuda()
    doc_model = doc_model.cuda()

    label_model = label_model.eval()
    doc_model = doc_model.eval()

    Y = torch.arange(testset.n_labels).cuda()

    tp, fp, fn = 0, 0, 0
    tp_3 = fp_3 = fn_3 = tp_5 = fp_5 = fn_5 = 0
    tp_r = fp_r = fn_r = 0
    num_batch = test_data_loader.__len__()
    total_loss = 0.
    pbar = tqdm(total=num_batch)
    with torch.no_grad():
        for data in test_data_loader:
            pbar.update(1)

            docs, labels, _ = data
            docs, labels = docs.cuda(), labels.cuda()

            doc_emb = doc_model(docs)
            dot = doc_emb @ label_model(Y).T
            t = torch.sigmoid(dot)

            y_pred = 1.0 * (t > 0.5)
            y_true = labels

            tp += (y_true * y_pred).sum(dim=0)
            fp += ((1 - y_true) * y_pred).sum(dim=0)
            fn += (y_true * (1 - y_pred)).sum(dim=0)

            h_r = torch.zeros(dot.shape).cuda()
            h_3 = torch.zeros(dot.shape).cuda()
            h_5 = torch.zeros(dot.shape).cuda()
            tk_3 = torch.topk(dot, k=3, dim=1)[1]
            tk_5 = torch.topk(dot, k=5, dim=1)[1]

            for i in range(dot.shape[0]):
                h_3[i][tk_3[i]] = 1.0
                h_5[i][tk_5[i]] = 1.0
                h_r[i][torch.topk(dot[i], k=int(y_true[i].sum().cpu()))[1]] = 1.0

            tp_3 += (y_true * h_3).sum()
            fp_3 += ((1 - y_true) * h_3).sum()
            fn_3 += (y_true * (1 - h_3)).sum()

            tp_5 += (y_true * h_5).sum()
            fp_5 += ((1 - y_true) * h_5).sum()
            fn_5 += (y_true * (1 - h_5)).sum()

            tp_r += (y_true * h_r).sum()
            fp_r += ((1 - y_true) * h_r).sum()
            fn_r += (y_true * (1 - h_r)).sum()


        eps = 1e-7
        p_r = tp_r/(tp_r+fp_r+eps)
        p_3 = tp_3/(tp_3+fp_3+eps)
        p_5 = tp_5/(tp_5+fp_5+eps)
        r_r = tp_r/(tp_r+fn_r+eps)
        r_3 = tp_3/(tp_3+fn_3+eps)
        r_5 = tp_5/(tp_5+fn_5+eps)
        p = tp.sum()/(tp.sum() + fp.sum() + eps)
        r = tp.sum()/(tp.sum() + fn.sum() + eps)
        micro_f = 2*p*r/(p+r+eps)
        macro_p = tp/(tp+fp+eps)
        macro_r = tp/(tp+fn+eps)
        macro_f = (2*macro_p*macro_r/(macro_p + macro_r + eps)).mean()
    pbar.close()

    print(
        f'P: {p.item():.6f}, R: {r.item():.6f}, micro-F: {micro_f.item():.6f}, '
        f'macro-F: {macro_f.item():.6f}, P@3: {p_3.item():.6f}, P@5: {p_5.item():.6f}, '
        f'R@3: {r_3.item():.6f}, R@5: {r_5.item():.6f}, R-Prec: {p_r.item():.6f}'
        )