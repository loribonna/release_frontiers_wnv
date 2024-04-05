import os
# SET CUDA REPRODUCIBLE
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import math
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
import torch
import pandas as pd
from tqdm import tqdm


def test(model, test_loader, loss_fn, device, params, metrics, roc_score, split, args, log_dir="data/experiments", return_outs=False):

    # set the model to evaluation mode
    model.eval()
    test_loss = 0

    array_pred = np.array([])
    array_labels = np.array([])
    array_softmax = np.array([])
    all_outputs = []
    all_input_date_and_cod = []

    with torch.no_grad():
        for batch_idx, (data) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            if args.debug_mode_test and batch_idx > 1:
                break
            inputs = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int64)
            companies_cod = data[-2]
            img_dates = data[-1]

            # compute the model output
            extra = None
            if args.mode != 'temporal':
                extra = data[-3].to(device)

            if args.mode != 'graph':
                outputs = model(inputs, extra=extra)
            else:
                blocks_adj_matrix = data[2].to(device, dtype=torch.float)
                outputs = model(inputs, blocks_adj_matrix=blocks_adj_matrix, neighbours_numbers=6, extra=extra)

            _, preds = torch.max(outputs, 1)
            softmax = torch.nn.Softmax(dim=1)
            val_softmax = softmax(outputs).detach()
            all_outputs.append(outputs.detach().cpu().numpy())

            if return_outs:
                all_input_date_and_cod.append((img_dates, companies_cod, labels.cpu().numpy()))

            loss = loss_fn(outputs, labels)

            test_loss += loss.item()

            if math.isnan(loss.item()):
                return None

            if batch_idx == 0:
                array_labels = labels.cpu().numpy()
                array_pred = preds.cpu().numpy()
                array_softmax = val_softmax.cpu().numpy()
                lists_to_save = [companies_cod, img_dates, list(labels.cpu().numpy()), list(preds.cpu().numpy()),
                                 list(val_softmax[:, 0].cpu().numpy()), list(val_softmax[:, 1].cpu().numpy())]
                df_to_save = pd.DataFrame(lists_to_save).transpose()
                df_to_save.columns = ['company_cod', 'dates', 'labels', 'pred_thr_0.5', 'softmax_cls0', 'softmax_cls1']
            else:
                array_labels = np.concatenate((array_labels, labels.cpu().numpy()), axis=0)
                array_pred = np.concatenate((array_pred, preds.cpu().numpy()), axis=0)
                array_softmax = np.concatenate((array_softmax, val_softmax.cpu().numpy()), axis=0)

                # save excel files with pred and labels for each company and each date, only for the binary case
                lists_to_save = [companies_cod, img_dates, list(labels.cpu().numpy()), list(preds.cpu().numpy()),
                                 list(val_softmax[:, 0].cpu().numpy()), list(val_softmax[:, 1].cpu().numpy())]
                df_to_save_append = pd.DataFrame(lists_to_save).transpose()
                df_to_save_append.columns = ['company_cod', 'dates', 'labels', 'pred_thr_0.5', 'softmax_cls0', 'softmax_cls1']
                df_to_save = pd.concat((df_to_save, df_to_save_append), axis=0)

    test_loss /= len(test_loader)

    metrics_calc = {metric: metrics[metric](array_labels, array_pred, 'binary') for metric in metrics}
    evals = confusion_matrix(array_labels, array_pred)
    if len(evals) < 4:
        print("\nCannot compute confusion matrix because of too few samples")
        print("Please request the full dataset")
        print("\nCurrent accuracy (%):", (array_labels == array_pred).mean()*100)
        return None
    tn, fp, fn, tp = evals.ravel()
    sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
    metrics_calc['sensitivity'] = sensitivity
    metrics_calc['specificity'] = specificity
    metrics_calc['tp'] = tp
    metrics_calc['tn'] = tn
    metrics_calc['fp'] = fp
    metrics_calc['fn'] = fn

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_calc.items())
    print("- Test metrics : " + metrics_string)
    print("- Test Loss : {:.4f}".format(test_loss))

    try:
        roc_calc = {rs: roc_score[rs](array_labels, array_softmax[:, 1]) for rs in roc_score}
        roc_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in roc_calc.items())
        print("- Test: " + roc_string)

        precision, recall, _ = precision_recall_curve(array_labels, array_softmax.argmax(1))
        metrics_calc.update({"AUPRC": auc(recall, precision)})
    except BaseException:
        metrics_calc.update({"AUPRC": -1})

    print("\n")
    if return_outs:
        return metrics_calc, np.concatenate(all_outputs), np.hstack(all_input_date_and_cod)
    else:
        return metrics_calc
