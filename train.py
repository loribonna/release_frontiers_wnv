import logging
import time

import numpy as np
import torch
from tqdm import tqdm


def train(model, train_loader, loss_fn, optimizer, device,
          params, metrics, roc_score, args, split):
    start_time = time.time()
    model.train()

    train_loss = 0.0

    array_pred = np.array([])
    array_labels = np.array([])
    array_softmax = np.array([])

    # Iterate over data.
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data) in enumerate(train_loader):
            if args.debug_mode and batch_idx > 2:
                break

            inputs = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int64)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            # Forward
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
            val_softmax = softmax(outputs).detach()  # I take the prob of the positive class

            loss = loss_fn(outputs, labels)
            assert not torch.isnan(loss).any()

            loss.backward()

            # statistics
            train_loss += loss.item()

            # write loss avg on log file
            t.set_postfix(loss='{:05.3f}'.format(loss.item()))
            t.update()

            # update the params of the model
            optimizer.step()

            # concat array in order to calculate the overall metrics for the epoch
            if batch_idx == 0:
                array_labels = labels.cpu().numpy()
                array_pred = preds.cpu().numpy()
                array_softmax = val_softmax.cpu().numpy()
            else:
                array_labels = np.concatenate((array_labels, labels.cpu().numpy()), axis=0)
                array_pred = np.concatenate((array_pred, preds.cpu().numpy()), axis=0)
                array_softmax = np.concatenate((array_softmax, val_softmax.cpu().numpy()), axis=0)

        # PRINT THE METRICS AT THE END OF EACH EPOCH
        metrics_calc = {metric: metrics[metric](array_labels, array_pred, 'binary') for metric in metrics}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_calc.items())
        logging.info("- Train metrics: " + metrics_string)
        logging.info("- Train Loss : {:.4f}".format(train_loss / len(train_loader)))
        roc_calc = {rs: roc_score[rs](array_labels, array_softmax[:, 1]) for rs in roc_score}
        roc_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in roc_calc.items())
        logging.info("- Train roc: " + roc_string)

    time_elapsed = time.time() - start_time
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return
