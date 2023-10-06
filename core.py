import torch
import numpy as np
from utils.utils import calc_auroc
import warnings
warnings.filterwarnings('ignore')


def train(epoch, train_loader, model, optimizer, criterion_CE, criterion_l1, device, opt):
    train_loss = 0.
    train_accuracy = 0.
    run_accuracy = 0.

    run_loss = 0.
    total = 0.
    model.train()
    iters = 0
    for datas, label in (train_loader):
        iters += 1
        inputs = datas[0].to(device)
        mask = datas[1].float().to(device)
        label = label.long().to(device)
        #print(f'The shape of input and label are {inputs.shape} and {label.shape}')

        optimizer.zero_grad()
        if opt.loc:
            outs, masks = model(inputs)
            loss_ce = criterion_CE(outs, label)
            loss_bce = criterion_l1(masks, mask)
            loss = opt.lambda_loc * loss_bce + opt.lambda_cls * loss_ce
        else:
            outs = model(inputs)
            loss = criterion_CE(outs, label)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        total += label.size(0)
        run_loss += loss.item()

        _, prediction = torch.max(outs, 1)
        run_accuracy += (prediction == label).sum().item()
        if iters % opt.freq == 0:
            print('epoch {} | iter {} | train accuracy: {:.4f}% | loss:  {:.4f}'.format(epoch, iters,
                                                                                        100 * run_accuracy / (
                                                                                                label.size(
                                                                                                    0) * opt.freq),
                                                                                        run_loss / opt.freq))
            train_accuracy += run_accuracy
            train_loss += run_loss
            run_accuracy, run_loss = 0., 0.


def test(epoch, valid_loader, model, criterion_CE, criterion_l1, device, opt):
    model.eval()
    with torch.no_grad():
        accuracy = 0.
        total = 0.
        val_loss = 0.
        for datas, label in valid_loader:
            inputs = datas[0].to(device)
            mask = datas[1].float().to(device)
            label = label.long().to(device)

            if opt.loc:
                outs, masks = model(inputs)
                loss_ce = criterion_CE(outs, label)
                loss_bce = criterion_l1(masks, mask)
                loss = opt.lambda_loc * loss_bce + opt.lambda_cls * loss_ce
            else:
                outs = model(inputs)
                loss = criterion_CE(outs, label)

            _, prediction = torch.max(outs, 1)
            total += label.size(0)
            accuracy += (prediction == label).sum().item()
            val_loss += loss.item()

        print('epoch {} | The Val acc is {:.4f}% | total loss is {:.4f}'.format(epoch,
                                                                                np.round(100. * accuracy / total, 4),
                                                                                val_loss / len(valid_loader)))

        return accuracy, val_loss / len(valid_loader)


def test_osr(known_loader, unknown_loader, model, device, opt):
    model.eval()
    _prediction_k = []
    _prediction_u = []
    with torch.no_grad():
        accuracy = 0.
        total = 0.
        for datas, label in known_loader:
            inputs = datas[0].to(device)
            mask = datas[1].float().to(device)
            label = label.long().to(device)

            if opt.loc:
                outs, masks = model(inputs)
            else:
                outs = model(inputs)

            _, prediction = torch.max(outs, 1)
            total += label.size(0)
            accuracy += (prediction == label).sum().item()
            _prediction_k.append(outs.data.cpu().numpy())

        print(' The Test acc is {:.4f}% '.format(np.round(100. * accuracy / total, 4)))

        for datas, label in unknown_loader:
            inputs = datas[0].to(device)

            if opt.loc:
                outs, masks = model(inputs)
            else:
                outs = model(inputs)

            _, prediction = torch.max(outs, 1)
            _prediction_u.append(outs.data.cpu().numpy())


        _prediction_k = np.concatenate(_prediction_k, 0)
        _prediction_u = np.concatenate(_prediction_u, 0)

        x1, x2 = np.max(np.array(_prediction_k), axis=1), np.max(np.array(_prediction_u), axis=1)
        auc_prob, _, _ = calc_auroc(x1, x2)
        print(' The open set AUC is {:.4f}%'.format(auc_prob * 100))
