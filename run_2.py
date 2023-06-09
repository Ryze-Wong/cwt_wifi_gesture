import argparse

import numpy as np
import torch
import torch.nn as nn
import argparse
import random
import os
from tqdm import tqdm
import time
from util import load_data_n_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    cohen_kappa_score, matthews_corrcoef


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(model, train_tensor_loader, test_tensor_loader, num_epochs, learning_rate, criterion, device, data_model):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    accuracy_list = []
    cur_time = time.time()
    cur_time = str(cur_time).split('.')[0]
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        test_accuracy = 0
        for (inputs, labels) in tqdm(train_tensor_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            # print("outputs shape:")
            # print(outputs.shape)
            # print("labels shape")
            # print(labels.shape)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)

        epoch_loss = epoch_loss / len(train_tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(train_tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
        model.eval()
        with torch.no_grad():
            labels_list = []
            pre_list = []
            for data in test_tensor_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels.to(device)
                labels = labels.type(torch.LongTensor)

                outputs = model(inputs)
                outputs = outputs.type(torch.FloatTensor)
                outputs.to(device)

                # loss = criterion(outputs, labels)
                predict_y = torch.argmax(outputs, dim=1).to(device)
                test_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
            accuracy = test_accuracy / len(test_tensor_loader)
            print('accuracy:', accuracy)
            accuracy_list.append(accuracy)
            print('save beat weight....', np.max(accuracy_list))
            if accuracy >= np.max(accuracy_list):
                torch.save(model, "./weights/" + data_model + "/" + cur_time + "_best_weight.pth")
        # if epoch % 50 == 0:
        #     torch.save(model, "./weights/" + data_model + "/" + str(epoch + 1) + "_" + str(time.time()) + "_" + str(
        #         epoch_accuracy) + "_" + str(epoch_loss) + "---.pth")
        # print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
    return


def test(model, tensor_loader, criterion, device, data_model):
    model.eval()
    test_acc = 0
    test_loss = 0
    total_samples = 0
    weighted_precision_sum = 0
    weighted_recall_sum = 0
    weighted_f1_sum = 0
    weighted_ck_sum = 0
    weighted_mcc_sum = 0

    classes = 6
    sum_cm = np.zeros((classes, classes))

    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)

        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)

        # 将PyTorch Tensor转换为NumPy数组
        y_true = labels.cpu().numpy()
        y_pred = predict_y.cpu().numpy()

        num_samples = len(y_true)
        total_samples += num_samples

        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 计算准确率
        acc = accuracy_score(y_true, y_pred)

        # 计算精确率
        precision = precision_score(y_true, y_pred, average='macro')

        # 计算召回率
        recall = recall_score(y_true, y_pred, average='macro')

        # 计算F1分数
        f1 = f1_score(y_true, y_pred, average='macro')

        # 计算cohen_kappa_score
        cohen_kappa = cohen_kappa_score(y_true, y_pred)

        # 计算matthews_corrcoef
        mcc = matthews_corrcoef(y_true, y_pred)

        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)

        # print("Confusion Matrix:")
        # print(cm)
        # print("Accuracy:", acc)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1 Score:", f1)
        # print("cohen_kappa_score: ", cohen_kappa)
        # print("MCC: ", mcc)

        if "SignFi" not in data_model:
            sum_cm += cm
        weighted_precision_sum += num_samples * precision
        weighted_recall_sum += num_samples * recall
        weighted_f1_sum += num_samples * f1
        weighted_ck_sum += num_samples * cohen_kappa
        weighted_mcc_sum += num_samples * mcc

    print("--------------------below are {} average evaluations".format(data_model))
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print("confusion matrix: ")
    if "SignFi" not in data_model:
        print(sum_cm)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    print("Precision: ", weighted_precision_sum / total_samples)
    print("Recall: ", weighted_recall_sum / total_samples)
    print("F1 Score: ", weighted_f1_sum / total_samples)
    print("cohen_kappa_score: ", weighted_ck_sum / total_samples)
    print("MCC: ", weighted_mcc_sum / total_samples)


def main():
    seed_torch()
    root = './data/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['ARIL', 'SignFi'], default='ARIL')
    parser.add_argument('--model',
                        choices=['ResNet18', 'ResNet18_CBAM', 'ResNet50', 'ResNet18_CBAM2', 'ResNet18_CBAM2_mish', 'ARIL_ResNet18_CBAM2_parallel'],
                        default='ResNet18')
    parser.add_argument("--test", default=False, action='store_true', help='If added, the epoch will be 2')

    args = parser.parse_args()

    print("data root: ", root)
    print("test_action: ", args.test)

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root, args.test)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("train epochs: {}".format(train_epoch))

    data_model = args.dataset + '_' + args.model

    train(
        model=model,
        train_tensor_loader=train_loader,
        test_tensor_loader=test_loader,
        num_epochs=train_epoch,
        learning_rate=1e-3,
        criterion=criterion,
        device=device,
        data_model=data_model
    )
    test(
        model=model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device,
        data_model=data_model
    )
    return


if __name__ == "__main__":
    main()
