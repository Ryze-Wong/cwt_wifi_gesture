import argparse

import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import time
from util import load_data_n_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device, data_model):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("train epochs: {}", num_epochs)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for (inputs, labels) in tqdm(tensor_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)

        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        if epoch % 50 == 0:
            torch.save(model, "./weights/" + data_model + "/" + str(epoch + 1) + "_" + str(time.time()) + "_" + str(
                epoch_accuracy) + "_" + str(epoch_loss) + "---.pth")
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
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
    if "SignFi" in data_model:
        classes = 276
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
        print("Confusion Matrix:")
        print(cm)

        # 计算准确率
        acc = accuracy_score(y_true, y_pred)
        print("Accuracy:", acc)

        # 计算精确率
        precision = precision_score(y_true, y_pred, average='macro')
        print("Precision:", precision)

        # 计算召回率
        recall = recall_score(y_true, y_pred, average='macro')
        print("Recall:", recall)

        # 计算F1分数
        f1 = f1_score(y_true, y_pred, average='macro')
        print("F1 Score:", f1)

        # 计算cohen_kappa_score
        cohen_kappa = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score: ", cohen_kappa)

        # 计算matthews_corrcoef
        mcc = matthews_corrcoef(y_true, y_pred)
        print("MCC: ", mcc)

        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
        sum_cm += cm
        weighted_precision_sum += num_samples * precision
        weighted_recall_sum += num_samples * recall
        weighted_f1_sum += num_samples * f1
        weighted_ck_sum += num_samples * cohen_kappa
        weighted_mcc_sum += num_samples * mcc

    print("--------------------below are {} average evaluations", data_model)
    test_acc = test_acc/len(tensor_loader)
    test_loss = test_loss/len(tensor_loader.dataset)
    print("confusion matrix: ")
    print(sum_cm)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    print("Precision: ", weighted_precision_sum / total_samples)
    print("Recall: ", weighted_recall_sum / total_samples)
    print("F1 Score: ", weighted_f1_sum / total_samples)
    print("cohen_kappa_score: ", weighted_ck_sum / total_samples)
    print("MCC: ", weighted_mcc_sum / total_samples)


def main():
    root = './data/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['ARIL', 'SignFi'], default='ARIL')
    parser.add_argument('--model', choices=['ResNet18', 'ResNet18_CBAM'], default='ResNet18')
    args = parser.parse_args()

    print("data root: ", root)

    train_loader, test_loader, model, train_epoch = load_data_n_model(args.dataset, args.model, root)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_model = args.dataset + '_' + args.model

    train(
        model=model,
        tensor_loader=train_loader,
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
