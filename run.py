import argparse
import torch
import torch.nn as nn
import argparse
import tqdm
import time
from util import load_data_n_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def train(model, tensor_loader, num_epochs, learning_rate, criterion, device, data_model):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for i, data in enumerate(tensor_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = criterion(outputs, labels)
            if i % 10 == 0:
                print("LOSS:", loss.item())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)

        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        if epoch % 50 == 0:
            torch.save(model, "./weights/" + data_model + "/" + str(epoch + 1) + "_" + time.time() + "_" + str(
                epoch_accuracy) + "_" + str(epoch_loss) + "---.pth")
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
    return


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
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

        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)


def main():
    root = './data/'
    parser = argparse.ArgumentParser('WiFi Imaging Benchmark')
    parser.add_argument('--dataset', choices=['ARIL', 'SignFi'], default='ARIL')
    parser.add_argument('--model', choices=['ResNet18'], default='ResNet18')
    args = parser.parse_args()

    print(root)

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
        device=device
    )
    return


if __name__ == "__main__":
    main()
