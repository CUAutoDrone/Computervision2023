from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR, SAVE_MODEL_EPOCH, MODEL_DIR
from model import create_model
from dataloader import train_loader, valid_loader
import torch
import matplotlib.pyplot as plt
import time
import sys
import numpy as np


def format_time(time_in_seconds):
    if time_in_seconds < 60:
        return f"{time_in_seconds%60:.2f}"
    if time_in_seconds < 3600:
        return f"{int(time_in_seconds//60)}:{int(time_in_seconds%60):02}"
    return f"{int(time_in_seconds//3600)}:{int((time_in_seconds%3600)//60):02}:{int(time_in_seconds%60):02}"


def train(train_data_loader, model):
    global train_itr
    train_loss_list = []

    for i, data in enumerate(train_data_loader, 1):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)

        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        losses.backward()
        optimizer.step()

        sys.stdout.write(
            f"\r     Training {100.0*i/len(train_data_loader):.2f}% Batch Loss: {loss_value:.4f}     ")
    print()
    return train_loss_list


def validate(valid_data_loader, model):
    global val_itr
    val_loss_list = []

    for i, data in enumerate(valid_data_loader, 1):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)

        # update the loss value beside the progress bar for each iteration
        sys.stdout.write(
            f"\r     Validating {100.0*i/len(valid_data_loader):.2f}% Batch Loss: {loss_value:.4f}     ")
    print()
    return val_loss_list


if __name__ == '__main__':
    MODEL_NAME = 'model'

    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(
        params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    figure_3, mean_train_ax = plt.subplots()
    figure_4, mean_valid_ax = plt.subplots()

    itr_train_loss = []
    itr_valid_loss = []
    avg_train_loss = []
    avg_valid_loss = []

    best_train_loss = 100_000
    best_valid_loss = 100_000

    for EPOCH in range(NUM_EPOCHS):
        print(f"\nEPOCH {EPOCH+1} of {NUM_EPOCHS}")
        start = time.time()
        train_loss = train(train_loader, model)
        valid_loss = validate(valid_loader, model)
        mean_train = np.mean(train_loss)
        mean_valid = np.mean(valid_loss)

        itr_train_loss = np.append(itr_train_loss, train_loss)
        itr_valid_loss = np.append(itr_valid_loss, valid_loss)
        avg_train_loss.append(mean_train)
        avg_valid_loss.append(mean_valid)

        end = time.time()
        print(f"     Mean Training Loss: {mean_train:.3f}")
        print(f"     Mean Validation Loss: {mean_valid:.3f}")
        print(f"     Took {format_time(end - start)}")

        train_ax.plot(itr_train_loss, color='blue')
        train_ax.set_xlabel('iterations')
        train_ax.set_ylabel('train loss')
        valid_ax.plot(itr_valid_loss, color='red')
        valid_ax.set_xlabel('iterations')
        valid_ax.set_ylabel('validation loss')

        mean_train_ax.plot(avg_train_loss, color='blue')
        mean_train_ax.set_xlabel('epoch')
        mean_train_ax.set_ylabel('train loss')
        mean_valid_ax.plot(avg_valid_loss, color='red')
        mean_valid_ax.set_xlabel('epoch')
        mean_valid_ax.set_ylabel('validation loss')

        figure_1.savefig(f"{OUT_DIR}/train_loss.png")
        figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
        figure_3.savefig(f"{OUT_DIR}/mean_train_loss.png")
        figure_4.savefig(f"{OUT_DIR}/mean_valid_loss.png")

        if (EPOCH+1) % SAVE_MODEL_EPOCH == 0:  # save model after every n epochs
            torch.save(model.state_dict(), f"{MODEL_DIR}/model{EPOCH+1}.pth")

        if mean_valid < best_valid_loss:
            best_valid_loss = mean_valid
            torch.save(model.state_dict(), f"{MODEL_DIR}/model_best_valid.pth")

        if (EPOCH+1) == NUM_EPOCHS:
            torch.save(model.state_dict(), f"{MODEL_DIR}/model{EPOCH+1}.pth")
