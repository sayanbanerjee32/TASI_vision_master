from tqdm import tqdm
import numpy as np

from torch_lr_finder import LRFinder

import torch
from torchvision import transforms, datasets

def lr_range_test(model, optimizer, criterion, device, end_lr=10, num_iter=100):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state

## Return prediction count based on model prediction and target
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

### train function that will be called for each epoch
def train(model, device, train_loader, optimizer, criterion):
  # this only tells the model that training will start so that training related configuration can be swithed on
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    # move all data to same device
    data, target = data.to(device), target.to(device)
    # No gradient accumulation
    optimizer.zero_grad()

    # Predict - this calls the forward function
    pred = model(data)

    # Calculate loss - 
    # difference between molde prediction and target values based on criterion function provided as input
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    # takes a step, i.e. updates parameter values based on gradients
    # these parameter values will be used for the next batch
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    # Shows traing progress with loss and accuracy update for each batch
    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

#   train_acc.append(100*correct/processed)
#   train_losses.append(train_loss/len(train_loader))
  # returns training accuracy and loss for the epoch  
  return (100*correct/processed), (train_loss/len(train_loader))

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

### test function that will be called for each epoch
def test(model, device, test_loader, criterion):
    # switching on eval / test mode
    model.eval()

    test_loss = 0
    correct = 0
    # no gradient calculation is required for test
    ## as parameters are not updated while test
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            # Calculate test loss - 
            # difference between molde prediction and target values based on criterion function provided as input
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    # test_acc.append(100. * correct / len(test_loader.dataset))
    # test_losses.append(test_loss)

    # print test loss and accuracy after each epoch
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # returns test accuracy and loss for the epoch
    return (100. * correct / len(test_loader.dataset)), test_loss

### runs train and test loop for each epoch
def train_orchestrator(model, device, train_loader, test_loader,
                       criterion, optimizer, scheduler,
                       learning_rate, num_epochs = 2, 
                       early_stopping = False, early_stopping_patience = 10):


    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    best_test_loss = np.inf
    best_epoch = -1

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        # call train function from utils.py
        trn_acc, trn_loss = train(model, device, train_loader, optimizer, criterion)
        # accumulate train accuracies and test losses for visualisation
        train_acc.append(trn_acc)
        train_losses.append(trn_loss)

        # call test function from utils.py
        tst_acc, tst_loss = test(model, device, test_loader, criterion)
        # accumulate test accuracies and test losses for visualisation
        test_acc.append(tst_acc)
        test_losses.append(tst_loss)

        scheduler.step(tst_loss)
        
        if learning_rate is not None and learning_rate != scheduler.get_last_lr()[0]:
            learning_rate = scheduler.get_last_lr()[0]
            print(f'Learning rate updated to: {learning_rate}')
            best_epoch = epoch
            # break

        # early stopping
        if early_stopping:
            if tst_loss < best_test_loss:
                best_test_loss = tst_loss
                best_epoch = epoch
                checkpoint(model, "best_model.pth")
            elif epoch - best_epoch > early_stopping_patience:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop
        else: 
            best_epoch = epoch

    if early_stopping: resume(model, "best_model.pth")

    return train_losses, train_acc, test_losses, test_acc, best_epoch, model
