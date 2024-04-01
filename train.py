from tqdm import tqdm

import torch
from torchvision import transforms, datasets

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
