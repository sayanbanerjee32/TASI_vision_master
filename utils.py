import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import pandas as  pd


SEED = 1
## function to check if GPU is available and return relevant device
def get_device():
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    return cuda, torch.device("cuda" if cuda else "cpu")    

# function to plot train and test accuracies and losses
def plot_accuracy_losses(train_losses, train_acc, test_losses, test_acc, num_epochs):
    plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    ax = plt.gca()
    ax.set_xlim([0, num_epochs + 1])
    plt.ylabel('Loss')
    plt.plot(range(1, num_epochs + 1),
             train_losses[:num_epochs],
              'r', label='Training Loss')
    plt.plot(range(1, num_epochs + 1),
             test_losses[:num_epochs],
             'b', label='Test Loss')
    ax.grid(linestyle='-.')
    plt.legend()
    plt.subplot(2,1,2)
    ax = plt.gca()
    ax.set_xlim([0, num_epochs+1])
    plt.ylabel('Accuracy')
    plt.plot(range(1, num_epochs + 1),
             train_acc[:num_epochs],
              'r', label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1),
             test_acc[:num_epochs],
              'b', label='Test Accuracy')
    ax.grid(linestyle='-.')
    plt.legend()
    plt.show()

  
# get predicted value based on argmax
def GetPrediction(pPrediction):
  return pPrediction.argmax(dim=1)


# Returns individual image with target, prediction and loss
# after batch inference
def get_individual_loss(model, device, data_loader, criterion):
    # switching on eval / test mode
    model.eval()

    loss_list = []


    # no gradient calculation is required for test
    ## as parameters are not updated while test
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            for d,t,p,l in zip(data, target,
                               GetPrediction(output),
                               criterion(output, target, reduction='none')):
                loss_list.append((d.to('cpu'),
                                  t.to('cpu').item(),
                                  p.to('cpu').item(),
                                  l.to('cpu').item()))


    return loss_list
# the following function will plot images with target and predicted values
# where prediction is wrong - this will be in order or decreasing loss
# group by target
def plot_top_loss(model, device, data_loader, criterion,
                  label_names = None, img_rows = 5, img_cols = 5,
                  de_normalize = True,
                  mean = [0.4914,0.4822,0.4465],
                  std = [0.247,0.243,0.262]):
    loss_list = get_individual_loss(model, device, data_loader, criterion)
    loss_df = pd.DataFrame(loss_list, columns=['image', 'target', 'prediction', 'loss'])

    if label_names is not None:
        loss_df['target'] = loss_df['target'].apply(lambda x: label_names[x])
        loss_df['prediction'] = loss_df['prediction'].apply(lambda x: label_names[x])

    if de_normalize:
        loss_df['image'] = loss_df['image'].apply(lambda img: inverse_normalize(img, mean, std))

    # incorrect
    incorrect_df = loss_df[loss_df.prediction != loss_df.target]
    print(f"total wrong predictions: {incorrect_df.shape[0]}")

    incr_groups = incorrect_df.groupby(['target','prediction']).agg({'loss':'median',
                                                             'image':'count'}).reset_index().sort_values(by='image', ascending=False)

    incorrect_df = incorrect_df.sort_values(by='loss', ascending=False)
    plot_image(images = incorrect_df['image'].to_list(),
               target_labels= incorrect_df['target'].to_list(),
               pred_labels= incorrect_df['prediction'].to_list(),
               losses = incorrect_df['loss'].to_list(),
               rows = img_rows, cols = img_cols)
    return incr_groups

def plot_image(images, target_labels, pred_labels = None, losses = None, rows = 5, cols = 5,
               img_size=(5,5), font_size = 7):
    figure = plt.figure(figsize=img_size)
    for index in range(1, cols * rows  + 1):
        plt.subplot(rows, cols, index)
        if pred_labels is not None and losses is not None:
            plt.title(f'target: {target_labels[index]}\nprediction: {pred_labels[index]}\nloss: {round(losses[index],2)}',
                  fontsize = font_size)
        else:
            plt.title(f'target: {target_labels[index]}', fontsize = font_size)
        plt.axis('off')
        plt.imshow(images[index].permute(1, 2, 0))
    figure.tight_layout()
    plt.show()

def inverse_normalize(tensor, mean, std):
    inv_normalize = transforms.Normalize(
                    mean= [-m/s for m, s in zip(mean, std)],
                std= [1/s for s in std]
                )
    return inv_normalize(tensor)
