import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import pandas as  pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


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

## (is not efficient) copied from https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    # mean = torch.zeros(3)
    # std = torch.zeros(3)
    # print('==> Computing mean and std..')
    # for inputs, targets in dataloader:
    #     for i in range(3):
    #         mean[i] += inputs[:,i,:,:].mean()
    #         std[i] += inputs[:,i,:,:].std()
    # mean.div_(len(dataset))
    # std.div_(len(dataset))
    # train data mean
    mean = (dataset.data.mean(axis=(0,1,2))/dataset.data.max())
    # train data standard deviation
    std = (dataset.data.std(axis=(0,1,2))/dataset.data.max())
    return mean, std

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
                  mean = [0.4914,0.4822,0.4465],
                  std = [0.247,0.243,0.262],
                  need_grad_cam_image = False,
                  target_layers = None,
                  plot_TP = True):
    loss_list = get_individual_loss(model, device, data_loader, criterion)
    loss_df = pd.DataFrame(loss_list, columns=['transform_image', 'target', 'prediction', 'loss'])

    if label_names is not None:
        loss_df['target_name'] = loss_df['target'].apply(lambda x: label_names[x])
        loss_df['prediction_name'] = loss_df['prediction'].apply(lambda x: label_names[x])

    ## plot confusion matrix
    plot_confusion_matrix(actual = loss_df['target_name'].to_list(),
                          predicted =  loss_df['prediction_name'].to_list())
    
    loss_df['image'] = loss_df['transform_image'].apply(lambda img: inverse_normalize(img,
                                                                                mean, std).permute(1, 2, 0).numpy())

    # correct
    if plot_TP:
        correct_df = loss_df[loss_df.prediction == loss_df.target]
        print(f"total correct predictions: {correct_df.shape[0]}")
        correct_df = correct_df.sort_values(by='loss', ascending=True)
        if need_grad_cam_image: plot_grad_cam_image(model = model,
                                                target_layers = target_layers,
                                                input_tensors = correct_df['transform_image'].to_list(),
                                                images = correct_df['image'].to_list(),
                                                cam_targets = correct_df['prediction'].to_list(),
                                                target_labels= correct_df['target_name'].to_list(),
                                                pred_labels= correct_df['prediction_name'].to_list(),
                                                losses = correct_df['loss'].to_list(),
                                                rows = img_rows, cols = img_cols)
        else: plot_image(images = correct_df['image'].to_list(),
                target_labels= correct_df['target_name'].to_list(),
                pred_labels= correct_df['prediction_name'].to_list(),
                losses = correct_df['loss'].to_list(),
                rows = img_rows, cols = img_cols)
    
    # incorrect - default behaviour
    incorrect_df = loss_df[loss_df.prediction != loss_df.target]
    print(f"total wrong predictions: {incorrect_df.shape[0]}")

    incr_groups = incorrect_df.groupby(['target_name','prediction_name']).agg({'loss':'median',
                                                             'image':'count'}).reset_index().sort_values(by='image', ascending=False)

    incorrect_df = incorrect_df.sort_values(by='loss', ascending=False)
    if need_grad_cam_image: plot_grad_cam_image(model = model,
                                                target_layers = target_layers,
                                                input_tensors = incorrect_df['transform_image'].to_list(),
                                                images = incorrect_df['image'].to_list(),
                                                cam_targets = incorrect_df['prediction'].to_list(),
                                                target_labels= incorrect_df['target_name'].to_list(),
                                                pred_labels= incorrect_df['prediction_name'].to_list(),
                                                losses = incorrect_df['loss'].to_list(),
                                                rows = img_rows, cols = img_cols)
    else: plot_image(images = incorrect_df['image'].to_list(),
               target_labels= incorrect_df['target_name'].to_list(),
               pred_labels= incorrect_df['prediction_name'].to_list(),
               losses = incorrect_df['loss'].to_list(),
               rows = img_rows, cols = img_cols)
    return incr_groups

def plot_image(images, target_labels, pred_labels = None, losses = None, rows = 5, cols = 5,
               img_size=(5,5), font_size = 7):
    figure = plt.figure(figsize=img_size)
    for index in range(cols * rows):
        plt.subplot(rows, cols, index+1)
        if pred_labels is not None and losses is not None:
            plt.title(f'target: {target_labels[index]}\nprediction: {pred_labels[index]}\nloss: {round(losses[index],2)}',
                  fontsize = font_size)
        else:
            plt.title(f'target: {target_labels[index]}', fontsize = font_size)
        plt.axis('off')
        plt.imshow(images[index])
    figure.tight_layout()
    plt.show()

def plot_grad_cam_image(model, target_layers, input_tensors, images, cam_targets, target_labels,
                        pred_labels = None, losses = None, rows = 5, cols = 5,
               img_size=(5,5), font_size = 7):
    ## create grad_cam
    cam_op_list = []
    cam_v_list = []
    for i in range(rows*cols):
        c, v = get_gradcam_image(model,
                  target_layers = target_layers,
                  input_tensor = input_tensors[i].unsqueeze(0),
                  rgb_img = images[i],
                  cam_target = cam_targets[i])
        cam_op_list.append(c)
        cam_v_list.append(v)

    figure = plt.figure(figsize=img_size)
    for index in range(cols * rows):
        plt.subplot(rows, cols, index+1)
        if pred_labels is not None and losses is not None:
            plt.title(f'target: {target_labels[index]}\nprediction: {pred_labels[index]}\nloss: {round(losses[index],2)}',
                  fontsize = font_size)
        else:
            plt.title(f'target: {target_labels[index]}', fontsize = font_size)
        plt.axis('off')
        plt.imshow(cam_v_list[index])
    figure.tight_layout()
    plt.show()

def inverse_normalize(tensor, mean, std):
    inv_normalize = transforms.Normalize(
                    mean= [-m/s for m, s in zip(mean, std)],
                std= [1/s for s in std]
                )
    return inv_normalize(tensor)

def get_gradcam_image(model, target_layers, input_tensor, rgb_img, cam_target, cam_batch_size = 64):
    cam = GradCAM(model=model, target_layers=target_layers)
    cam.batch_size = cam_batch_size
    targets = [ClassifierOutputTarget(cam_target)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # You can also get the model outputs without having to re-inference
    return cam.outputs, visualization

def plot_confusion_matrix(actual, predicted ):
    cm = confusion_matrix(actual,predicted)
    sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=sorted(list(set(actual))), 
            yticklabels=sorted(list(set(actual)))
            )
    plt.yticks(rotation=0)
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()
