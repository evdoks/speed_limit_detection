import os
import numpy as np
from glob import glob
from torch import optim, cuda
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
from timeit import default_timer as timer

# %% find labels that correspond to a speed limit sign

dataset_path = '../datasets/mtsd_speed_limits_sliced'
new_dataset_path = '../datasets/mtsd_speed_limits_sliced_torch'

image_sets = ['train', 'val']

class_folder = 'speed_limit'
no_class_folder = 'no_speed_limit'

# %% create folder structure require by PyTorch ImageFolder dataset loader

os.system(f"rm -rf {new_dataset_path}")
os.system(f"mkdir {new_dataset_path}")
for t in image_sets:
    new_path = f'{new_dataset_path}/{t}'
    os.system(f"mkdir {new_path}")
    os.system(f"mkdir {new_path}/{class_folder}")
    os.system(f"mkdir {new_path}/{no_class_folder}")

    images = glob(f"{dataset_path}/images/{t}/*.jpg")
    for i in images:
        if os.path.exists(i.replace('images', 'labels').replace('.jpg', '.txt')):
            os.system(f"cp {i} {new_path}/{class_folder}/{os.path.split(i)[1]}")
        else:
            os.system(f"cp {i} {new_path}/{no_class_folder}/{os.path.split(i)[1]}")

# %% Find minimum and maximum resolutions of an image in the dataset

image_size_path = "../datasets/mtsd_speed_limits/images"
min_width = np.Inf
min_hight = np.Inf
min_width_imname = ''
min_hight_imname = ''

max_width = 0
max_hight = 0
max_width_imname = ''
max_hight_imname = ''

for t in image_sets:
    imnames = glob(f'{image_size_path}/{t}/*.jpg')
    for imname in imnames:
        im = Image.open(imname)
        if min_width > im.size[0]:
            min_width = im.size[0]
            min_width_imname = imname
        elif max_width < im.size[0]:
            max_width = im.size[0]
            max_width_imname = imname
        if min_hight > im.size[1]:
            min_hight = im.size[1]
            min_hight_imname = imname
        elif max_hight < im.size[1]:
            max_hight = im.size[1]
            max_hight_imname = imname

print(f"Min image dimestions: {min_width}x{min_hight}")
print(f"Image with smallest width: {min_width_imname}")
print(f"Image with smallest hight: {min_hight_imname}")
print(f"Max image dimestions: {max_width}x{max_hight}")
print(f"Image with smallest width: {max_width_imname}")
print(f"Image with smallest hight: {max_hight_imname}")

# %% Initialize some training parameters

save_file_name = 'resnet18_speedlim.pt'
checkpoint_path = 'resnet18_speedlim.pth'

# Change to fit hardware
batch_size = 32

# Whether to train on a gpu
multi_gpu = False
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False
print(train_on_gpu, multi_gpu)

# %%  Image transformations

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image


# %% Visualize image transormation

ex_img = Image.open('../datasets/mtsd_speed_limits_sliced_torch/train/speed_limit/zysNxmnTNk5QJHkr0X1VgQ_2_3.jpg')
t = image_transforms['train']
plt.figure(figsize=(24, 24))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    _ = imshow_tensor(t(ex_img), ax=ax)
plt.tight_layout()

# %% Datasets from folders

traindir = "../datasets/mtsd_speed_limits_sliced_torch/train"
validdir = "../datasets/mtsd_speed_limits_sliced_torch/val"
data = {
    'train': datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
    #  'test':
    #  datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}
# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=1),
    'val': DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=1),
    #  'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True,num_workers=10)
}

# %% output labels
categories = []
for d in os.listdir(traindir):
    categories.append(d)

n_classes = len(categories)
print(f'There are {n_classes} different classes.')

# %% ResNet18 model with fully connected layer replaced by a convolutional layer to accomodate arbitrary input sizes
# See https://learnopencv.com/fully-convolutional-image-classification-on-arbitrary-sized-image/


class FullyConvolutionalResnet18(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):

        # Start with standard resnet18 defined here
        super().__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        self.avgpool = nn.AvgPool2d((7, 7))

        # Convert the original fc layer to a convolutional layer.
        self.last_conv = torch.nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)
        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    # Reimplementing forward pass.
    def _forward_impl(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # Notice, there is no forward pass
        # through the original fully connected layer.
        # Instead, we forward pass through the last conv layer
        x = self.last_conv(x)
        return x


model = FullyConvolutionalResnet18(pretrained=False, num_classes=n_classes).eval()

# %% Resnet18 model with fully connected layer updated to predict our classes

model = models.resnet18(pretrained=True)

# Freeze model weights to rely purely on transfer learning
#  for param in model.parameters():
#      param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, n_classes),
                         nn.LogSoftmax(dim=1))
#  model.fc = nn.Linear(num_ftrs, n_classes)

if train_on_gpu:
    model = model.to('cuda')

if multi_gpu:
    model = nn.DataParallel(model)

model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())


def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=1):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except Exception:
        model.epochs = 0
        print('Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} '
                'seconds elapsed in epoch.',
                end='\r')

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: '
                              '{valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.')
    # Format history
    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


# %% train the model

model, history = train(model,
                       criterion,
                       optimizer,
                       dataloaders['train'],
                       dataloaders['val'],
                       save_file_name=save_file_name,
                       max_epochs_stop=10,
                       n_epochs=5,
                       print_every=1)

# %% training and validation losses

plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Negative Log Likelihood')
plt.title('Training and Validation Losses')

# %% training and validation accuracy

plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(100 * history[c], label=c)
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Average Accuracy')
plt.title('Training and Validation Accuracy')

# %% Save Model


def save_checkpoint(model, path):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet18']), "Path must have the correct model name"

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if model_name == 'vgg16':
        # Check to see if model was parallelized
        if multi_gpu:
            checkpoint['classifier'] = model.module.classifier
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'resnet18':
        if multi_gpu:
            checkpoint['fc'] = model.module.fc
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['fc'] = model.fc
            checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


# %%

save_checkpoint(model, path=checkpoint_path)

# %% Load Checkpoint


def load_checkpoint(path):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Get the model name
    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet18']), "Path must have the correct model name"

    # Load in checkpoint
    checkpoint = torch.load(path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Make sure to set parameters as not trainable
        #  for param in model.parameters():
        #      param.requires_grad = False
        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


# %%

model, optimizer = load_checkpoint(path=checkpoint_path)

# %% Inferece


def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor


def predict(image_path, model, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    real_class = image_path.split('/')[-2]

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)

    # Resize
    if train_on_gpu:
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class


def random_test_image():
    """Pick a random test image from the test directory"""
    c = np.random.choice(categories)
    root = f'{new_dataset_path}/val/{c}/'
    img_path = root + np.random.choice(os.listdir(root))
    return img_path


# %% Make prediction on a radnom image

np.random.seed = 100

test_image = random_test_image()
_ = imshow_tensor(process_image(random_test_image()))
img, top_p, top_classes, real_class = predict(random_test_image(), model, topk=2)
print(top_p, top_classes, real_class)

# %% Convert pt model to ONNX

dummy_input = torch.randn(1, 3, 224, 224)
dummy_input = dummy_input.to('cuda')
input_names = ["actual_input"]
output_names = ["output"]
torch.onnx.export(model,
                  dummy_input,
                  "resnet18_speedlim.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )
