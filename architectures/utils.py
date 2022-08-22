import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np

def get_saved_model(architecture, weight_path):
    """
    @param weight_path: points to path where weights are stored to load model
    @returns: Trained model
    @throws: Exception if no weights stored at weight_path
    """
    if os.path.isfile(weight_path):
        model = architecture().cpu()
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        return model
    else:
        raise Exception(f'No such file {weight_path}.')


class NormalizeImage:
  """
  Transform to normalize mean and std of pixel values (better for training)
  """
  def __call__(self, image):
    """
    Parameters:
      image has shape (1, height, width)
    Returns:
      same image but where the contents of each image have been normalized to mean 0 std 1
    """
    height, width = image.shape[1], image.shape[2]
    _image = image.reshape(height, width)
    mean, std = torch.mean(_image), torch.std(_image)
    return (_image - mean)/std

  
mnist_transform = transforms.Compose([transforms.ToTensor()])
mnist_training_set = tv.datasets.MNIST(root = './data', train = True, download = True, transform=mnist_transform)

def get_recreated_image(model, img):
  img = img.reshape(1, 1, 28, 28)
  after_img = model(img).reshape(28,28).detach()
  return after_img

def show_examples(model, num_examples = 20):
  examples_img = np.zeros((2 * 28, num_examples * 28))
  # fig, ax = plt.subplots(2, num_examples, figsize=(10 * num_examples, 10))
  for img_idx in range(num_examples):
    img = mnist_training_set[img_idx][0] # 0 for Tensor, 1 for label
    after_img = get_recreated_image(model, img)
    examples_img[0:28, 28*img_idx:28*(img_idx+1)] = img
    examples_img[28:, 28*img_idx:28*(img_idx+1)] = after_img
  plt.axis('off')
  plt.imshow(examples_img)

def get_img(idx):
  return mnist_training_set[idx][0]

def interpolate(model, start_img, end_img, num_frames = 5):
  _start_img = start_img.reshape(1, 1, 28, 28)
  _end_img = end_img.reshape(1, 1, 28, 28)
  latent_start_img = model.encoder(_start_img)
  latent_end_img = model.encoder(_end_img)
  
  frame_fade_levels = np.linspace(0, 1, num_frames)
  latent_img_frames = [(1-fade_level) * latent_start_img + fade_level * latent_end_img for fade_level in frame_fade_levels]
  decoded_img_frames = list(map(lambda latent_img_frame: torch.sigmoid(model.decoder(latent_img_frame)), latent_img_frames))
  return list(map(lambda decoded_img_frame: decoded_img_frame.reshape(28, 28).detach(), decoded_img_frames))

def display_interpolation_by_img(model, start_img, end_img, num_frames = 5):
  frames = interpolate(model, start_img, end_img, num_frames)
  interpolated_img = np.zeros((28, 28*num_frames))
  for idx, frame in enumerate(frames):
    interpolated_img[0:28, 28*idx:28*(idx+1)] = frames[idx]
  plt.figure(figsize = (30,30*num_frames))
  plt.axis('off')
  plt.imshow(interpolated_img)

def display_interpolation_by_idx(model, start_idx, end_idx, num_frames):
  start_img = get_img(start_idx)
  end_img = get_img(end_idx)
  display_interpolation_by_img(model, start_img, end_img, num_frames)


