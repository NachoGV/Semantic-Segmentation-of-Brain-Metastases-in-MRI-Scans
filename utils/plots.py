import matplotlib.pyplot as plt

def plot_gt_vs_pred(image, pred, train):

  channels = ['TC', 'WT', 'ET']

  # Labels
  plt.figure("input", (6,2))
  for i in range(3):
    plt.subplot(1, 3, i + 1)
    if train:
        plt.imshow(image[i, :, :, 70].detach().cpu())
    else:
        plt.imshow(image[i, :, :, 80].detach().cpu())
    plt.title(channels[i])
    plt.axis('off')
  plt.suptitle("Labels")
  plt.tight_layout()
  plt.show()

  # Prediction
  plt.figure("output", (6, 2))
  for i in range(3):
    plt.subplot(1, 3, i + 1)
    if train:
        plt.imshow(pred[i, :, :, 70].detach().cpu())
    else:
        plt.imshow(pred[i, :, :, 80].detach().cpu())
    plt.axis('off')
  plt.suptitle("Predictions")
  plt.tight_layout()
  plt.show()