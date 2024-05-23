import logging
from preprocessing.covid_dataset import CovidDataset
import monai.data
from monai.data import DataLoader, decollate_batch, pad_list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from preprocessing.transforms import get_hrct_transforms, get_cbct_transforms, \
    get_val_hrct_transforms, get_val_cbct_transforms
from utils.helpers import load_images_from_path
from config.constants import (COVID_CASES_PATH, INFECTION_MASKS_PATH, SEED)
from config.config import Config
import torch
import numpy as np
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer, sliding_window_inference


def main():
    # Create a Config instance and get the arguments
    config = Config().get_args()

    # Load images and masks
    logging.info(f"Loading images from {COVID_CASES_PATH}")
    images = load_images_from_path(COVID_CASES_PATH)
    labels = load_images_from_path(INFECTION_MASKS_PATH)

    # Convert images and masks to a list of dictionaries with keys "img" and "mask"
    data_dicts = [{"img": img, "mask": mask} for img, mask in zip(images, labels)]
    logging.debug(data_dicts)

    shuffler = np.random.RandomState(SEED)
    shuffler.shuffle(np.array(data_dicts))
    data_dicts = list(data_dicts)

    test_train_split = int(len(data_dicts) * 0.8)
    train_paths, test_paths = data_dicts[:test_train_split], data_dicts[test_train_split:]

    val_train_split = int(len(train_paths) * 0.8)
    train_paths, val_paths = train_paths[:val_train_split], train_paths[val_train_split:]

    training_ds = CovidDataset(volumes=train_paths, hrct_transform=get_hrct_transforms(), cbct_transform=get_cbct_transforms())
    validation_ds = CovidDataset(volumes=val_paths, hrct_transform=get_val_hrct_transforms(), cbct_transform=get_val_cbct_transforms())
    test_ds = CovidDataset(volumes=test_paths, hrct_transform=get_val_hrct_transforms(), cbct_transform=get_val_cbct_transforms())

    batch_size = 1
    # Define the dataloaders for training, validation
    train_loader = DataLoader(training_ds, batch_size=batch_size,  pin_memory=True)
    val_loader = DataLoader(validation_ds, batch_size=batch_size,  pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,  pin_memory=True)

    # Now iterate over the training and validation dataloaders to train a model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    post_transform = monai.transforms.Compose([monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold_values=0.5)])

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),

    ).to(device)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training and validation loop
    for epoch in range(config.epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config.epochs}")

        # Initialize the epoch loss and step counter
        epoch_loss = 0
        step = 1

        # Calculate the number of steps per epoch
        steps_per_epoch = len(training_ds) // train_loader.batch_size

        # Set the model to training mode
        model.train()

        # Iterate over the training data
        for batch_data in train_loader:
            # Move the inputs and labels to the device
            inputs, labels = batch_data['img'].to(device), batch_data['mask'].to(device)

            print(inputs.shape, labels.shape)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: compute the output by passing inputs to the model
            outputs = model(inputs)

            # Compute the loss
            loss = loss_function(outputs, labels)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            # Update the epoch loss
            epoch_loss += loss.item()

            # Print the current step and loss
            print(f"{step}/{steps_per_epoch}, train_loss: {loss.item()}")

            # Increment the step counter
            step += 1

        # Print the average loss for this epoch
        print(f"epoch {epoch + 1} average loss: {epoch_loss / steps_per_epoch}")

        # Set the model to evaluation mode
        model.eval()

        # Validation loop
        with torch.inference_mode():
            # Initialize the validation loss and step counter
            val_loss = 0
            val_steps = 1
            # Iterate over the validation data
            for val_data in val_loader:
                # Move the inputs and labels to the device
                inputs, labels = val_data['img'].to(device), val_data['mask'].to(device)

                # Forward pass: compute the output by passing inputs to the model
                # Use the inferer for the forward pass
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_logits = sliding_window_inference(inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_transform(i) for i in decollate_batch(val_logits)]

                dice_metric(y_pred=val_outputs, y=labels)

                # Increment the step counter
                val_steps += 1

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            # Print the validation loss for this epoch
            print(f"epoch {epoch + 1} Dice Score: {metric:.3f}")

    print("Finished Training")

    # dice_metric = DiceMetric(include_background=False, reduction="mean")
    #
    # # base on test_loader generate outputs for evaluation
    # outputs = [net(test_data["img"].to(device)) for test_data in test_loader]
    #
    # # now from the test_ds get the labels for evaluation
    # labels = [test_data["mask"] for test_data in test_ds]
    #
    # # labels to device
    # labels = torch.stack(labels).to(device)
    #
    # # compute metric based on the outputs and labels
    # metric = dice_metric(y_pred=outputs, y=labels)
    #
    # print("Dice score:", metric)


if __name__ == '__main__':
    main()
