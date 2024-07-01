# Changelog
## 2024 - 07 - 01
- Fixed bug with AsDiscrete transformation.
  - Fixed bug where the AsDiscrete transformation would not work due to the usage of the parameter `threshold_values` instead of `threshold`.
- Added configuration of trainning process
  - Added functionality for config class, and added configuration for the training process with the `configure_training_process` process in the `main.py` file
  - Added a nets folder with the configuration of the networks
- Changed the percentage for validation and test to 20% and 10%
  - Changed the percentage of the validation and test to 20% and 10% respectively. Train is still 70%.
- Changed optimizer to Adam
  - Changed the optimizer from AdamW to Adam for simplicity.
- Added the logging of the metrics by step
## 2024 - 06 - 18
- Fixed bug with CovidDataset
  - Fixed bug where the CovidDataset would not difference between radiopaedia and coronacases due to incorrect usage of dictionary making the coronacases use the radiopaedia transformations.
- Fixed HRCT transformations bug
  - HRCT transformations weren't given the correct orientation. Changed from ALI to PLI, so the bed is on the bottom of the image.
- Fixed validation and test check_function
  - Fixed bug where the check_function would not work due to the function being nested in another function with the same name. Changed the main function ass well to be called correctly.
- Fixed bug where the train and validation loss where logged incorrectly.
  - The mean of the train and validation loss were logged incorrectly due to the division of the mean being done by the number of slices instead of the batches.
- Added tensorboard logs correctly.
  - Added tensorboard logs for training and a train-validation loss experiment.
- Fixed dataset loading images
  - The load of the images was not consistent due to the lack of sorting of the files in the dataset directory.
## 2024 - 05 - 29
- Fixed shuffle bug 
  - Fixed bug where shuffle would not work due to incorrect reference to the numpy array
- Changed SEED constant to 420 
  - Changed the SEED constant to 420 for better distribution of the different volumes
- Changed spliting of data to 70-10-20
  - Changed the spliting of the data to 70% training, 10% validation and 20% testing
- Changed from 4 patches per volume of 96x96x96 to 16 patches of 32x32x32
  - Changed the spatial size of the patches to 32x32x32 for better performance and avoiding dynamic resizing and padding.
