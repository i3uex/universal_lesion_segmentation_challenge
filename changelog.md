# Changelog
## 2024 - 05 - 29
- Fixed shuffle bug 
  - Fixed bug where shuffle would not work due to incorrect reference to the numpy array
- Changed SEED constant to 420 
  - Changed the SEED constant to 420 for better distribution of the different volumes
- Changed spliting of data to 70-10-20
  - Changed the spliting of the data to 70% training, 10% validation and 20% testing
- Changed from 4 patches per volume of 96x96x96 to 16 patches of 32x32x32
  - Changed the spatial size of the patches to 32x32x32 for better performance and avoiding dynamic resizing and padding.