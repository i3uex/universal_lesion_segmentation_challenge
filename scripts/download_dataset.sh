# Create folder for the datasets, if it doesn't exist

mkdir -p ./Datasets

# Download and unpack Zenodo's dataset

cd ./Datasets || exit
mkdir ./Zenodo
cd ./Zenodo || exit

wget https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip
wget https://zenodo.org/record/3757476/files/Infection_Mask.zip
wget https://zenodo.org/record/3757476/files/Lung_and_Infection_Mask.zip
wget https://zenodo.org/record/3757476/files/Lung_Mask.zip

unzip COVID-19-CT-Seg_20cases.zip -d COVID-19-CT-Seg_20cases
unzip Infection_Mask.zip -d Infection_Mask
unzip Lung_and_Infection_Mask.zip -d Lung_and_Infection_Mask
unzip Lung_Mask.zip -d Lung_Mask

rm COVID-19-CT-Seg_20cases.zip
rm Infection_Mask.zip
rm Lung_and_Infection_Mask.zip
rm Lung_Mask.zip