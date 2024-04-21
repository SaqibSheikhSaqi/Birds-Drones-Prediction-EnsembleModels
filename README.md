Due to limited size, the test and training data are divided into different folders. Create the following two folders: 'Train' and 'Test'. Then, within each folder, create 'Birds' and 'Drones' subfolders.
Unzip the test and train datasets into the relevant folders. The structure should resemble:
BirdsDronesDataset
    Train
        -Birds
        -Drones
    Test
        -Birds
        -Drones
Ensembler_BirdsDrones.ipynb is the ensemble model for birds and drones prediction.
The dataset utilized comprises 4212 images for each category of 'bird' and 'drone'. 
Stringent methodology was applied for dataset preparation and model training to ensure the reliability of the findings. 
The proposed MobVGG model, trained on both 'bird' and 'drone' images, achieves superior accuracy (96%)
