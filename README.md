# TRANSFER LEARNING EFFICIENTNET


This was my final project required to conclude my study at university.

## Features
I've used Pytorch pakcage to utilize transfer learning EfficientNet and I also used some other packages like pytorch-summary for print the summary of the used model, splitfolders for splitting the datasets based on this project interest. the list packages I've used can be found at `enviroment.yml`
I've created some other costume object like 
- `Config` for storing configurable value like ratio for data splitting, learning-rate's value and more.
- `CheckPoint` for storing some state used by this project like best weights of used model, accuracy score, loss score and more.
- `Plotting` for plotting releated purpose
- `EarlyStopping` object for regulating the trainig using early stopping method
- `Controller` is like the engine which run the training loop process.

objects I've created are heavily inspired from the [jcopdl](https://pypi.org/project/jcopdl/)'s package.

## Brief Explanation
My final project is about to train a pretrained deep learning model (EfficientNet-B0) in order to be able to classify aksara-Sunda's image.