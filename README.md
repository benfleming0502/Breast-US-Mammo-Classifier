This repository contains the code base for the breast cancer patient classification models that are to be used as apart of my dissertation.
The datasets used will include ultrasounds and mammograms.
I won't be comitting any of the images to the repository, as they are confidential, if you wish to use your own copies of the datasets follow the below instructions.

Currently only the DCL_USG dataset is automated.
Copy the Benign, Malignant and Normal folders into a parent folder titled "DCL_USG", which should be placed in the root.
Then run the strip_dcl_usg_dataset.py file, which will edit the files to the format we are using, and generates a test and training split,
which will be under training_ultrasounds and test_ultrasounds. Then you can run main.py to start the training process.

There's no dedicated testing program yet, as I'm just prototyping, the final test accuracy can be found on the last epoch of the training log that is outputted to the console.
