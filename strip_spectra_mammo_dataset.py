import os
import shutil
import random
from utils import remove_bad_files

def get_last_segment(file_dir):
    try:
        id = str.split(str.split(file_dir, "\\")[-1], " ")[0]
    except:
        id = str.split(str.split(file_dir, "\\")[-1], "_")[0]
    last_segment = id
    if id == "Ultrasound" or id == "Ultrasouound":
        print(file_dir)
        return None
    if id.lower() == "mammogram" or id.lower() == "mammogram]":
        try:
            id = str.split(str.split(file_dir, "\\")[-2], " ")[1]
        except:
            id = str.split(str.split(file_dir, "\\")[-2], "_")[1]
        last_segment = "SPECTRA_" + id

    return last_segment


def get_split(dataset_location,
              input_class, 
              output_class, 
              split_ratio, 
              training_location, 
              test_location,
              progress_notice=True):
    training_output = training_location + output_class + "\\"
    test_output = test_location + output_class + "\\"
    images = {}
    for root, dirs, files in os.walk(dataset_location + input_class):
        # This allows us to sample per patient, not per image, to leave open
        # adaptation opportunities when combining multiple images from the same
        # patient.
        for file in files:
            if file[0] != "A" or "Normal" in root:
                patient_ID = get_last_segment(root)
                if not patient_ID is None:
                    images.setdefault(patient_ID, [])
                    images[patient_ID].append(root + "\\" + file)

    # Copy images from source to the training samples of the class provided.
    i = 0
    image_keys = list(images.keys())
    training_size = int(split_ratio*len(images))
    training_keys = random.sample(image_keys, training_size)
    print(f"Copying {training_size} training samples")
    for patient_ID in training_keys:
        i += 1
        if progress_notice and i % 10 == 0:
            print(f"Copying patient {i}/{training_size}")
        for image in images[patient_ID]:
            ls = get_last_segment(image)
            if not ls is None:
                shutil.copyfile(image,
                           training_output
                           + patient_ID
                           + "_"
                           + ls)
    print(f"Finished copying {i}/{training_size} samples")
    # Copy images from source to the test sample location.
    test_size = len(images)-training_size
    print(f"Copying {test_size} test samples")
    i = 0
    test_keys = list(set(image_keys) - set(training_keys))
    for patient_ID in test_keys:
        i += 1
        if progress_notice and i % 10 == 0:
            print(f"Copying patient {i}/{test_size}")
        for image in images[patient_ID]:
            ls = get_last_segment(image)
            if not ls is None:
                shutil.copyfile(image,
                           test_output
                           + patient_ID
                           + "_"
                           + ls)
    print(f"Finished copying {i}/{test_size} samples")
            

def reset_workspace(test, training):

    if os.path.isdir(test):
        print("Deleting previous test data")
        shutil.rmtree(test)
    if os.path.isdir(training):
        print("Deleting previous training data")
        shutil.rmtree(training)

    test_benign = test + "benign"
    test_malignant = test + "malignant"
    test_normal = test + "normal"
    training_benign = training + "benign"
    training_malignant = training + "malignant"
    training_normal = training + "normal"

    os.makedirs(test_benign)
    os.makedirs(test_malignant)
    os.makedirs(test_normal)
    os.makedirs(training_benign)
    os.makedirs(training_malignant)
    os.makedirs(training_normal)

def main(reset=True,train_split=0.7):
    wrk_dir = os.getcwd()
    test = wrk_dir + "\\test_mammograms\\"
    training = wrk_dir + "\\training_mammograms\\"
    dataset = wrk_dir + "\\Spectra_Mammos\\"

    if reset:
        reset_workspace(test, training)

    get_split(dataset, "Benign", "benign", train_split, training, test)
    get_split(dataset, "Malignant", "malignant", train_split, training, test)
    get_split(dataset, "Normal", "normal", train_split, training, test)

    remove_bad_files(training)
    remove_bad_files(test)

if __name__ == "__main__":
    main()