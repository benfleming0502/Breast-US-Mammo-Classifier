import strip_york_US_dataset
import strip_dcl_USG_dataset
import strip_spectra_US_dataset

def main(reset=True, train_split=0.7):
    strip_york_US_dataset.main(reset=reset, train_split=train_split)
    strip_dcl_USG_dataset.main(reset=False, train_split=train_split)
    strip_spectra_US_dataset.main(reset=False, train_split=train_split)


if __name__ == "__main__":
    main()