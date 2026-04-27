from CustomModels import *
from test_all_checkpoints import test_models_checkpoints
from torchvision import transforms
import CustomTransforms

def main():

    regular_transforms = transforms.Compose([
            transforms.Resize((591, 741)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    preprocessed_transforms = transforms.Compose([
            transforms.Resize((591, 741)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            CustomTransforms.CLAHETransform(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
    # Testing lowres preprocessed
    model = PatchTransformerResnet18()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./patch18-preproc-checkpoints-lowres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "patch-18-preproc-lowres",
                            image_transforms=preprocessed_transforms)
    del model
    torch.cuda.empty_cache()

    model = PatchResnet18()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./resnet18-preproc-checkpoints-lowres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "resnet-18-preproc-lowres",
                            image_transforms=preprocessed_transforms)
    del model
    torch.cuda.empty_cache()

    model = PatchTransformerResnet50()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./patch50-preproc-checkpoints-lowres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "patch-50-preproc-lowres",
                            image_transforms=preprocessed_transforms)
    del model
    torch.cuda.empty_cache()

    model = PatchResnet50()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./resnet50-preproc-checkpoints-lowres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "resnet-50-preproc-lowres",
                            image_transforms=preprocessed_transforms)
    del model
    torch.cuda.empty_cache()

    # Testing lowres unprocessed
    model = PatchTransformerResnet18()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./patch18-checkpoints-lowres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "patch-18-lowres",
                            image_transforms=regular_transforms)
    del model
    torch.cuda.empty_cache()

    model = PatchResnet18()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./resnet18-checkpoints-lowres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "resnet-18-lowres",
                            image_transforms=regular_transforms)
    del model
    torch.cuda.empty_cache()

    model = PatchTransformerResnet50()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./patch50-checkpoints-lowres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "patch-50-lowres",
                            image_transforms=regular_transforms)
    del model
    torch.cuda.empty_cache()

    model = PatchResnet50()

    test_dir = "./test_mammograms"
    checkpoint_dir = "./resnet50-checkpoints-lowres"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_models_checkpoints(model, test_dir, checkpoint_dir, device, "resnet-50-lowres",
                            image_transforms=regular_transforms)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()