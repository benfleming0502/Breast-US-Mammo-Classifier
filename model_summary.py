import torchinfo
from CustomModels import PatchTransformerResnet18
from torchvision.models import resnet50

test_model = PatchTransformerResnet18()

summ = torchinfo.summary(test_model, (1, 3, 2364, 2964))
