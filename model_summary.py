import torchinfo
from CustomModels import PatchResnet50, PatchTransformerResnet50


test_model = PatchResnet50()

summ = torchinfo.summary(test_model, (1, 3, 2364, 2964))

test_model2 = PatchTransformerResnet50()

summ2 = torchinfo.summary(test_model2, (1, 3, 2364, 2964))
