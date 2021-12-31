import torch.nn as nn
import torchvision.models as models

class VGG19(nn.Module):

    def __init__(self, model, num_classes):
        vgg19 = models.vgg19(pretrained=True)
        removed = list(vgg19.classifier.children())[:-1]
        vgg19.classifier = nn.Sequential(*removed)
        vgg19.classifier = nn.Sequential(*list(vgg19.classifier.children()),
                                         nn.Linear(in_features=4096, out_features=10, bias=True))
        super(VGG19, self).__init__()
        self.model = model

        # Freeze those weights
        for p in self.model.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        return self.model(x)
