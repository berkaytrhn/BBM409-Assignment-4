import torch
import torch.nn as nn
import torchvision.models as models


class VGG19(nn.Module):

    def __init__(self, num_classes):
        vgg19 = models.vgg19(pretrained=True)
        removed = list(vgg19.classifier.children())[:-1]
        vgg19.classifier = nn.Sequential(*removed)
        vgg19.classifier = nn.Sequential(*list(vgg19.classifier.children()),
                                         nn.Linear(in_features=4096, out_features=num_classes, bias=True))
        super(VGG19, self).__init__()
        self.model = vgg19

        # Freeze those weights
        for p in self.model.features.parameters():
            p.requires_grad = False

    def forward(self, x):
            return self.vgg19(x)

    def train_model(model,
                    data_loader,
                    dataset_size,
                    optimizer,
                    scheduler,
                    num_epochs):
        criterion = nn.BCEWithLogitsLoss()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            scheduler.step()
            model.train()

            running_loss = 0.0
            # Iterate over data.
            for bi, d in enumerate(data_loader):
                inputs = d["image"]
                labels = d["labels"]
                inputs = inputs.to(torch.device, dtype=torch.float)
                labels = labels.to(torch.device, dtype=torch.float)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_size
            print('Loss: {:.4f}'.format(epoch_loss))
        return model
vgg19 = VGG19(10)

