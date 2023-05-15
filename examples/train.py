from torch.utils.data import Dataset
from froodo import *
from tqdm import trange
import numpy as np

import matplotlib.pyplot as plt

from froodo import PneumoniaDataSetAdapter

"""
This model training script was kindly provided by the other project group and adapted to the pneumonia data set. 
"""

# Download the dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# and pass the base path to the adapter here
adapter = PneumoniaDataSetAdapter("~/Downloads/chest_xray/", split="train")
dataloader = torch.utils.data.DataLoader(
    adapter.dataset, batch_size=16, shuffle=True, num_workers=0
)

model = torch.hub.load(
    "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False, num_classes=2
)
model = model.cuda()
model = model.train()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
loss_func = torch.nn.CrossEntropyLoss()

# train
num_epochs = 50
num_samples_per_epoch = 100


assert num_samples_per_epoch <= len(dataloader)
all_losses = []


for epoch in range(num_epochs):
    epoch_losses = []
    print(
        "start epoch {epoch} using LR: {lr:.4f}".format(
            epoch=epoch, lr=optimizer.param_groups[0]["lr"]
        )
    )

    with trange(num_samples_per_epoch) as tbar:
        for b in tbar:
            tbar.set_description("Epoch {}/{}".format(epoch + 1, num_epochs))

            image, label = next(iter(dataloader))
            image = image.cuda()
            label = label.cuda()
            optimizer.zero_grad()

            out = model(image)

            loss = loss_func(out, label)

            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()

            tbar.set_postfix(loss=loss)
            epoch_losses.append(loss)
    mean = np.mean(epoch_losses)
    all_losses.append(mean)
    print("mean loss: ", mean)
    lr_scheduler.step(mean)

plt.plot(all_losses)
plt.title("Training Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

torch.save(model.state_dict(), "./model.pth")
torch.save(optimizer.state_dict(), "./optimizer.pth")
torch.save(lr_scheduler.state_dict(), "./scheduler.pth")
