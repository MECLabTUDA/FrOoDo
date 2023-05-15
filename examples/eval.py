import torch
from torch.utils.data import Dataset
from tqdm import trange
from froodo import PneumoniaDataSetAdapter

"""
This model evaluation script was kindly provided by the other project group and adapted to the pneumonia data set. 
"""

# Download the dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# and pass the base path to the adapter here
adapter = PneumoniaDataSetAdapter("~/Downloads/chest_xray/", split="test")
dataloader = torch.utils.data.DataLoader(
    adapter.dataset, batch_size=16, shuffle=True, num_workers=0
)

dataloader = iter(dataloader)

model = torch.hub.load(
    "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False, num_classes=2
)
model.load_state_dict(torch.load("./model.pth"))
model = model.cuda()
model = model.eval()

all_correct_classifications = 0
all_classifications = 0

with torch.no_grad():
    with trange(len(dataloader)) as tbar:
        for b in tbar:
            image, label = next(dataloader)
            B = image.shape[0]
            image = image.cuda()
            label = label.cuda()

            out = model(image)
            binary_labels = torch.argmax(out, dim=1)
            correct_classifications = torch.sum(binary_labels == label)
            tbar.set_postfix(loss=correct_classifications / B)

            all_correct_classifications += correct_classifications
            all_classifications += B

print("accuracy: ", all_correct_classifications / all_classifications)
