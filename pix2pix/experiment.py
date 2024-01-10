import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
import torch.optim as optim
from Generator import Generator
from Discriminator import Discriminator
from tqdm import tqdm 
import torch.nn as nn

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_train = False
train_dir = "data/face2comics/train"
val_dir = "data/face2comics/val"
test_dir = "data/test"
learning_rate = 2e-4
batch_size = 16
num_workers = 1
image_size = 256
channels = 3
L1_lambda = 100
lambda_gp = 10
epoch_num = 500
load_model = True
save_model = True
discriminator_checkpoint = "discriminator.pth"
generator_checkpoint = "generator.pth"


print("Using {} device".format(device))

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
    ],
    additional_targets={"image0": "image"},
)

test_transform = A.Compose([A.Resize(width=256, height=256), A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0), ToTensorV2()])
my_transform = A.Compose([A.Resize(width=256, height=256)])
input_transform = A.Compose([A.ColorJitter(p=0.2), A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0), ToTensorV2()])

target_transform = A.Compose([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0), ToTensorV2()])

# when inputs and targets are concatenated 
class PairedDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_file = self.files[index]
        img_path = os.path.join(self.root_dir, img_file)
        img = np.array(Image.open(img_path))
        input_img = img[:, :600, :]
        target_img = img[:, 600:, :]

        input_img = my_transform(image=input_data)["image"]
        target_img = my_transform(image=target_data)["image"]

        input_img = input_transform(image=input_img)["image"]
        target_img = target_transform(image=target_img)["image"]

        return input_img, target_img
    
# when inputs and targets are in two different folders
class MyDataset(Dataset):
    def __init__(self, input_folder, target_folder):
        self.input_folder = input_folder
        self.target_folder = target_folder
        
        self.input_filenames = os.listdir(input_folder)
        self.target_filenames = os.listdir(target_folder)
        
        self.input_filenames.sort()
        self.target_filenames.sort()
        
    def __len__(self):
        return len(self.input_filenames)
    
    def __getitem__(self, index):
        input_filepath = os.path.join(self.input_folder, self.input_filenames[index])
        target_filepath = os.path.join(self.target_folder, self.target_filenames[index])
        

        input_data = np.array(Image.open(input_filepath))
        target_data = np.array(Image.open(target_filepath))
        data_transform = transforms(image=input_data, image0=target_data)
        input_img = data_transform["image"]
        target_img = data_transform["image0"]
        
        input_img = input_transform(image=input_img)["image"]
        target_img = target_transform(image=target_img)["image"]
        
        return input_img, target_img

def train(discriminator, generator, train_loader, disc_opt, gen_opt, l1_loss, bce, device):
    generator.train()
    discriminator.train()
    discriminator_train_loss = 0.0
    generator_train_loss = 0.0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train discriminator
        disc_opt.zero_grad()
        fake_result = generator(inputs)
        D_real = discriminator(inputs, targets)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = discriminator(inputs, fake_result.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        discriminator_loss = D_real_loss + D_fake_loss
        discriminator_loss.backward()
        discriminator_train_loss += discriminator_loss.item()
        disc_opt.step()

        # Train generator
        gen_opt.zero_grad()
        G_fake_loss = bce(discriminator(inputs, fake_result), torch.ones_like(D_fake))
        generator_loss = l1_loss(fake_result, targets)*L1_lambda + G_fake_loss
        generator_loss.backward()
        gen_opt.step()
        generator_train_loss += generator_loss.item()


    generator_train_loss /= len(train_loader)
    discriminator_train_loss /= len(train_loader)

    print(f"Train set: generator train loss: {generator_train_loss:.4f}, discriminator train loss: {discriminator_train_loss:.4f}")

    return generator_loss, discriminator_loss


def validate(generator, discriminator, val_loader, l1_loss, bce, device):
    generator.eval()
    discriminator.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            fake_img = generator(inputs)
            D_fake = discriminator(inputs, fake_img)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            generator_loss = l1_loss(fake_img, targets)*L1_lambda + G_fake_loss
            val_loss += generator_loss.item()

    val_loss /= len(val_loader)
    print(f"Validation set: generator validate loss: {val_loss:.4f}")
    return val_loss

def test(generator, test_dir, device):
    generator.eval()
    folder = "test_results"
    if not os.path.exists(folder):
        os.mkdir(folder)
    test_files = os.listdir(test_dir)
    with torch.no_grad():
        for index in range(len(test_files)):
            img_file = test_files[index]
            img_path = os.path.join(test_dir, img_file)
            img = np.array(Image.open(img_path))
            input_img = my_transform(image=img)["image"]
            input_img = input_transform(image=input_img)["image"].to(device)
            input_img = input_img.unsqueeze(0)
            fake_img = generator(input_img)
            fake_img = fake_img * 0.5 + 0.5  
            save_image(fake_img, folder + f"/gen_{index}.png")
            save_image(input_img * 0.5 + 0.5, folder + f"/input_{index}.png")

   
discriminator = Discriminator(in_channels=3).to(device)
generator = Generator(in_channels=3, features=64).to(device)
opt_disc = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999),)
opt_gen = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
BCE = nn.BCEWithLogitsLoss()
L1_LOSS = nn.L1Loss()

if load_model:
    gen_checkpoint = torch.load(generator_checkpoint, map_location=device)
    generator.load_state_dict(gen_checkpoint['state_dict'])
    opt_gen.load_state_dict(gen_checkpoint["optimizer"])
    for param_group in opt_gen.param_groups:
        param_group["lr"] = learning_rate
    print("=> generator checkpoint loaded!")

    disc_checkpoint = torch.load(discriminator_checkpoint, map_location=device)
    discriminator.load_state_dict(disc_checkpoint['state_dict'])
    opt_disc.load_state_dict(disc_checkpoint["optimizer"])
    for param_group in opt_disc.param_groups:
        param_group["lr"] = learning_rate
    print("=> discriminator checkpoint loaded!")




if is_train:
  #train_dataset = PairedDataset(root_dir=train_dir)
  train_input_folder = train_dir + "/input"
  train_target_folder = train_dir + "/target"
  train_dataset = MyDataset(input_folder=train_input_folder,target_folder=train_target_folder)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  #val_dataset = PairedDataset(root_dir=val_dir)
val_input_folder = val_dir + "/input"
val_target_folder = val_dir + "/target"
val_dataset = MyDataset(input_folder=val_input_folder,target_folder=val_target_folder)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def save_some_examples(gen, val_loader, epoch, folder, device):
    my_input, my_target = next(iter(val_loader))
    my_input, my_target = my_input.to(device), my_target.to(device)
    gen.eval()
    with torch.no_grad():
        fake_result = gen(my_input)
        fake_result = fake_result * 0.5 + 0.5  
        save_image(fake_result, folder + f"/y_gen_{epoch}.png")
        save_image(my_input * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(my_target * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()

if is_train:
  for epoch in range(epoch_num):
      print("=>epoch: " + str(epoch))
      train(discriminator, generator, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, device)
      validate(generator, discriminator, val_loader, L1_LOSS, BCE, device)
      test(generator, test_dir, device=device)
      if save_model:
          gen_checkpoint = {
              "state_dict": generator.state_dict(),
              "optimizer": opt_gen.state_dict(),
          }
          torch.save(gen_checkpoint, generator_checkpoint)
          print("=> generator checkpoint saved!")
  
          disc_checkpoint = {
              "state_dict": discriminator.state_dict(),
              "optimizer": opt_disc.state_dict(),
          }
          torch.save(disc_checkpoint, discriminator_checkpoint)
          print("=> discriminator checkpoint saved!")
          save_some_examples(generator, val_loader, epoch, folder="evaluation", device=device)
          
else:
  save_some_examples(generator, val_loader, 0, folder="evaluation", device=device)
  test(generator, test_dir, device=device)


