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
test_dir = "data/test"
test_dir1 = "data/face2comics/val/target"
learning_rate = 1e-5
lambda_cycle = 10
lambda_identity = 0
batch_size = 4
num_workers = 2
image_size = 256
channels = 3
epoch_num = 500
load_model = True
save_model = True
dis_a_checkpoint = "discriminator_a.pth"
dis_b_checkpoint = "discriminator_b.pth"
gen_a_checkpoint = "generator_a.pth"
gen_b_checkpoint = "generator_b.pth"


print("Using {} device".format(device))

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

test_transform = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)
    
# when inputs and targets are in two different folders
class MyDataset(Dataset):
    def __init__(self, A_folder, B_folder):
        self.A_folder = A_folder
        self.B_folder = B_folder
        
        self.A_filenames = os.listdir(A_folder)
        self.B_filenames = os.listdir(B_folder)
        
        self.length_dataset = max(len(self.A_filenames), len(self.B_filenames)) 
        self.style_A_len = len(self.A_filenames)
        self.style_B_len = len(self.B_filenames)

    def __len__(self):
        return self.length_dataset
            
    def __getitem__(self, index):
        A_file_name = self.A_filenames[index % self.style_A_len]
        B_file_name = self.B_filenames[index % self.style_B_len]

        A_file_path = os.path.join(self.A_folder, A_file_name)
        B_file_path = os.path.join(self.B_folder, B_file_name)

        A_file_img = np.array(Image.open(A_file_path).convert("RGB"))
        B_file_img = np.array(Image.open(B_file_path).convert("RGB"))

        transform_result = transforms(image=A_file_img, image0=B_file_img)
        A_file_img = transform_result["image"]
        B_file_img = transform_result["image0"]
        
        return A_file_img, B_file_img

def train(discriminator_A, discriminator_B, generator_A, generator_B, train_loader, disc_opt, gen_opt, l1_loss, mse, device, d_scaler, g_scaler):
    discriminator_A.train()
    discriminator_B.train()
    generator_A.train()
    generator_B.train()
  
    
    for style_A, style_B in tqdm(train_loader):
        style_A, style_B = style_A.to(device), style_B.to(device)        
        # Train discriminator
        with torch.cuda.amp.autocast():        
          disc_opt.zero_grad()
                  
          fake_A = generator_A(style_B)
          D_A_real = discriminator_A(style_A)
          D_A_fake = discriminator_A(fake_A.detach())
          D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
          D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
          D_A_loss = D_A_real_loss + D_A_fake_loss
  
          fake_B = generator_B(style_A)
          D_B_real = discriminator_B(style_B)
          D_B_fake = discriminator_B(fake_B.detach())
          D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
          D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
          D_B_loss = D_B_real_loss + D_B_fake_loss
  
          D_loss = (D_A_loss + D_B_loss) / 2
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


        # Train generator
        with torch.cuda.amp.autocast():
          D_A_fake = discriminator_A(fake_A)
          D_B_fake = discriminator_B(fake_B)
          loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
          loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))
  
          cycle_A = generator_A(fake_B)
          cycle_B = generator_B(fake_A)
          cycle_A_loss = l1_loss(style_A, cycle_A)
          cycle_B_loss = l1_loss(style_B, cycle_B)
  
          identity_A = generator_A(style_A)
          identity_B = generator_B(style_B)
          identity_A_loss = l1_loss(style_A, identity_A)
          identity_B_loss = l1_loss(style_B, identity_B)
          G_loss = loss_G_A + loss_G_B + cycle_A_loss*lambda_cycle + cycle_B_loss*lambda_cycle + identity_A_loss*lambda_identity + identity_B_loss*lambda_identity
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
    print(f"Train set: generator_A train loss: {loss_G_A:.4f}, generator_B train loss: {loss_G_B:.4f}, discriminator_A train loss: {D_A_loss:.4f}, discriminator_B train loss: {D_B_loss:.4f}")


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
            input_img = test_transform(image=img)["image"].to(device)
        
            fake_img = generator(input_img)
            fake_img = fake_img * 0.5 + 0.5  
            save_image(fake_img, folder + f"/gen_{index}.png")
            save_image(input_img * 0.5 + 0.5, folder + f"/input_{index}.png")
        print("test results saved!!!")

def test1(generator, test_dir, device):
    generator.eval()
    folder = "test_results1"
    if not os.path.exists(folder):
        os.mkdir(folder)
    test_files = os.listdir(test_dir)
    with torch.no_grad():
        for index in range(len(test_files)):
            img_file = test_files[index]
            img_path = os.path.join(test_dir, img_file)
            img = np.array(Image.open(img_path))
            input_img = test_transform(image=img)["image"].to(device)
        
            fake_img = generator(input_img)
            fake_img = fake_img * 0.5 + 0.5  
            save_image(fake_img, folder + f"/gen_{index}.png")
            save_image(input_img * 0.5 + 0.5, folder + f"/input_{index}.png")
        print("test results saved!!!")

discriminator_A = Discriminator(in_channels=3).to(device)
discriminator_B = Discriminator(in_channels=3).to(device)
generator_A = Generator(img_channels=3, num_residuals=9).to(device)
generator_B = Generator(img_channels=3, num_residuals=9).to(device)
opt_disc = optim.Adam(list(discriminator_A.parameters()) + list(discriminator_B.parameters()), lr=learning_rate, betas=(0.5, 0.999))
opt_gen = optim.Adam(list(generator_A.parameters()) + list(generator_B.parameters()), lr=learning_rate, betas=(0.5, 0.999))

l1_loss = nn.L1Loss()
mse = nn.MSELoss()

if load_model:
    gen_A_checkpoint = torch.load(gen_a_checkpoint, map_location=device)
    generator_A.load_state_dict(gen_A_checkpoint['state_dict'])
    opt_gen.load_state_dict(gen_A_checkpoint["optimizer"])
    for param_group in opt_gen.param_groups:
        param_group["lr"] = learning_rate
    print("=> generator_A checkpoint loaded!")

    gen_B_checkpoint = torch.load(gen_b_checkpoint, map_location=device)
    generator_B.load_state_dict(gen_B_checkpoint['state_dict'])
    opt_gen.load_state_dict(gen_B_checkpoint["optimizer"])
    for param_group in opt_gen.param_groups:
        param_group["lr"] = learning_rate
    print("=> generator_B checkpoint loaded!")

    disc_A_checkpoint = torch.load(dis_a_checkpoint, map_location=device)
    discriminator_A.load_state_dict(disc_A_checkpoint['state_dict'])
    opt_disc.load_state_dict(disc_A_checkpoint["optimizer"])
    for param_group in opt_disc.param_groups:
        param_group["lr"] = learning_rate
    print("=> discriminator_A checkpoint loaded!")

    disc_B_checkpoint = torch.load(dis_b_checkpoint, map_location=device)
    discriminator_B.load_state_dict(disc_B_checkpoint['state_dict'])
    opt_disc.load_state_dict(disc_B_checkpoint["optimizer"])
    for param_group in opt_disc.param_groups:
        param_group["lr"] = learning_rate
    print("=> discriminator_B checkpoint loaded!")

if is_train:
    train_A_folder = train_dir + "/face"
    train_B_folder = train_dir + "/comics"
    train_dataset = MyDataset(train_A_folder, train_B_folder)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


if is_train:
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epoch_num):
        print("=>epoch: " + str(epoch))
        train(discriminator_A, discriminator_B, generator_A, generator_B, train_loader, opt_disc, opt_gen, l1_loss, mse, device, d_scaler, g_scaler)
        if save_model:
            generator_a_checkpoint = {
                "state_dict": generator_A.state_dict(),
                "optimizer": opt_gen.state_dict(),
            }
            torch.save(generator_a_checkpoint, gen_a_checkpoint)
            print("=> generator_A checkpoint saved!")

            generator_b_checkpoint = {
                "state_dict": generator_B.state_dict(),
                "optimizer": opt_gen.state_dict(),
            }
            torch.save(generator_b_checkpoint, gen_b_checkpoint)
            print("=> generator_B checkpoint saved!")

            discriminator_a_checkpoint = {
                "state_dict": discriminator_A.state_dict(),
                "optimizer": opt_disc.state_dict(),
            }
            torch.save(discriminator_a_checkpoint, dis_a_checkpoint)
            print("=> discriminator_A checkpoint saved!")

            discriminator_b_checkpoint = {
                "state_dict": discriminator_B.state_dict(),
                "optimizer": opt_disc.state_dict(),
            }
            torch.save(discriminator_b_checkpoint, dis_b_checkpoint)
            print("=> discriminator_A checkpoint saved!")
        test(generator_B, test_dir, device=device)            
else:
    test(generator_B, test_dir, device=device)
    test1(generator_A, test_dir1, device=device)
        



