import torchvision.transforms as transforms

from utils.eval import *

from vagan import *
import vagan

#############################
# Hyperparameters
#############################
seed = 123
lr = 0.001
beta1 = 0.0
beta2 = 0.9
num_workers = 0
data_path = "dataset"

dis_batch_size = 1  # 64
max_epoch = 40
lambda_kld = 1e-3
latent_dim = 128
cont_dim = 16
cont_k = 8192
cont_temp = 0.07
datasets = 'MNIST'
# multi-scale contrastive setting
layers = ["b1", "final"]

device_ids = [0]

# device = 1

device = torch.device("cuda:0")

name = ("").join(layers)
log_fname = f"logs/{datasets}-{name}"
fid_fname = f"logs/FID_{datasets}-{name}"
viz_dir = f"viz/{datasets}-{name}"
models_dir = f"saved_models/{datasets}-{name}"
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
lambda_cont = 1.0 / len(layers)
fix_seed(random_seed=seed)

#############################
# Make and initialize the Networks
#############################

net = vagan.SNNgenerator().cuda(device)
dual_encoder = DualDiscriminator(in_channels=1, cont_dim=16).cuda(device)  # For grayscale images

# dual_encoder = DualDiscriminator(cont_dim).cuda(device)
dual_encoder.apply(weights_init)
dual_encoder_M = DualDiscriminator(in_channels=1, cont_dim=16).cuda(device)

for p, p_momentum in zip(dual_encoder.parameters(), dual_encoder_M.parameters()):
    p_momentum.data.copy_(p.data)
    p_momentum.requires_grad = False
gen_avg_param = copy_params(net.fsdecoder)
d_queue, d_queue_ptr = {}, {}
for layer in layers:
    d_queue[layer] = torch.randn(cont_dim, cont_k).cuda(device)
    d_queue[layer] = F.normalize(d_queue[layer], dim=0)
    d_queue_ptr[layer] = torch.zeros(1, dtype=torch.long)

#############################
# Make the optimizers
#############################
opt_encoder0 = torch.optim.Adam(net.fsencoder.parameters(),
                                0.001, )
opt_decoder0 = torch.optim.Adam(net.fsdecoder.parameters(),
                                0.001, )
opt_encoder = torch.optim.Adam(net.fsencoder.parameters(),
                               lr, (beta1, beta2))
opt_decoder = torch.optim.Adam(net.fsdecoder.parameters(),
                               lr, (beta1, beta2))

shared_params = list(dual_encoder.block1.parameters()) + \
                list(dual_encoder.block2.parameters()) + \
                list(dual_encoder.block3.parameters()) + \
                list(dual_encoder.block4.parameters()) + \
                list(dual_encoder.l5.parameters())
opt_shared = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                     shared_params),
                              3 * lr, (beta1, beta2))
opt_disc_head = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        dual_encoder.head_disc.parameters()),
                                 3 * lr, (beta1, beta2))
cont_params = list(dual_encoder.head_b1.parameters()) + \
              list(dual_encoder.head_b2.parameters()) + \
              list(dual_encoder.head_b3.parameters()) + \
              list(dual_encoder.head_b4.parameters())
opt_cont_head = torch.optim.Adam(filter(lambda p: p.requires_grad, cont_params),
                                 3 * lr, (beta1, beta2))

#############################
# Make the dataloaders
#############################
print("Dataset:", datasets)
if datasets == 'CIFAR10':
    ds = torchvision.datasets.CIFAR10(data_path, train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5)),
                                      ]))
    train_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                               shuffle=True, pin_memory=True, drop_last=True,
                                               num_workers=num_workers)
    ds = torchvision.datasets.CIFAR10(data_path, train=False, download=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5)),
                                      ]))
    test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              num_workers=num_workers)

elif datasets == "MNIST":
    print("Datasets MNIST")
    # SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    # transform_train = transforms.Compose([
    #        transforms.Resize((32)),
    #        transforms.ToTensor(),
    #        SetRange
    #    ])

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure correct size is set
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    ds = torchvision.datasets.MNIST(data_path, train=True, download=True,
                                    transform=transform_train
                                    )
    train_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                               shuffle=True, pin_memory=True, drop_last=True,
                                               num_workers=0)
    ds = torchvision.datasets.MNIST(data_path, train=False, download=False,
                                    transform=transform_train
                                    )
    test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              num_workers=0)
elif datasets == 'FashionMNIST':
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((32)),
        transforms.ToTensor(),
        SetRange
    ])
    ds = torchvision.datasets.FashionMNIST(data_path, train=True, download=True,
                                           transform=transform_train
                                           )
    train_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                               shuffle=True, pin_memory=True, drop_last=True,
                                               num_workers=0)
    ds = torchvision.datasets.FashionMNIST(data_path, train=False, download=False,
                                           transform=transform_train
                                           )
    test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              num_workers=0)

global_steps = 0
best_fid = 50000
import cleanfid


def calc_clean_fid(network, epoch):
    network = network.eval()
    with torch.no_grad():
        num_gen = 5000
        fid_score = cleanfid.get_clean_fid_score(network, 'CIFAR10', 0,
                                                 num_gen)
        return fid_score


# Training loop
for epoch in range(max_epoch):
    net.train()
    dual_encoder.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)  # Assuming 'cond' requires class labels

        # Reset gradient
        opt_encoder.zero_grad()
        opt_decoder.zero_grad()
        opt_shared.zero_grad()
        opt_disc_head.zero_grad()
        opt_cont_head.zero_grad()

        # Forward pass, now including the 'cond' argument
        recon_images, q_z, p_z, sampled_z = net(images, labels)  # Corrected unpacking

        print("train images.shape", images.shape)
        real_features = dual_encoder(images)

        print("train recon_images.shape", recon_images.shape)

        fake_features = dual_encoder(recon_images.detach())

        # print("recon_images.type:", type(recon_images))
        # print("images.type:", type(images))
        # print("real_features.type:", type(real_features))
        # print("fake_features.type:", type(fake_features))

        print("recon_images.shape, images.shape, real_features.shape, fake_features.shape", recon_images.shape,
              images.shape, real_features.shape, fake_features.shape)
        # Calculate loss
        # print("recon_images.shape, images.shape", recon_images.shape, images.shape)
        recon_loss = F.mse_loss(recon_images, images)
        feature_loss = F.mse_loss(fake_features, real_features.detach())
        loss = recon_loss + lambda_kld * feature_loss

        # Backward pass
        loss.backward()
        opt_encoder.step()
        opt_decoder.step()
        opt_shared.step()
        opt_disc_head.step()
        opt_cont_head.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    # Evaluate the model
    if epoch % 5 == 4:
        fid_score = calc_clean_fid(net, epoch)
        print(f'Epoch {epoch + 1} FID score: {fid_score}')
        if fid_score < best_fid:
            best_fid = fid_score
            print(f'New best FID score: {best_fid}')
            # Save the best model
            torch.save(net.state_dict(), f'{models_dir}/best_model.pth')
            torch.save(dual_encoder.state_dict(), f'{models_dir}/best_dual_encoder.pth')

print('Finished Training')
