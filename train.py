import os
import time
import datetime
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
from networks import VAE
from data_loader import CelebAMaskHQMLoader
from utils import denorm, generate_label, convert_onehot, labels_colorize


def kl_divergence(mu, logvar):
    d_kl = 0.5 * torch.sum(mu.pow(2)+logvar.exp()-logvar-1)
    return d_kl


def pixel_cross_entropy(predict, target, weight=None, size_average=True):
    n, c, h, w = predict.size()
    nt, ht, wt = target.size()
    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        predict = f.interpolate(predict, size=(ht, wt), mode="bilinear", align_corners=True)

    predict = predict.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    return f.cross_entropy(predict, target, weight=weight, size_average=size_average, ignore_index=250)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MaskVAE Training.')
    parser.add_argument('--version', type=str, default='MaskVAE_v0')
    parser.add_argument('--dataset', type=str, default='../../segment/data/train_label',
                        help='dataset folder path of CelebAMask-HQ(label)')
    parser.add_argument('--model_save_path', type=str, default='./outputs/models')
    parser.add_argument('--sample_save_path', type=str, default='./outputs/samples')
    parser.add_argument('--continuous_training_from_iter', type=int, default=None)
    parser.add_argument('--total_step', type=int, default=40000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--beta1', type=float, default=.5)
    parser.add_argument('--beta2', type=float, default=.999)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--wt_kl', type=float, default=1e-5)
    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument('--model_save_step', type=int, default=500)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    device = args.device
    transform_label = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.imsize, args.imsize))
    ])

    dataloader = CelebAMaskHQMLoader(args.dataset, transform=transform_label,
                                     batch_size=args.batch_size, num_workers=0, mode=True)
    data_iter = iter(dataloader.load())

    vae = VAE(19, 32, 32, 1024).to(device)

    model_save_path = os.path.join(args.model_save_path, args.version)
    sample_save_path = os.path.join(args.sample_save_path, args.version)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path, exist_ok=True)

    start = 0
    if args.continuous_training_from_iter is not None:
        continuous_vae_path = os.path.join(model_save_path, f'{args.continuous_training_from_iter}_vae.pth')
        assert os.path.exists(continuous_vae_path)
        start = args.continuous_training_from_iter + 1
        vae.load_state_dict(torch.load(continuous_vae_path))

    optimizer = optim.Adam(vae.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    start_time = time.time()
    for step in range(start, args.total_step):
        vae.train()
        labels = next(data_iter)
        labels = labels.to(device)
        labels[:, 0, :, :] = labels[:, 0, :, :] * 255.
        labels_real_plain = labels[:, 0, :, :].long()     # for loss_rec computation
        labels_real_onehot = convert_onehot(labels_real_plain)   # for VAE input

        labels_predict, _, labels_mu, labels_logvar = vae(labels_real_onehot)

        loss_kl = kl_divergence(labels_mu, labels_logvar) * args.wt_kl
        loss_rec = pixel_cross_entropy(labels_predict, labels_real_plain)

        loss = loss_rec + loss_kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % args.log_step == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print("Elapsed [{}], step [{}/{}], reconstruction loss: {:.5f}, kl loss: {:.5f}, total loss: {:.5f}".
                  format(elapsed, step + 1, args.total_step, loss_rec.data, loss_kl, loss.data))

        # save intermediate model
        if (step+1) % args.model_save_step == 0:
            torch.save(vae.state_dict(), os.path.join(model_save_path, f'{step+1}_vae.pth'))

        # save sample
        if (step+1) % args.sample_step == 0:
            vae.eval()
            with torch.no_grad():
                labels_sample, _, _, _ = vae(labels_real_onehot)
                labels_rec_colors = generate_label(labels_sample, args.imsize)
                labels_ori_colors = labels_colorize(labels_real_plain.unsqueeze(1).cpu())
            torchvision.utils.save_image(torch.cat([labels_ori_colors, denorm(labels_rec_colors.data)], dim=0),
                                         os.path.join(sample_save_path, '{}.png'.format(step + 1)))

    # save final model
    torch.save(vae.state_dict(), os.path.join(model_save_path, 'vae.pth'))
