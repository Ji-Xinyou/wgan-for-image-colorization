# wgan-for-image-colorization
Wasserstein GAN for image colorization, fast and easy to train, relatively good performance.

In my experiment, with 5000 imgs as training set, 400 imgs as test set, this method performs close to Instance-aware colorization(with officially provided weight) under the metric of **PSNR and SSIM**. But this method only takes only 5000 imgs to train, and less than 8 hours on a single RTX 3080.

## how to run
Currently this repo is not carefully written, sorry if inconvenience.

1. read train.py, especially the argument parser
2. create coressponding folder (/checkpoint etc.)
3. if u want to use wgan, add `--wgan`; if u want to use sobel, add `--Sobel`.

p.s. the original code uses `torch.save(model, path)` to save model, it is **NOT GOOD FOR HEALTH**, if u want to save checkpoints, use `torch.save(model.state_dict(), path)` instead, since former one requires the invariance of directory structure.

Feel free to contact me through email if needed.
