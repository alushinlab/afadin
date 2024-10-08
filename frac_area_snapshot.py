import numpy as np
import os.path as op
import sh
from glob import glob
import statistics
import math

import nd2
import skimage.io as si
import skimage.filters as sf
import skimage.util as su
import skimage.restoration as sr


def load_img(fname_img):
    img_cyx = nd2.imread(fname_img)
    print(fname_img)
    return img_cyx


def compute_actin_mask(img_cyx):
    actin_yx = img_cyx[0, :, :]
    thresh = sf.threshold_yen(actin_yx)
    actin_mask_yx = actin_yx > thresh
    return actin_mask_yx


def compute_pentamer_mask(img_cyx):
    pentamer_yx = img_cyx[1, :, :]
    bg_yx = sr.rolling_ball(pentamer_yx, radius=100)
    pentamer_yx = pentamer_yx - bg_yx
    thresh = sf.threshold_yen(pentamer_yx)
    pentamer_mask_yx = pentamer_yx > thresh
    return pentamer_mask_yx


def compute_fraction(actin_mask_yx, pentamer_mask_yx):
    bound_mask_yx = actin_mask_yx & pentamer_mask_yx
    frac = round((np.sum(bound_mask_yx) / np.sum(actin_mask_yx)), 4)
    return frac, bound_mask_yx


def compute_fraction_for_all(pattern):
    folder_output = 'output'
    sh.mkdir('-p', folder_output)

    conc_n = []
    frac_n = []
    for dname_img in sorted(glob(pattern)):
        dname_mask = op.join('output/mask', *dname_img.split('/')[-2:])
        sh.mkdir('-p', dname_mask)
        conc = float(dname_img.split('/')[-1].removesuffix('uM'))

        for fname_img in sorted(glob(op.join(dname_img, '*.nd2'))):
            conc_n.append(conc)

            img_cyx = load_img(fname_img)
            prefix = op.basename(fname_img).removesuffix('.nd2')

            actin_mask_yx = compute_actin_mask(img_cyx)
            fname_actin_mask = op.join(dname_mask, prefix + '_actin.png')
            si.imsave(fname_actin_mask, su.img_as_ubyte(actin_mask_yx), check_contrast=False)

            pentamer_mask_yx = compute_pentamer_mask(img_cyx)
            fname_pentamer_mask = op.join(dname_mask, prefix + '_pentamer.png')
            si.imsave(fname_pentamer_mask, su.img_as_ubyte(pentamer_mask_yx), check_contrast=False)

            frac, bound_mask_yx = compute_fraction(actin_mask_yx, pentamer_mask_yx)
            frac_n.append(frac)
            fname_bound_mask = op.join(dname_mask, prefix + '_bound.png')
            si.imsave(fname_bound_mask, su.img_as_ubyte(bound_mask_yx), check_contrast=False)

            print(conc, frac)

    output = {}
    for conc, frac in zip(conc_n, frac_n):
        output.setdefault(conc, []).append(frac)

    conc_N = []
    frac_mean_N = []
    frac_std_N = []
    frac_sem_N = []
    for conc, frac_i in output.items():
        conc_N.append(conc)
        frac_mean = statistics.mean(frac_i)
        frac_mean_N.append(frac_mean)
        frac_std = statistics.stdev(frac_i)
        frac_std_N.append(frac_std)
        frac_sem = frac_std / math.sqrt(len(frac_i))
        frac_sem_N.append(frac_sem)
        print('conc %.4f, mean %.4f, std %.4f, sem %.4f' % (conc, frac_mean, frac_std, frac_sem))

    fname_frac = op.join(folder_output, 'frac_area.txt')
    with open(fname_frac, 'w') as f:
        for (conc, frac) in zip(conc_n, frac_n):
            f.write('%.4f  %.4f\n' % (conc, frac))

    fname_stats = op.join(folder_output, 'stats_area.txt')
    with open(fname_stats, 'w') as f:
        for (conc, frac_mean, frac_std, frac_sem) in zip(conc_N, frac_mean_N, frac_std_N, frac_sem_N):
            f.write('%.4f  %.4f  %.4f  %.4f\n' % (conc, frac_mean, frac_std, frac_sem))


if __name__ == '__main__':
    compute_fraction_for_all('data/snapshot/*/*uM')
    pass
