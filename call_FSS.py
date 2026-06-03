import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

# generate reproducible example fields
def find_index(x0, y0, x, y):
    dx = np.abs(x - x0)
    dy = np.abs(y - x0)
    xidx = np.argmin(dx)
    yidx = np.argmin(dy)
    return xidx, yidx

def make_gauss_bell(center_x, center_y, width, height, xgrid, ygrid, zgrid):
    #x0, y0 = find_index(center_x, center_y, xgrid[0,:], ygrid[:,0])
    bell_mountain = np.exp(-(
        np.square(xgrid - center_x) / (2 * width ** 2) +
        np.square(ygrid - center_y) / (2 * width ** 2)))
    # bell_mountain = np.where(bell_mountain > 0.01, bell_mountain, 0.)
    zgrid += bell_mountain

def label_contours(ax, cl):
    ax.clabel(
        cl,                        # Typically best results when labelling line contours.
        colors=['black'],
        manual=False,              # Automatic placement vs manual placement.
        inline=True,               # Cut the line where the label will be placed.
        fmt=' {:.1f} '.format,     # Labes as integers, with some extra space.
        fontsize = 8)

x = np.arange(-3., 3.01, 0.01)
y = x
xx, yy = np.meshgrid(x, y)
zzm = np.zeros(xx.shape)
zzo = np.zeros(xx.shape)

# np.random.seed(seed=32)

np.random.seed(seed=127)

for ii in range(20):
    x = -2. + 4. * np.random.rand()
    y = -2. + 4. * np.random.rand()
    a = 2.2 + 1.2 * np.random.rand()
    s = 0.2 + 0.5 * np.random.rand()
    make_gauss_bell(x, y, s, a, xx, yy, zzm)
    x = -2. + 4. * np.random.rand()
    y = -2. + 4. * np.random.rand()
    a = 2.2 + 1.2 * np.random.rand()
    s = 0.2 + 0.5 * np.random.rand()
    make_gauss_bell(x, y, s, a, xx, yy, zzo)
thresholds = [0.2, 0.5, 1., 1.5, 2., 3., 4.] #, 5., 6., 7., 8.] #, 10., 12.]

f0 = []
for t in thresholds:
    f0.append(0.5 * (1. + np.where(zzo > t, 1., 0.).sum() / float(zzo.size)))
f0_fine = []
for t in np.arange(0., 1.001, 0.002) * 5:
    f0_fine.append(0.5 * (1. + np.where(zzo > t, 1., 0.).sum() / float(zzo.size)))

# draw obs and model fields
cmap = mpl.cm.GnBu
cmap.set_under(color=(1, 1, 1), alpha=1.)
fig, axs = plt.subplots(1,2, figsize=(15, 5.75), dpi=300, 
                        sharex=True, sharey=True)
levels = np.arange(0.0, np.ceil(np.max([zzo.max(), zzm.max()])), 0.2)
# print(zz.min(), zzo.min(), zzc.min(), zzoc.min())

print(thresholds)

norm = colors.BoundaryNorm(boundaries=thresholds, ncolors=256)

c = axs[0].contourf(zzo, cmap=cmap, levels=thresholds, extend='both')
cl1 = axs[0].contour(zzo, levels=thresholds, colors='k', linewidths=.3, norm=norm)
label_contours(axs[0], cl1)

axs[1].contourf(zzm, cmap=cmap, levels=thresholds, extend='both')
cl2 = axs[1].contour(zzm, levels=thresholds, colors='k', linewidths=.3, norm=norm)
label_contours(axs[1], cl2)

axs[0].set_ylabel("grid points")
axs[0].set_xlabel("grid points")
axs[1].set_xlabel("grid points")

axs[0].set_title(f"a) observations {zzo.max()}", loc="left")
axs[1].set_title(f"b) forecast {zzm.max()}", loc="left")
# plt.suptitle("Pseudo-rainfield for Model and Obs")
plt.tight_layout()
fig.subplots_adjust(right=0.9)
cax = fig.add_axes([0.92, 0.13, 0.02, 0.75])
plt.colorbar(c, cax=cax, label="RR [mm]")

plt.savefig("fields.png")
thresholds = [0.2, 0.5, 1., 1.5, 2., 3., 4.] #, 5., 6., 7., 8.] #, 10., 12.]
windows = np.arange(10, 400, 20, dtype=int)

# add calls to calculate the FSS using SAT, FFT, and OpenCL and other stuff elow this
