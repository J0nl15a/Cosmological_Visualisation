import matplotlib
matplotlib.use('Pdf')
import matplotlib.patches as mpatches
import math
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sphviewer.tools import QuickView
from matplotlib.colors import LogNorm
import pyread_eagle as read_eagle
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
#plt.rcParams["text.usetex"] = True
np.warnings.filterwarnings('ignore')
import sys
import glob

from mpi4py import MPI

# In[2]:

path = pd.read_csv(f'/cosma7/ICC-data/mccarthy/cosma5/EAGLE/L025N512/zooms/x064/halo_23_dn_50p0_ds_1p0_lb_movie_redux/path2.txt', header=None, sep=None, usecols=[6,12,18])

sim = f'/cosma7/ICC-data/mccarthy/cosma5/EAGLE/L025N512/zooms/x064/halo_23_dn_50p0_ds_1p0_lb_movie_redux/data/snapshot_'

def read_data_ARTEMIS(snap, name, part, sim=sim, path=path, boxsize=25, h=0.7, centre=False):
    file_name = glob.glob(sim + f'{snap}_z*p*/snap_{snap}_z*p*.0.hdf5')
    print(file_name)
    snap_file = read_eagle.EagleSnapshot(file_name[0])
    snap_file.select_region(path.loc[snap-85,6] - boxsize, path.loc[snap-85,6] + boxsize, path.loc[snap-85,12] - boxsize, path.loc[snap-85,12] + boxsize, path.loc[snap-85,18] - boxsize, path.loc[snap-85,18] + boxsize)
    pos = snap_file.read_dataset(part, "Coordinates") / h
    df = pd.DataFrame(pos, columns=['x','y','z'])
    if part != 1:
        df['mass'] = snap_file.read_dataset(part, "Mass") / h * 1e10
        df['hsml'] = snap_file.read_dataset(part, "SmoothingLength") / h
    else:
        pass
    
    df.name = name

    if centre == True:
        df['x'] = df.x - path.loc[snap-85,6] + boxsize / 2
        df['y'] = df.y - path.loc[snap-85,12] + boxsize / 2
        df['z'] = df.z - path.loc[snap-85,18] + boxsize / 2
        df.x.loc[df.x < 0] = df.x.loc[df.x < 0] + boxsize
        df.y.loc[df.y < 0] = df.y.loc[df.y < 0] + boxsize
        df.z.loc[df.z < 0] = df.z.loc[df.z < 0] + boxsize
        df.x.loc[df.x > boxsize] = df.x.loc[df.x > boxsize] - boxsize
        df.y.loc[df.y > boxsize] = df.y.loc[df.y > boxsize] - boxsize
        df.z.loc[df.z > boxsize] = df.z.loc[df.z > boxsize] - boxsize
    
    return df


def limit(lim, field, entry):
    max_val = pd.Series(lim, index=range(len(field)))
    field[entry] = field.mask(field > lim, max_val, axis=0)
    return field


def img_norm(img, boxsize):
    img_norm = img / np.power(boxsize / np.shape(img)[0], 2)
    #print(img_norm)
    #max_5 = np.max(img_norm)/5
    #img_norm[np.where(img_norm < max_5)] = max_5
    img_norm[np.where(img_norm < 1e-8)] = 1e-8
    img_log = np.log10(img_norm)
    return img_log


def trajectory(angle, flight_time, field, boxsize, frame, colour, extent=None, phi=0):

    steps = math.ceil(flight_time*25)

    for a in np.linspace(phi, phi+steps, steps):

        qv = QuickView(field[['x', 'y', 'z']].values, mass=(field.mass.values if field.name != 'dm' else None), hsml=(field.hsml.values if field.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=a, t=a, extent = extent)

        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        plt.imsave('./ARTEMIS_Video_test/ARTEMIS_img%d.png'%frame, img_log, cmap=colour, origin='lower')
        frame += 1

        if a != phi+steps:
            pass
        elif a == phi+steps:
            print(frame)
            #plt.clf()
            return a, frame


# In[ ]:


def rotate(phi_min, phi_max, field, boxsize, frame, colour, extent=None):
    
    for p in np.linspace(phi_min, phi_max, phi_max-phi_min): 
    
        qv = QuickView(field[['x', 'y', 'z']].values, mass=(field.mass.values if field.name != 'dm' else None), hsml=(field.hsml.values if field.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=p, t=0, extent = extent)

        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        plt.imsave('./ARTEMIS_Video_test/ARTEMIS_img%d.png'%frame, img_log, cmap=colour, origin='lower')
        frame += 1
        
        if p != phi_max:
            pass
        elif p == phi_max:
            print(frame)
            #plt.clf()
            return phi_max, frame


# In[ ]:


def zoom(max_zoom, magnification, seconds, field, boxsize, frame, colour, phi=0, min_zoom = 2):

    zoom_steps = math.ceil(seconds*25)
    zoom_start = max_zoom
    zoom_stop = ((max_zoom-min_zoom)/magnification)+min_zoom
    zoom_mag = np.linspace(zoom_start, zoom_stop, zoom_steps+1)
    print(zoom_mag)

    for m in zoom_mag: 

        if m == zoom_stop:
            print(frame)
            #plt.clf()
            return extent, frame
        elif m != zoom_stop:
            pass

        extent = [-m, m, -m, m]
        qv = QuickView(field[['x', 'y', 'z']].values, mass=(field.mass.values if field.name != 'dm' else None), hsml=(field.hsml.values if field.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent = extent)

        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        plt.imsave('./ARTEMIS_Video_test/ARTEMIS_img%d.png'%frame, img_log, cmap=colour, origin='lower')
        frame += 1
        

# In[ ]:


def change(field1, field2, seconds, boxsize, frame, colour1, colour2, extent=None, phi=0):
    steps = math.ceil(seconds*25)
    u = np.linspace(1, 0, steps+1)

    qv1 = QuickView(field1[['x', 'y', 'z']].values, mass=(field1.mass.values if field1.name != 'dm' else None), hsml=(field1.hsml.values if field1.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
    qv2 = QuickView(field2[['x', 'y', 'z']].values, mass=(field2.mass.values if field2.name != 'dm' else None), hsml=(field2.hsml.values if field2.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
    
    img1 = qv1.get_image()
    ext1 = qv1.get_extent()
    img_log1 = img_norm(img1, boxsize)
    img2 = qv2.get_image()
    ext2 = qv2.get_extent()
    img_log2 = img_norm(img2, boxsize)
    
    for i in range(0, steps+1):
        fig = plt.figure(i, figsize=(10,10), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        alpha1 = u[i]
        alpha2 = 1-u[i]
    
        a=plt.imshow(img_log1, extent=ext1, cmap=colour1, origin='lower', alpha=alpha1, aspect='auto')
        b=plt.imshow(img_log2, extent=ext2, cmap=colour2, origin='lower', alpha=alpha2, aspect='auto')
        plt.savefig('./ARTEMIS_Video_test/ARTEMIS_img%d.png'%frame)
        plt.clf()
        frame += 1
        
        if i != steps:
            pass
        elif i == steps:
            print(frame)
            #plt.clf()
            return frame


# In[ ]:


def fade(field, seconds, boxsize, frame, colour, extent=None, phi=0, fade_in = False, fade_out = False):
    
    steps = math.ceil(seconds*25)
    if fade_in == True:
        u = np.linspace(0, 1, steps+1)
    elif fade_out == True:
        u = np.linspace(1, 0, steps+1)

    qv = QuickView(field[['x', 'y', 'z']].values, mass=(field.mass.values if field.name != 'dm' else None), hsml=(field.hsml.values if field.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, extent=extent, p=phi)
    
    img = qv.get_image()
    ext = qv.get_extent()
    print(ext)
    if fade_in == True:
        max_zoom = ext[1]
    img_log = img_norm(img, boxsize)
    
    for i in range(0, steps+1):
        fig = plt.figure(i, figsize=(10,10), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        #ax.fill(0, 1, 'k')
        alpha = u[i]
        rect=mpatches.Rectangle((-4.3,-4.3), 8.6, 8.6, facecolor='k', alpha=(1-alpha))
        
        b=plt.gca().add_patch(rect)
        a=plt.imshow(img_log, extent=ext, cmap=colour, origin='lower', alpha=alpha, aspect='auto')
        plt.savefig('./ARTEMIS_Video_test/ARTEMIS_img%d.png'%frame)
        plt.clf()
        frame += 1
        
        if i != steps:
            pass
        elif i == steps:
            print(frame)
            #plt.clf()
            if fade_in == True:
                return max_zoom, frame
            else:
                return frame


# In[ ]:


def evolve(fields, boxsize, frame, colour, extent=None, phi=0):
    
    for f in fields:
        qv = QuickView(f[['x', 'y', 'z']].values, mass=(f.mass.values if f.name != 'dm' else None), hsml=(f.hsml.values if f.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
        
        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        for _ in range(2):
            plt.imsave('./ARTEMIS_Video_test/ARTEMIS_img%d.png'%frame, img_log, cmap=colour, origin='lower')
            frame += 1
        
    print(frame)
    #plt.clf()
    return frame



def wait(field, seconds, boxsize, frame, colour, extent=None, phi=0):
    steps = math.ceil(seconds*25)

    qv = QuickView(field[['x', 'y', 'z']].values, mass=(field.mass.values if field.name != 'dm' else None), hsml=(field.hsml.values if field.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)

    img = qv.get_image()
    ext = qv.get_extent()
    img_log = img_norm(img, boxsize)
    print(img_log.min(), img_log.max())
    print(ext)

    for _ in range(steps+1):
        plt.imsave('./ARTEMIS_Video_test/ARTEMIS_img%d.png'%frame, img_log, vmin=img_log.max()/5, vmax=img_log.max(), cmap=colour, origin='lower')
        frame += 1

    print(frame)
    return frame



# In[ ]:

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    boxsize = 25
    size = 1.5
    min_zoom = 0
    extent = [-size, size, -size, size]

    dm_999 = read_data_ARTEMIS(999, 'dm', 1)
    gas_999 = read_data_ARTEMIS(999, 'gas', 0)
    stars_999 = limit(0.01, read_data_ARTEMIS(999, 'stars', 4), 'hsml')

    print(dm_999)
    print(gas_999)
    print(stars_999)

    #Datasets
    #gas_10 = read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 0)
    #gas_evol = (gas_0,
                #read_data(read_snap(snaps[1][0], snaps[1][1], snaps[1][2]), 0), 
                #read_data(read_snap(snaps[2][0], snaps[2][1], snaps[2][2]), 0),
                #read_data(read_snap(snaps[3][0], snaps[3][1], snaps[3][2]), 0),
                #read_data(read_snap(snaps[4][0], snaps[4][1], snaps[4][2]), 0),
                #read_data(read_snap(snaps[5][0], snaps[5][1], snaps[5][2]), 0),
                #read_data(read_snap(snaps[6][0], snaps[6][1], snaps[6][2]), 0),
                #read_data(read_snap(snaps[7][0], snaps[7][1], snaps[7][2]), 0),
                #read_data(read_snap(snaps[8][0], snaps[8][1], snaps[8][2]), 0),
                #read_data(read_snap(snaps[9][0], snaps[9][1], snaps[9][2]), 0),
                #gas_10)
    #stars_10 = limit(.01, read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 4), 'hsml')
    #stars_28 = limit(.01, read_data(read_snap(snaps[28][0], snaps[28][1], snaps[28][2]), 4), 'hsml')
    #star_evol = (stars_10,
                 #limit(.01, read_data(read_snap(snaps[11][0], snaps[11][1], snaps[11][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[12][0], snaps[12][1], snaps[12][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[13][0], snaps[13][1], snaps[13][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[14][0], snaps[14][1], snaps[14][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[15][0], snaps[15][1], snaps[15][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[16][0], snaps[16][1], snaps[16][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[17][0], snaps[17][1], snaps[17][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[18][0], snaps[18][1], snaps[18][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[20][0], snaps[20][1], snaps[20][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[21][0], snaps[21][1], snaps[21][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[22][0], snaps[22][1], snaps[22][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[24][0], snaps[24][1], snaps[24][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[25][0], snaps[25][1], snaps[25][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[26][0], snaps[26][1], snaps[26][2]), 4), 'hsml'),
                 #limit(.01, read_data(read_snap(snaps[27][0], snaps[27][1], snaps[27][2]), 4), 'hsml'),
                 #stars_28)

    frame = wait(dm_999, 2, boxsize, 0, 'viridis', extent=extent)
    frame = change(dm_999, gas_999, 2, boxsize, frame, 'viridis', 'plasma', extent=extent)
    frame = wait(gas_999, 2, boxsize, frame, 'plasma', extent=extent)
    frame = change(gas_999, stars_999, 2, boxsize, frame, 'plasma', 'bone', extent=extent)
    frame = wait(stars_999, 2, boxsize, frame, 'bone', extent=extent)

    #Start with DM fade in
    #max_zoom, frame = fade(dm_0, 50, boxsize, 0, 'viridis', fade_in = True)
    #print('Done with fade in')
    #DM zoom x2.5
    #extent, frame = zoom(max_zoom, 2.5, 125, dm_0, boxsize, frame, 'viridis')
    #print('Done with zoom in x2.5')
    #Rotate DM by pi
    #phi, frame = rotate(0, 130, dm_0, boxsize, frame, 'viridis', extent)
    #print('Done with rotate by pi')
    #Change from DM to gas
    #frame = change(dm_0, gas_0, 50, boxsize, frame, 'viridis', 'plasma', extent, phi)
    #print('Done with change to gas')
    #Evolve gas through z
    #frame = evolve(gas_evol, boxsize, frame, 'plasma', extent, phi)
    #print('Done with evolve')
    #Rotate gas by pi/2
    #phi, frame = rotate(130, 230, gas_10, boxsize, frame, 'plasma', extent)
    #print('Done with rotate by pi/2')
    #Gas zoom x2
    #extent, frame = zoom(extent[1], 2, 125, gas_10, boxsize, frame, 'plasma', phi)
    #print('Done with zoom in x2')
    #Change from gas to stars
    #frame = change(gas_10, stars_10, 50, boxsize, frame, 'plasma', 'bone', extent, phi)
    #print('Done with change to stars')
    #Rotate stars by pi/2
    #phi, frame = rotate(230, 345, stars_10, boxsize, frame, 'bone', extent)
    #print('Done with rotate by pi/2')
    #Stars zoom x(-4)
    #extent, frame = zoom(extent[1], 0.25, 125, stars_10, boxsize, frame, 'bone', phi)
    #print('Done with zoom out x4')
    #Evolve star through z
    #frame = evolve(star_evol, boxsize, frame, 'bone', extent, phi)
    #print('Done with star evolve')
    #Fade out on stars
    #frame = fade(stars_28, 50, boxsize, frame, 'bone', extent, phi, fade_out = True)
    #print('Done with fade out')
    print(frame)


# In[ ]:




