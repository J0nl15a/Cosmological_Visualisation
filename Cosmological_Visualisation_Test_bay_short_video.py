
#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.patches as mpatches
#matplotlib.use('Agg')
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

# In[2]:

boxsize = 12
halo_positions = pd.read_csv('EAGLE_halo_positions.csv', sep = ',')
x = halo_positions.x.values
y = halo_positions.y.values
z = halo_positions.z.values
print(x)
print(y)
print(z)

# In[6]:


#h = 0.6777
size = 10
#extent = [-size, size, -size, size]
min_zoom = 2

# In[7]:

snaps = [(0, 20, 0), (1, 15, 132), (2, 9, 993), (3, 8, 988), (4, 8, 75), (5, 7, 50), (6, 5, 971), (7, 5, 487), (8, 5, 37), (9, 4, 485), (10, 3, 984), (11, 3, 528), (12, 3, 17), (13, 2, 478), (14, 2, 237), (15, 2, 12), (16, 1, 737), (17, 1, 487), (18, 1, 259), (19, 1, 4), (20, 0, 865), (21, 0, 736), (22, 0, 615), (23, 0, 503), (24, 0, 366), (25, 0, 271), (26, 0, 183), (27, 0, 101), (28, 0, 0)]

def read_snap(k, z_sh, p, x=x, y=y, z=z, size=size):
    s = read_eagle.EagleSnapshot("/cosma7/data/Eagle/DataRelease/L0012N0188/PE/REF_COSMA5/data/snapshot_" + f"{k:03d}" + "_z" + f"{z_sh:03d}" + "p" + f"{p:03d}" + "/snap_" + f"{k:03d}" + "_z" + f"{z_sh:03d}" + "p" + f"{p:03d}" + ".0.hdf5")
    s.select_region(x - size, x + size, y - size, y + size, z - size, z + size)
    return s

def read_data(snap, state):
    pos = snap.read_dataset(state, "Coordinates")
    field = pd.DataFrame(pos, columns=['x','y','z'])
    if state != 1:
        field['mass'] = snap.read_dataset(state, "Mass") * 1e10
        field['hsml'] = snap.read_dataset(state, "SmoothingLength")
        field.name = str(field)
    else:
        field.name = 'dm'
    
    return field

#dm = read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 1)

#snap23 = read_eagle.EagleSnapshot("/cosma7/data/Eagle/DataRelease/L0012N0188/PE/REF_COSMA5/data/snapshot_023_z000p503/snap_023_z000p503.0.hdf5")
#snap23.select_region(x - size, x + size, y - size, y + size, z - size, z + size)

# In[27]:

#pos_gas_28 = snap28.read_dataset(0, "Coordinates") / h
#gas_28 = pd.DataFrame(pos_gas_28, columns=['x','y','z'])
#gas_28['mass'] = snap28.read_dataset(0, "Mass") / h * 1e10
#gas_28['hsml'] = snap28.read_dataset(0, "SmoothingLength") / h
#gas_28.name = 'gas_28'

# In[29]:

def limit(lim, field, entry):
    max_val = pd.Series(lim, index=range(len(field)))
    field[entry] = field.mask(field > lim, max_val, axis=0)
    return field

#stars_28 = limit(0.01, stars_28, 'hsml')
#print(stars_28)


# In[ ]:


def img_norm(img, boxsize):
    img_norm = img / np.power(boxsize / np.shape(img)[0], 2)
    #print(img_norm)
    #max_5 = np.max(img_norm)/5
    #img_norm[np.where(img_norm < max_5)] = max_5
    img_norm[np.where(img_norm < 1e-8)] = 1e-8
    img_log = np.log10(img_norm)
    return img_log


# In[ ]:


def rotate(phi_min, phi_max, field, boxsize, frame, colour, extent):
    
    for p in range(phi_min, phi_max+1): 
    
        qv = QuickView(field[['x', 'y', 'z']].values, mass=(field.mass.values if field.name != 'dm' else None), hsml=(field.hsml.values if field.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=p, t=0, extent = extent)

        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        plt.imsave('./Video_test1/vid_test1_%d.png'%frame, img_log, cmap=colour, origin='lower')
        frame += 1
        
        if p != phi_max:
            pass
        elif p == phi_max:
            print(frame)
            #plt.clf()
            return phi_max, frame


# In[ ]:


def zoom(max_zoom, magnification, zoom_steps, field, boxsize, frame, colour, phi=0, min_zoom = min_zoom):
    
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
        plt.imsave('./Video_test1/vid_test1_%d.png'%frame, img_log, cmap=colour, origin='lower')
        frame += 1
        

# In[ ]:


def change(field1, field2, steps, boxsize, frame, colour1, colour2, extent, phi=0):
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
        plt.savefig('./Video_test1/vid_test1_%d.png'%frame)
        plt.clf()
        frame += 1
        
        if i != steps:
            pass
        elif i == steps:
            print(frame)
            #plt.clf()
            return frame


# In[ ]:


def fade(field, steps, boxsize, frame, colour, extent=None, phi=0, fade_in = False, fade_out = False):
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
        plt.savefig('./Video_test1/vid_test1_%d.png'%frame)
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


def evolve(fields, boxsize, frame, colour, extent, phi=0):
    
    for f in fields:
        print(f)
        print(frame)
        qv = QuickView(f[['x', 'y', 'z']].values, mass=(f.mass.values if f.name != 'dm' else None), hsml=(f.hsml.values if f.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
        
        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        for _ in range(3):
            plt.imsave('./Video_test1/vid_test1_%d.png'%frame, img_log, cmap=colour, origin='lower')
            frame += 1
        
    print(frame)
    #plt.clf()
    return frame


# In[ ]:

#Datasets
dm_0 = read_data(read_snap(snaps[0][0], snaps[0][1], snaps[0][2]), 1)
print(dm_0)
gas_0 = read_data(read_snap(snaps[0][0], snaps[0][1], snaps[0][2]), 0)
gas_10 = read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 0)
gas_evol = (gas_0,
            read_data(read_snap(snaps[1][0], snaps[1][1], snaps[1][2]), 0), 
            read_data(read_snap(snaps[2][0], snaps[2][1], snaps[2][2]), 0),
            read_data(read_snap(snaps[3][0], snaps[3][1], snaps[3][2]), 0),
            read_data(read_snap(snaps[4][0], snaps[4][1], snaps[4][2]), 0),
            read_data(read_snap(snaps[5][0], snaps[5][1], snaps[5][2]), 0),
            read_data(read_snap(snaps[6][0], snaps[6][1], snaps[6][2]), 0),
            read_data(read_snap(snaps[7][0], snaps[7][1], snaps[7][2]), 0),
            read_data(read_snap(snaps[8][0], snaps[8][1], snaps[8][2]), 0),
            read_data(read_snap(snaps[9][0], snaps[9][1], snaps[9][2]), 0),
            gas_10)
stars_10 = limit(.01, read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 4), 'hsml')
stars_28 = limit(.01, read_data(read_snap(snaps[28][0], snaps[28][1], snaps[28][2]), 4), 'hsml')
star_evol = (stars_10,
             limit(.01, read_data(read_snap(snaps[11][0], snaps[11][1], snaps[11][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[12][0], snaps[12][1], snaps[12][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[13][0], snaps[13][1], snaps[13][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[14][0], snaps[14][1], snaps[14][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[15][0], snaps[15][1], snaps[15][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[16][0], snaps[16][1], snaps[16][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[17][0], snaps[17][1], snaps[17][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[18][0], snaps[18][1], snaps[18][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[20][0], snaps[20][1], snaps[20][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[21][0], snaps[21][1], snaps[21][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[22][0], snaps[22][1], snaps[22][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[24][0], snaps[24][1], snaps[24][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[25][0], snaps[25][1], snaps[25][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[26][0], snaps[26][1], snaps[26][2]), 4), 'hsml'),
             limit(.01, read_data(read_snap(snaps[27][0], snaps[27][1], snaps[27][2]), 4), 'hsml'),
             stars_28)

#star = limit(.01, read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 4), 'hsml')
#for j in stars_28['hsml']:
    #if j != 0.01:
        #print(j)
#sys.exit()

#Start with DM fade in
max_zoom, frame = fade(dm_0, 50, boxsize, 0, 'viridis', fade_in = True)
print('Done with fade in')
#DM zoom x2.5
extent, frame = zoom(max_zoom, 2.5, 125, dm_0, boxsize, frame, 'viridis')
print('Done with zoom in x2.5')
#Rotate DM by pi
phi, frame = rotate(0, 130, dm_0, boxsize, frame, 'viridis', extent)
print('Done with rotate by pi')
#Change from DM to gas
frame = change(dm_0, gas_0, 50, boxsize, frame, 'viridis', 'plasma', extent, phi)
print('Done with change to gas')
#Evolve gas through z
frame = evolve(gas_evol, boxsize, frame, 'plasma', extent, phi)
print('Done with evolve')
#Rotate gas by pi/2
phi, frame = rotate(130, 230, gas_10, boxsize, frame, 'plasma', extent)
print('Done with rotate by pi/2')
#Gas zoom x2
extent, frame = zoom(extent[1], 2, 125, gas_10, boxsize, frame, 'plasma', phi)
print('Done with zoom in x2')
#Change from gas to stars
frame = change(gas_10, stars_10, 50, boxsize, frame, 'plasma', 'bone', extent, phi)
print('Done with change to stars')
#Rotate stars by pi/2
phi, frame = rotate(230, 345, stars_10, boxsize, frame, 'bone', extent)
print('Done with rotate by pi/2')
#Stars zoom x(-4)
extent, frame = zoom(extent[1], 0.25, 125, stars_10, boxsize, frame, 'bone', phi)
print('Done with zoom out x4')
#Evolve star through z
frame = evolve(star_evol, boxsize, frame, 'bone', extent, phi)
print('Done with star evolve')
#Fade out on stars
frame = fade(stars_28, 50, boxsize, frame, 'bone', extent, phi, fade_out = True)
print('Done with fade out')
print(frame)


# In[ ]:




