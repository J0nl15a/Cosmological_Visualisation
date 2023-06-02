from Cosmological_Visualisation_Test_bay_short_video import *
import pandas as pd
import sys

def trajectory_triple(angle, flight_time, fields, boxsize, frame, colours, extent=None):

    steps = math.ceil(flight_time*25)

    for a in np.linspace(angle, angle+steps, steps):

        qv = QuickView(field[['x', 'y', 'z']].values, mass=(field.mass.values if field.name != 'dm' else None), hsml=(field.hsml.values if field.name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=a, t=a, extent = extent)

        img = qv.get_image()
        img_log = img_norm(img, boxsize)
        plt.imsave('./Video_test1/vid_test1_%d.png'%frame, img_log, cmap=colour, origin='lower')
        frame += 1

        if a != phi+steps:
            pass
        elif a == phi+steps:
            print(frame)
            #plt.clf()
            return a, frame


# In[ ]:


def rotate_triple(phi_min, phi_max, fields, boxsize, frame, colours, extent=None):

    for p in np.linspace(phi_min, phi_max, phi_max-phi_min):

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


def zoom_triple(max_zoom, magnification, seconds, fields, boxsize, frame, colours, phi=0, min_zoom = min_zoom):

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
        plt.imsave('./Video_test1/vid_test1_%d.png'%frame, img_log, cmap=colour, origin='lower')
        frame += 1


def fade_triple(fields, seconds, boxsize, frame, colours, extent=None, phi=0, fade_in = False, fade_out = False):

    steps = math.ceil(seconds*25)
    if fade_in == True:
        u = np.linspace(0, 1, steps+1)
    elif fade_out == True:
        u = np.linspace(1, 0, steps+1)

    qv1 = QuickView(fields[0][['x', 'y', 'z']].values, mass=(fields[0].mass.values if fields[0].name != 'dm' else None), hsml=(fields[0].hsml.values if fields[0].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, extent=extent, p=phi)
    qv2 = QuickView(fields[1][['x', 'y', 'z']].values, mass=(fields[1].mass.values if fields[1].name != 'dm' else None), hsml=(fields[1].hsml.values if fields[1].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, extent=extent, p=phi)

    img1 = qv1.get_image()
    ext1 = qv1.get_extent()
    img2 = qv2.get_image()
    ext2 = qv2.get_extent()
    img_log1 = img_norm(img1, boxsize)
    img_log2 = img_norm(img2, boxsize)
    
    if isinstance(fields[2], pd.DataFrame):
        qv3 = QuickView(fields[2][['x', 'y', 'z']].values, mass=(fields[2].mass.values if fields[2].name != 'dm' else None), hsml=(fields[2].hsml.values if fields[2].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000,extent=extent, p=phi)
        img3 = qv3.get_image()
        ext3 = qv3.get_extent()
        img_log3 = img_norm(img3, boxsize)
    else:
        pass

    if fade_in == True:
        max_zoom = ext1[1]

    for i in range(steps+1):
        fig = plt.figure(1, figsize=(30,10), facecolor = 'xkcd:gunmetal')
        ax1 = plt.Axes(fig, [0., 0., 1., 1.])
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.set_axis_off()
        fig.add_axes(ax1)
        alpha = u[i]
        rect=mpatches.Rectangle((-1,-1), 1, 1, facecolor='white', alpha=(1-alpha))

        #b=plt.gca().add_patch(rect)
        a=plt.imshow(img_log1, extent=ext1, cmap=colours[0], origin='lower', alpha=alpha, aspect='auto')

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_axis_off()
        fig.add_axes(ax2)
        #b=plt.gca().add_patch(rect)
        a=plt.imshow(img_log2, extent=ext2, cmap=colours[1], origin='lower', alpha=alpha, aspect='auto')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_axis_off()
        fig.add_axes(ax3)
        #b=plt.gca().add_patch(rect)
        if isinstance(fields[2], pd.DataFrame):
            a=plt.imshow(img_log3, extent=ext3, cmap=colours[2], origin='lower', alpha=alpha, aspect='auto')
        else:
            a=plt.gca().add_patch(mpatches.Rectangle((0,0), 1, 1, facecolor='k', alpha=alpha))

        plt.tight_layout()
        plt.savefig('./Video_test1/vid_test1_%d.png'%frame)
        plt.clf()
        frame += 1

        if i != steps:
            pass
        elif i == steps:
            print(frame)
            if fade_in == True:
                return max_zoom, frame
            else:
                return frame


def evolve_triple(fields, boxsize, frame, colours, extent=None, phi=0):

    for f in fields:
        qv1 = QuickView(f[0][['x', 'y', 'z']].values, mass=(f[0].mass.values if f[0].name != 'dm' else None), hsml=(f[0].hsml.values if f[0].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
        qv2 = QuickView(f[1][['x', 'y', 'z']].values, mass=(f[1].mass.values if f[1].name != 'dm' else None), hsml=(f[1].hsml.values if f[1].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)

        img1 = qv1.get_image()
        ext1 = qv1.get_extent()
        img_log1 = img_norm(img1, boxsize)
        img2 = qv2.get_image()
        ext2 = qv2.get_extent()
        img_log2 = img_norm(img2, boxsize)

        if isinstance(f[2], pd.DataFrame):
            qv3 = QuickView(f[2][['x', 'y', 'z']].values, mass=(f[2].mass.values if f[2].name != 'dm' else None), hsml=(f[2].hsml.values if f[2].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
        
            img3 = qv3.get_image()
            ext3 = qv3.get_extent()
            img_log3 = img_norm(img3, boxsize)
        else:
            pass

        print(f[0])
        print('DM = ' + str(img1.min()) + ', ' + str(img1.max()))
        print('Gas = ' + str(img2.min()) + ', ' + str(img2.max()))
        if isinstance(f[2], pd.DataFrame):
            print('Stars = ' + str(img3.min()) + ', ' + str(img3.max()))
        else:
            print('Stars = 0, 0')

        for _ in range(2):
            fig = plt.figure(1, figsize=(30,10), facecolor = 'xkcd:gunmetal')
            ax1 = plt.Axes(fig, [0., 0., 1., 1.])
            ax1 = fig.add_subplot(1,3,1)
            ax1.set_axis_off()
            fig.add_axes(ax1)
            c=plt.imshow(img_log1, extent=ext1, cmap=colours[0], origin='lower', aspect='auto')

            ax2 = fig.add_subplot(1,3,2)
            ax2.set_axis_off()
            fig.add_axes(ax2)
            c=plt.imshow(img_log2, extent=ext2, cmap=colours[1], origin='lower', aspect='auto')

            ax3 = fig.add_subplot(1,3,3)
            ax3.set_axis_off()
            fig.add_axes(ax3)
            if isinstance(f[2], pd.DataFrame):
                c=plt.imshow(img_log3, extent=ext3, cmap=colours[2], origin='lower', aspect='auto')
            else:
                c=plt.gca().add_patch(mpatches.Rectangle((0,0), 1, 1, facecolor='k'))
            
            plt.tight_layout()
            
            plt.savefig('./Video_test1/vid_test1_%d.png'%frame)
            plt.clf()
            frame += 1

    print(frame)
    return frame


def wait_triple(fields, seconds, boxsize, frame, colours, extent=None, phi=0):
    steps = math.ceil(seconds*25)

    qv1 = QuickView(fields[0][['x', 'y', 'z']].values, mass=(fields[0].mass.values if fields[0].name != 'dm' else None), hsml=(fields[0].hsml.values if fields[0].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)
    qv2 = QuickView(fields[1][['x', 'y', 'z']].values, mass=(fields[1].mass.values if fields[1].name != 'dm' else None), hsml=(fields[1].hsml.values if fields[1].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)

    img1 = qv1.get_image()
    ext1 = qv1.get_extent()
    img_log1 = img_norm(img1, boxsize)
    img2 = qv2.get_image()
    ext2 = qv2.get_extent()
    img_log2 = img_norm(img2, boxsize)

    if isinstance(fields[2], pd.DataFrame):
        qv3 = QuickView(fields[2][['x', 'y', 'z']].values, mass=(fields[2].mass.values if fields[2].name != 'dm' else None), hsml=(fields[2].hsml.values if fields[2].name != 'dm' else None), r='infinity', plot=False, logscale=False, xsize=1000, ysize=1000, p=phi, extent=extent)

        img3 = qv3.get_image()
        ext3 = qv3.get_extent()
        img_log3 = img_norm(img3, boxsize)
    else:
        pass

    for _ in range(steps+1):
        fig = plt.figure(1, figsize=(30,10), facecolor = 'xkcd:gunmetal')
        ax1 = plt.Axes(fig, [0., 0., 1., 1.])
        ax1 = fig.add_subplot(1,3,1)
        ax1.set_axis_off()
        fig.add_axes(ax1)
        c=plt.imshow(img_log1, extent=ext1, cmap=colours[0], origin='lower', aspect='auto')
                         
        ax2 = fig.add_subplot(1,3,2)
        ax2.set_axis_off()
        fig.add_axes(ax2)
        c=plt.imshow(img_log2, extent=ext2, cmap=colours[1], origin='lower', aspect='auto')

        ax3 = fig.add_subplot(1,3,3)
        ax3.set_axis_off()
        fig.add_axes(ax3)
        if isinstance(fields[2], pd.DataFrame):
            c=plt.imshow(img_log3, extent=ext3, cmap=colours[2], origin='lower', aspect='auto')
        else:
            c=plt.gca().add_patch(mpatches.Rectangle((0,0), 1, 1, facecolor='k'))

        plt.tight_layout()
        plt.savefig('./Video_test1/vid_test1_%d.png'%frame)
        plt.clf()
        frame += 1
                         
    print(frame)
    return frame


boxsize = 12
halo_positions = pd.read_csv('EAGLE_halo_positions.csv', sep = ',')
x = halo_positions.x.values
y = halo_positions.y.values
z = halo_positions.z.values

size = 12
min_zoom = 1

if __name__ == "__main__":

    snaps = [(0, 20, 0), (1, 15, 132), (2, 9, 993), (3, 8, 988), (4, 8, 75), (5, 7, 50), (6, 5, 971), (7, 5, 487), (8, 5, 37), (9, 4, 485), (10, 3, 984), (11, 3, 528), (12, 3, 17), (13, 2, 478), (14, 2, 237), (15, 2, 12), (16, 1, 737), (17, 1, 487), (18, 1, 259), (19, 1, 4), (20, 0, 865), (21, 0, 736), (22, 0, 615), (23, 0, 503), (24, 0, 366), (25, 0, 271), (26, 0, 183), (27, 0, 101), (28, 0, 0)]

    dm_0 = read_data(read_snap(snaps[0][0], snaps[0][1], snaps[0][2]), 1, centre=True)
    dm_28 = read_data(read_snap(snaps[28][0], snaps[28][1], snaps[28][2]), 1, centre=True)
    gas_0 = read_data(read_snap(snaps[0][0], snaps[0][1], snaps[0][2]), 0, centre=True)
    gas_28 = read_data(read_snap(snaps[28][0], snaps[28][1], snaps[28][2]), 0, centre=True)

    
    dm_0_norm = read_data(read_snap(snaps[0][0], snaps[0][1], snaps[0][2]), 1)
    dm_28_norm = read_data(read_snap(snaps[28][0], snaps[28][1], snaps[28][2]), 1)
    gas_0_norm = read_data(read_snap(snaps[0][0], snaps[0][1], snaps[0][2]), 0)
    gas_28_norm = read_data(read_snap(snaps[28][0], snaps[28][1], snaps[28][2]), 0)

    stars_0 = None
    stars_28 = limit(.01, read_data(read_snap(snaps[28][0], snaps[28][1], snaps[28][2]), 4, centre=True), 'hsml')
    stars_28_norm = limit(.01, read_data(read_snap(snaps[28][0], snaps[28][1], snaps[28][2]), 4), 'hsml')

    evol = [(dm_0, gas_0, None),
            (read_data(read_snap(snaps[1][0], snaps[1][1], snaps[1][2]), 1, centre=True), read_data(read_snap(snaps[1][0], snaps[1][1], snaps[1][2]), 0, centre=True), None),
            (read_data(read_snap(snaps[2][0], snaps[2][1], snaps[2][2]), 1, centre=True), read_data(read_snap(snaps[2][0], snaps[2][1], snaps[2][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[2][0], snaps[2][1], snaps[2][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[3][0], snaps[3][1], snaps[3][2]), 1, centre=True), read_data(read_snap(snaps[3][0], snaps[3][1], snaps[3][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[3][0], snaps[3][1], snaps[3][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[4][0], snaps[4][1], snaps[4][2]), 1, centre=True), read_data(read_snap(snaps[4][0], snaps[4][1], snaps[4][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[4][0], snaps[4][1], snaps[4][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[5][0], snaps[5][1], snaps[5][2]), 1, centre=True), read_data(read_snap(snaps[5][0], snaps[5][1], snaps[5][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[5][0], snaps[5][1], snaps[5][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[6][0], snaps[6][1], snaps[6][2]), 1, centre=True), read_data(read_snap(snaps[6][0], snaps[6][1], snaps[6][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[6][0], snaps[6][1], snaps[6][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[7][0], snaps[7][1], snaps[7][2]), 1, centre=True), read_data(read_snap(snaps[7][0], snaps[7][1], snaps[7][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[7][0], snaps[7][1], snaps[7][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[8][0], snaps[8][1], snaps[8][2]), 1, centre=True), read_data(read_snap(snaps[8][0], snaps[8][1], snaps[8][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[8][0], snaps[8][1], snaps[8][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[9][0], snaps[9][1], snaps[9][2]), 1, centre=True), read_data(read_snap(snaps[9][0], snaps[9][1], snaps[9][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[9][0], snaps[9][1], snaps[9][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 1, centre=True), read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[11][0], snaps[11][1], snaps[11][2]), 1, centre=True), read_data(read_snap(snaps[11][0], snaps[11][1], snaps[11][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[11][0], snaps[11][1], snaps[11][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[12][0], snaps[12][1], snaps[12][2]), 1, centre=True), read_data(read_snap(snaps[12][0], snaps[12][1], snaps[12][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[12][0], snaps[12][1], snaps[12][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[13][0], snaps[13][1], snaps[13][2]), 1, centre=True), read_data(read_snap(snaps[13][0], snaps[13][1], snaps[13][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[13][0], snaps[13][1], snaps[13][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[14][0], snaps[14][1], snaps[14][2]), 1, centre=True), read_data(read_snap(snaps[14][0], snaps[14][1], snaps[14][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[14][0], snaps[14][1], snaps[14][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[15][0], snaps[15][1], snaps[15][2]), 1, centre=True), read_data(read_snap(snaps[15][0], snaps[15][1], snaps[15][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[15][0], snaps[15][1], snaps[15][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[16][0], snaps[16][1], snaps[16][2]), 1, centre=True), read_data(read_snap(snaps[16][0], snaps[16][1], snaps[16][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[16][0], snaps[16][1], snaps[16][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[17][0], snaps[17][1], snaps[17][2]), 1, centre=True), read_data(read_snap(snaps[17][0], snaps[17][1], snaps[17][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[17][0], snaps[17][1], snaps[17][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[18][0], snaps[18][1], snaps[18][2]), 1, centre=True), read_data(read_snap(snaps[18][0], snaps[18][1], snaps[18][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[18][0], snaps[18][1], snaps[18][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 1, centre=True), read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[20][0], snaps[20][1], snaps[20][2]), 1, centre=True), read_data(read_snap(snaps[20][0], snaps[20][1], snaps[20][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[20][0], snaps[20][1], snaps[20][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[21][0], snaps[21][1], snaps[21][2]), 1, centre=True), read_data(read_snap(snaps[21][0], snaps[21][1], snaps[21][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[21][0], snaps[21][1], snaps[21][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[22][0], snaps[22][1], snaps[22][2]), 1, centre=True), read_data(read_snap(snaps[22][0], snaps[22][1], snaps[22][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[22][0], snaps[22][1], snaps[22][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 1, centre=True), read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[24][0], snaps[24][1], snaps[24][2]), 1, centre=True), read_data(read_snap(snaps[24][0], snaps[24][1], snaps[24][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[24][0], snaps[24][1], snaps[24][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[25][0], snaps[25][1], snaps[25][2]), 1, centre=True), read_data(read_snap(snaps[25][0], snaps[25][1], snaps[25][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[25][0], snaps[25][1], snaps[25][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[26][0], snaps[26][1], snaps[26][2]), 1, centre=True), read_data(read_snap(snaps[26][0], snaps[26][1], snaps[26][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[26][0], snaps[26][1], snaps[26][2]), 4, centre=True), 'hsml')),
            (read_data(read_snap(snaps[27][0], snaps[27][1], snaps[27][2]), 1, centre=True), read_data(read_snap(snaps[27][0], snaps[27][1], snaps[27][2]), 0, centre=True), limit(.01, read_data(read_snap(snaps[27][0], snaps[27][1], snaps[27][2]), 4, centre=True), 'hsml')),
            (dm_28, gas_28, stars_28)]

    evol_norm = [(dm_0_norm, gas_0_norm, None),
                 (read_data(read_snap(snaps[1][0], snaps[1][1], snaps[1][2]), 1), read_data(read_snap(snaps[1][0], snaps[1][1], snaps[1][2]), 0), None),
                 (read_data(read_snap(snaps[2][0], snaps[2][1], snaps[2][2]), 1), read_data(read_snap(snaps[2][0], snaps[2][1], snaps[2][2]), 0), limit(.01, read_data(read_snap(snaps[2][0], snaps[2][1], snaps[2][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[3][0], snaps[3][1], snaps[3][2]), 1), read_data(read_snap(snaps[3][0], snaps[3][1], snaps[3][2]), 0), limit(.01, read_data(read_snap(snaps[3][0], snaps[3][1], snaps[3][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[4][0], snaps[4][1], snaps[4][2]), 1), read_data(read_snap(snaps[4][0], snaps[4][1], snaps[4][2]), 0), limit(.01, read_data(read_snap(snaps[4][0], snaps[4][1], snaps[4][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[5][0], snaps[5][1], snaps[5][2]), 1), read_data(read_snap(snaps[5][0], snaps[5][1], snaps[5][2]), 0), limit(.01, read_data(read_snap(snaps[5][0], snaps[5][1], snaps[5][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[6][0], snaps[6][1], snaps[6][2]), 1), read_data(read_snap(snaps[6][0], snaps[6][1], snaps[6][2]), 0), limit(.01, read_data(read_snap(snaps[6][0], snaps[6][1], snaps[6][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[7][0], snaps[7][1], snaps[7][2]), 1), read_data(read_snap(snaps[7][0], snaps[7][1], snaps[7][2]), 0), limit(.01, read_data(read_snap(snaps[7][0], snaps[7][1], snaps[7][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[8][0], snaps[8][1], snaps[8][2]), 1), read_data(read_snap(snaps[8][0], snaps[8][1], snaps[8][2]), 0), limit(.01, read_data(read_snap(snaps[8][0], snaps[8][1], snaps[8][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[9][0], snaps[9][1], snaps[9][2]), 1), read_data(read_snap(snaps[9][0], snaps[9][1], snaps[9][2]), 0), limit(.01, read_data(read_snap(snaps[9][0], snaps[9][1], snaps[9][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 1), read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 0), limit(.01, read_data(read_snap(snaps[10][0], snaps[10][1], snaps[10][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[11][0], snaps[11][1], snaps[11][2]), 1), read_data(read_snap(snaps[11][0], snaps[11][1], snaps[11][2]), 0), limit(.01, read_data(read_snap(snaps[11][0], snaps[11][1], snaps[11][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[12][0], snaps[12][1], snaps[12][2]), 1), read_data(read_snap(snaps[12][0], snaps[12][1], snaps[12][2]), 0), limit(.01, read_data(read_snap(snaps[12][0], snaps[12][1], snaps[12][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[13][0], snaps[13][1], snaps[13][2]), 1), read_data(read_snap(snaps[13][0], snaps[13][1], snaps[13][2]), 0), limit(.01, read_data(read_snap(snaps[13][0], snaps[13][1], snaps[13][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[14][0], snaps[14][1], snaps[14][2]), 1), read_data(read_snap(snaps[14][0], snaps[14][1], snaps[14][2]), 0), limit(.01, read_data(read_snap(snaps[14][0], snaps[14][1], snaps[14][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[15][0], snaps[15][1], snaps[15][2]), 1), read_data(read_snap(snaps[15][0], snaps[15][1], snaps[15][2]), 0), limit(.01, read_data(read_snap(snaps[15][0], snaps[15][1], snaps[15][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[16][0], snaps[16][1], snaps[16][2]), 1), read_data(read_snap(snaps[16][0], snaps[16][1], snaps[16][2]), 0), limit(.01, read_data(read_snap(snaps[16][0], snaps[16][1], snaps[16][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[17][0], snaps[17][1], snaps[17][2]), 1), read_data(read_snap(snaps[17][0], snaps[17][1], snaps[17][2]), 0), limit(.01, read_data(read_snap(snaps[17][0], snaps[17][1], snaps[17][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[18][0], snaps[18][1], snaps[18][2]), 1), read_data(read_snap(snaps[18][0], snaps[18][1], snaps[18][2]), 0), limit(.01, read_data(read_snap(snaps[18][0], snaps[18][1], snaps[18][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 1), read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 0), limit(.01, read_data(read_snap(snaps[19][0], snaps[19][1], snaps[19][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[20][0], snaps[20][1], snaps[20][2]), 1), read_data(read_snap(snaps[20][0], snaps[20][1], snaps[20][2]), 0), limit(.01, read_data(read_snap(snaps[20][0], snaps[20][1], snaps[20][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[21][0], snaps[21][1], snaps[21][2]), 1), read_data(read_snap(snaps[21][0], snaps[21][1], snaps[21][2]), 0), limit(.01, read_data(read_snap(snaps[21][0], snaps[21][1], snaps[21][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[22][0], snaps[22][1], snaps[22][2]), 1), read_data(read_snap(snaps[22][0], snaps[22][1], snaps[22][2]), 0), limit(.01, read_data(read_snap(snaps[22][0], snaps[22][1], snaps[22][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 1), read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 0), limit(.01, read_data(read_snap(snaps[23][0], snaps[23][1], snaps[23][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[24][0], snaps[24][1], snaps[24][2]), 1), read_data(read_snap(snaps[24][0], snaps[24][1], snaps[24][2]), 0), limit(.01, read_data(read_snap(snaps[24][0], snaps[24][1], snaps[24][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[25][0], snaps[25][1], snaps[25][2]), 1), read_data(read_snap(snaps[25][0], snaps[25][1], snaps[25][2]), 0), limit(.01, read_data(read_snap(snaps[25][0], snaps[25][1], snaps[25][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[26][0], snaps[26][1], snaps[26][2]), 1), read_data(read_snap(snaps[26][0], snaps[26][1], snaps[26][2]), 0), limit(.01, read_data(read_snap(snaps[26][0], snaps[26][1], snaps[26][2]), 4), 'hsml')),
                 (read_data(read_snap(snaps[27][0], snaps[27][1], snaps[27][2]), 1), read_data(read_snap(snaps[27][0], snaps[27][1], snaps[27][2]), 0), limit(.01, read_data(read_snap(snaps[27][0], snaps[27][1], snaps[27][2]), 4), 'hsml')),
                 (dm_28_norm, gas_28_norm, stars_28_norm)]


    colours = ['viridis', 'plasma', 'bone']


    #Start with DM fade in
    #max_zoom, frame = fade_triple([dm_0, gas_0, None], 2, boxsize, 0, colours, fade_in = True)
    #print('Done with fade in')
#    frame = wait_triple([dm_0, gas_0, None], 1, boxsize, 0, colours)
    #Evolve DM through z
#    frame = evolve_triple(evol, boxsize, frame, colours)
#    print('Done with DM evolve')
    #sys.exit()
    #print(dm_28)
    #print(gas_28)
    #print(stars_28)
#    frame = wait_triple([dm_28, gas_28, stars_28], 1, boxsize, frame, colours)

    frame = wait_triple([dm_0_norm, gas_0_norm, None], 1, boxsize, 0, colours)
    frame = evolve_triple(evol_norm, boxsize, frame, colours)
    frame = wait_triple([dm_28_norm, gas_28_norm, stars_28_norm], 1, boxsize, frame, colours)

