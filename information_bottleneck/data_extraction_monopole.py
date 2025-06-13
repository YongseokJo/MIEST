import numpy as np

for field in ['Mtot', 'T', 'Mgas','Mstar','HI','ne','Vcdm','Z']:
    # TNG
    fmaps = \
            "/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Maps_{}_IllustrisTNG_LH_z=0.00.npy".format(field)
    maps     = np.load(fmaps)
    monopole = maps.sum(axis=1).sum(axis=1)
    print(monopole.shape)
    np.save("../data/wph_IllustrisTNG_{}_monopole".format(field),monopole)

    #SIMBA
    fmaps = \
            "/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Maps_{}_SIMBA_LH_z=0.00.npy".format(field)
    maps     = np.load(fmaps)
    monopole = maps.sum(axis=1).sum(axis=1)
    print(monopole.shape)
    np.save("../data/wph_SIMBA_{}_monopole".format(field),monopole)

    #ASTRID
    fmaps = \
            "/mnt/home/fvillaescusa/CAMELS/PUBLIC_RELEASE/CMD/2D_maps/data/Maps_{}_Astrid_LH_z=0.00.npy".format(field)
    maps     = np.load(fmaps)
    monopole = maps.sum(axis=1).sum(axis=1)
    print(monopole.shape)
    np.save("../data/wph_Astrid_{}_monopole".format(field),monopole)
