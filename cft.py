import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from skimage.morphology import flood_fill
from pyctimg import *


mpl.rc('image', cmap = 'image') # set image colormap to 'bone' throghout the code
ansys_wd = 'Ansys_WD' # set location for saving Ansys 'cdb' files
tau_b = 4250; tau_ef = 2750 # bone segmentation threshold, element formation threshold
f_ds = 2 # downsampling factor
sig_blu = 10; s_bk = 15 # standard deviation for blurring, size of blurrking kernel
e_cut = 6800; e_min = 50 # cut_off elastic modulus, minimum elastic modulus
vcc_valHU = [0.51, -1000]; vcc_HUrho = [0.38010, -7.37444]; vcc_duazon = [10225, 1000] # linear conversion of pixel values to HU, bone mineral density, and elastic modulus
r_cts = 60.7 # resolution of CT scan
dire = 'RAW_3D' # directory containing input images
nams = [] # list of image names in the directory
for nam in os.listdir(dire):
	if nam[-3:] == 'npy': # check if file format is 'npy'
		nams.append(nam)
print(nams)
for nam_full in nams: # iterate over files in directory
	t_s = time.time() # start time
	tnj = np.load('{}/{}'.format(dire, nam_full)) # load raw image
	nam = np.copy(nam_full[:8]) # name iteration
	print(nam)
	os.makedirs('CFT/{}'.format(nam)) # create directory for iteration
	image = Image(np.copy(tnj)) # create Image instance from 'tnj' copy
	image.blur_by_Gaussian(sig_blu, s_bk)
	np.save('CFT/{}/blurred'.format(nam), image.value)
	image.segment_by_threshold(tau_b)
	np.save('CFT/{}/segmented(bs)'.format(nam), image.value)
	tnf = image.detect_frame()
	np.save('CFT/{}/frame'.format(nam), tnf)
	mxf_key, vcn_ds, mxf_pad, mxf = find_3d_frame(tnf, f_ds)
	np.save('CFT/{}/frame_3D'.format(nam), mxf)
	np.save('CFT/{}/elemental_space.npy'.format(nam), vcn_ds)
	np.save('CFT/{}/frame_pad'.format(nam), mxf_pad)
	np.save('CFT/{}/frame_key'.format(nam), mxf_key)
	image = Image(tnj) # create Image instance from 'tnj'
	image.adjust_to_frame(mxf_pad, mxf)
	np.save('CFT/{}/adjusted_to_frame'.format(nam), image.value)
	tnk = makekernel(f_ds); tnk = tnk/np.sum(tnk) # create kernel for downsampling
	image.downsample(tnk, f_ds, vcn_ds) 
	np.save('CFT/{}/downsampled'.format(nam), image.value)
	image.segment_by_threshold(tau_ef)
	np.save('CFT/{}/segmented(ef)'.format(nam), image.value)
	tnj_mask = image.value != 0 # create binary mask from image
	tnj_mask = np.uint8(tnj_mask) # update data type of mask
	i, j = find_first_pixel(tnj_mask[0])
	tnj_mask = flood_fill(tnj_mask, (0, i, j), 2, connectivity = 1) # set connected-by-face pixels to 2
	tnj_mask[tnj_mask == 1] = 0 # set pixels with value 1 to 0
	tnj_mask[tnj_mask == 2] = 1 # set pixels with value 2 to 1
	image.value = image.value*tnj_mask # transfer connectivity of mask to image
	np.save('CFT/{}/clean'.format(nam), image.value)
	mesh = Mesh(image.value) # create Mesh instance from image
	mesh.create_mesh()
	mxp_nod, mxn_ele, vce = mesh.convert_mesh(f_ds, r_cts, e_cut, e_min, vcc_valHU, vcc_HUrho, vcc_duazon)
	np.save('CFT/{}/nodal_positions'.format(nam), mxp_nod)
	np.save('CFT/{}/nodes_on_element'.format(nam), mxn_ele)
	np.save('CFT/{}/elastic_modulus'.format(nam), vce)
	da = (2*f_ds+1)*r_cts/1000 # calculate element size
	mesh.write_Ansys_input(f_ds, '{}/{}_cft'.format(ansys_wd, nam))
	t_e = time.time() # stop time
	dt = np.int16(round(t_e-t_s)); print(dt, '\n') # calculate and print runtime
	f = open('CFT/{}/run_time.txt'.format(nam), 'w'); f.write(str(dt)); f.close() # save runtime