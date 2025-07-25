import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from skimage.morphology import flood_fill
from pyctimg import *


mpl.rc('image', cmap = 'image') # set image colormap to 'bone' throghout the code
ansys_wd = 'Ansys_WD' # set location for saving Ansys 'cdb' files
abaq_wd = 'Abaqus_WD' # set location for saving Abaqus 'inp' file
tau_ef = 2750 # element formation threshold
f_ds = 2 # downsampling factor
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
	tnc = np.load('SNAKE/{}/tnc.npy'.format(nam)) # loading contour
	print(nam)
	os.makedirs('SRT/{}/'.format(nam))
	image = Image(tnj) # creating object instance from raw image
	contour = Contour(tnc) # creating object instance from contour
	tnf = contour.detect_frame()
	np.save('SRT/{}/frame'.format(nam), tnf)
	mxf_key, vcn_ds, mxf_pad, mxf = find_3d_frame(tnf, f_ds)
	np.save('SRT/{}/elemental_space.npy'.format(nam), vcn_ds)
	np.save('SRT/{}/frame_3D'.format(nam), mxf)
	np.save('SRT/{}/frame_pad'.format(nam), mxf_pad)
	np.save('SRT/{}/frame_key'.format(nam), mxf_key)
	contour.create_full_mask(image.shape[1:])
	bone_mask = Image(contour.mask) # create object instance from mask
	image.adjust_to_frame(mxf_pad, mxf)
	bone_mask.adjust_to_frame(mxf_pad, mxf)
	np.save('SRT/{}/adjusted_to_frame'.format(nam), image.value)
	tnk = makekernel(f_ds); tnk = tnk/np.sum(tnk) # create kernel for downsampling
	image.downsample_w_mask(tnk, f_ds, vcn_ds, bone_mask.value)
	np.save('SRT/{}/downsampled'.format(nam), image.value)
	image.segment_by_threshold(tau_ef)
	np.save('SRT/{}/segmented(ef)'.format(nam), image.value)
	tnj_mask = image.value != 0
	tnj_mask = np.uint8(tnj_mask)
	i, j = find_first_pixel(tnj_mask[0])
	tnj_mask = flood_fill(tnj_mask, (0, i, j), 2, connectivity = 1) # set connected-by-face pixels to 2
	tnj_mask[tnj_mask == 1] = 0 # set pixels with value 1 to 0
	tnj_mask[tnj_mask == 2] = 1 # set pixels with value 2 to 1
	image.value = image.value*tnj_mask # transfer connectivity of mask to image
	np.save('SRT/{}/clean'.format(nam), image.value)
	mesh = Mesh(image.value) # create Mesh instance from image
	mesh.create_mesh()
	mxp_nod, mxn_ele, vce = mesh.convert_mesh(f_ds, r_cts, e_cut, e_min, vcc_valHU, vcc_HUrho, vcc_duazon)
	np.save('SRT/{}/nodal_positions'.format(nam), mxp_nod)
	np.save('SRT/{}/nodes_on_element'.format(nam), mxn_ele)
	np.save('SRT/{}/elastic_modulus'.format(nam), vce)
	da = (2*f_ds+1)*r_cts/1000 # calculate element size
	mesh.write_Ansys_input(f_ds, '{}/{}_srt'.format(ansys_wd, nam))
	mesh.write_Abaqus_input(f_ds, '{}/{}_srt'.format(abaq_wd, nam), da)
	t_e = time.time() # stop time
	dt = np.int16(round(t_e-t_s)); print(dt, '\n') # calculate and print runtime
	f = open('SRT/{}/run_time.txt'.format(nam), 'w'); f.write(str(dt)); f.close() # save runtime