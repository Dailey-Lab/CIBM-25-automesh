import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from pyctimg import *


mpl.rc('image', cmap = 'bone') # set image colormap to 'bone' throghout the code
tau_b = 4250; tau_zero = 1 # bone segmentation threshold, zero curvature threshold
s_q = 5 # size of neighborhood
s_sk = 10 # size of structuring kernel
sig_blu = 10; s_bk = 15 # standard deviation for blurring, size of blurrking kernel
vcw = [1, 0.1, 0.2] # weight vector
f_rep = s_q # kick-out factor
s_dk = 3; n_cp = 72 # size of derivative kernel, number of contour points
n_ite = 1 # number of iterations for dilation and erosion
l_w = 350 # half-length of image in visualizations
mxw = np.reshape(vcw, (-1, 1)) # reshape weight vector to a column matrix
mxk_de = np.ones((s_sk, s_sk), dtype = np.int16) # define kernel for dilation and erosion

dire = 'RAW_3D' # directory containing input images
nams = [] # list of image names in the directory
for nam in os.listdir(dire):
	if nam[-3:] == 'npy': # check if file format is 'npy'
		nams.append(nam)
print(nams)


for nam in nams: # iterate over files in directory
	t_s = time.time() # start time
	tnj = np.load('{}/{}'.format(dire, nam)) # load raw image
	tnj_raw = np.copy(tnj)
	nam = nam[:8] # name iteration
	print(nam)
	os.makedirs('SNAKE/{}'.format(nam)) # create directory for iteration
	# Initialization
	mxj_ini = np.copy(tnj[0]) # take first axial image
	tnj_ini = np.array([mxj_ini]) # reshape to 3rd order tensor
	image = Image(tnj_ini) # create Image instance from 'tnj_ini'
	image.blur_by_Gaussian(sig_blu, s_bk)
	image.segment_by_threshold(tau_b) 
	image.detect_frame() 
	# depending on the orientation of bone, only one or both of these two functions work properly:
	#tnc = image.detect_contour_simple_col() 
	tnc = image.detect_contour_simple_row()
	tnc = np.array(tnc) # convert to numpy array
	contour = Contour(tnc) # create Contour instance
	contour.convert_to_polar()
	contour.regulerize_angularly(n_cp)
	contour.convert_to_cartesian()
	mxc = contour.contour[0] # extract contour
	vcc = np.int16(np.round(np.mean(mxc, axis = 0))) # find center of contour by averaging coordiantes
	fig0, axe0 = plt.subplots(1, figsize = (8.68, 8.69), dpi = 100)
	axe0.imshow(mxj_ini) # show first axial image
	axe0.plot(mxc[:, 1], mxc[:, 0], linewidth = 0, marker = '.')
	axe0.plot(vcc[1], vcc[0], marker = 's', color = 'r', markersize = 2)
	axe0.set_xlabel(''); axe0.set_ylabel('')
	axe0.set_xticks([]); axe0.set_yticks([])
	axe0.set_xlim([vcc[1]-l_w, vcc[1]+l_w])
	axe0.set_ylim([vcc[0]-l_w, vcc[0]+l_w])
	plt.tight_layout()
	plt.savefig('SNAKE/{}/0.png'.format(nam))
	plt.close()
	# Contour
	image = Image(tnj) # create Image instance from 'tnj'
	image.segment_by_threshold(tau_b)
	tnj_seg = np.copy(image.value) # store segmented images
	image.blur_by_Gaussian(sig_blu, s_bk)
	tnj_blu = np.copy(image.value) # store blurred image
	image.calc_derivative(s_dk)
	tnj_der = np.copy(image.value) # store derivative of image
	image.erode(mxk_de, n_ite)
	image.dilate(mxk_de, n_ite)
	tnc = np.zeros((image.shape[0], n_cp, 2), dtype = np.int16) # initialize 3rd order tensor for contour
	tnn = np.zeros((image.shape[0], n_cp, 2), dtype = float) # initialize 3rd order tensor for normal vectors to points of contour
	tnr = np.zeros((image.shape[0], n_cp, 2), dtype = float) # initialize 3rd order tensor for radial vectors from points of contour
	tnc[0] = mxc # set first contour to 'mxc'
	mxt = np.zeros((image.shape[0], n_cp), dtype = bool) # initialize matrix for curvature type at points of contour
	for k in range(1, image.shape[0]): # iterate over number of images excluding first one
		print(k)
		mxj = image.value[k] # take current axial image
		mxc_g = np.copy(mxc) # take previous axial contour
		mxn = np.zeros((n_cp, 2), dtype = float) # initialize matrix for normal vector for previous contour
		mxr = np.zeros((n_cp, 2), dtype = float) # initialize matrix for radial vector for previous contour
		vct_cur = np.zeros((n_cp, ), dtype = bool) # initialize vector for curvature type for previous contour
		fig, axe = plt.subplots(2, 2, figsize = (9, 9), dpi = 100, num = k)
		for i in range(n_cp): # iterate over number of contour points
			t_cur = 0 # set curvature type to zero
			vcc = np.int16(np.round(np.mean(mxc, axis = 0))) # update center of contour
			if i == n_cp-1: # check if iteration belongs to last point on contour
				vcp_n = mxc[0] # take next point on contour
				vcp_nn = mxc[1] # take second next point on contour
			elif i == n_cp-2: # check if iteration belongs to second last point on contour
				vcp_n = mxc[i-1] # take next point on contour
				vcp_nn = mxc[0] # take second next point on contour
			else:
				vcp_n = mxc[i+1] # take next point on contour
				vcp_nn = mxc[i+2] # take second next point on contour
			point = Point(mxc[i]) # creat Point instance from point on contour
			point.form_square_neighborhood(s_q)
			mxq = point.neighborhood # store neighborhood

			vcn = calculate_normal_vector(mxc[i-2], mxc[i-1], mxc[i], vcp_n, vcp_nn)
			vcr = mxc[i]-vcc # calculate radial vector from center to point on contour
			s_vcr = np.linalg.norm(vcr) # calculate size of radial vector
			vcr = vcr/s_vcr # replace radial vector with its unit vector
			s_vcn = np.linalg.norm(vcn) # calculate normal vector
			if s_vcn > tau_zero: # check if normal vector is not zero
				vcn = vcn/s_vcn # replace normal vector with its unit vector
				phi = calculate_angle(vcn, vcr) # calculate angle between normal and radial vector
				if phi < np.pi/2: # check if angle is smaller than 90 degree
					mxq = np.int16(np.round(mxq+f_rep*vcn)) # shift neighborhood outward along normal vector
					t_cur = True # update curvature type
			
			# Regular Point Spacing
			mxe_the = np.zeros((mxq.shape[0], 1), dtype = np.float32) # initialize energy matrix for angular spacing
			for j in range(np.shape(mxq)[0]): # iterate over number of points in neighborhood
				vcu = mxq[j]-vcc; vcu = vcu/np.linalg.norm(vcu) # calculate unit radial vector to each point in neighborhood
				vcv = mxc[i-1]-vcc; vcv = vcv/np.linalg.norm(vcv) # calculate unit radial vector to previous point of contour
				the = calculate_angle(vcu, vcv) # calculate anglular space between points in neighborhood and previous point on contour
				mxe_the[j, 0] = np.abs(the) # update 'mxe_the'
			mxe_the = np.abs((360/n_cp)*np.pi/180-mxe_the) # subtract 'mxe_the' from ideal point spacing
			mxe_the = mxe_the/np.max(mxe_the) # normalize 'mxe_the' with respect to its maximum
			# Edge Attraction
			vcd = tnj_der[k, mxq[:, 0], mxq[:, 1]] # take image derivatives in neighborhood
			mxd = vcd.reshape(-1, 1) # reshape into column matrix
			d_max = max(mxd); d_min = min(mxd) # find minimum and maximum of 'mxd'
			if d_max-d_min > 5: # check if mxd is nonuniform
				mxe_d = (d_min-mxd)/(d_max-d_min) # normalize 'mxe_d'
			else:
				mxe_d = (d_min-mxd)/5 # normalize 'mxe_d'
			# Smoothness
			mxn_cur = calculate_normal_vector(mxc[i-2], mxc[i-1], mxq, vcp_n, vcp_nn) # calculate normal vector to points in neighborhood
			vce_kap = np.sqrt(np.sum(mxn_cur**2, axis = 1)) # calculate cuvature magnitudes of points in neighborhood as curvature energy
			vce_kap = vce_kap/max(vce_kap) # normalize curvature energy with respect to its maximum
			mxe_kap = vce_kap.reshape(-1, 1)  # reshape 'vce_kap'
			mxe = np.concatenate((mxe_the, mxe_kap, mxe_d), axis = 1) # assemble energy matrices into one energy matrix
			# Total
			mxe = mxe@mxw # sum energy for each point in neighborhood after weighing
			j_star = np.argmin(mxe) # get the argument of minimum energy
			mxc[i] = mxq[j_star] # update contour with point in neighborhood having minimum energy
			mxn[i] = vcn # update 'mxn' with normal vector to contour point before relocation
			mxr[i] = vcr # update 'mxr' with radial vector at contour point before relocation
			vct_cur[i] = t_cur # update 'vct_cur' curvature type
			axe[1, 1].plot(mxq[:, 1], mxq[:, 0], color = 'b', linewidth = 0, marker = '.', markersize = 0.3)
			if (mxc[i] == mxc[i-1]).all():
				print('Points collapsed at {}!'.format(mxc[i])) # inform if points collapse onto each other
		tnc[k] = mxc # update 'tnc'
		tnn[k] = mxn # update 'tnn'
		tnr[k] = mxr # update 'tnr'
		mxt[k] = vct_cur # update 'mxt'
		axe[0, 0].imshow(tnj_raw[k]) # show raw image
		axe[0, 1].imshow(tnj_seg[k]) # show segmented image
		axe[1, 0].imshow(tnj_blu[k]) # show blurred image
		axe[1, 1].imshow(tnj_der[k]) # show image derivative
		axe[0, 0].plot(tnc[k, :, 1], tnc[k, :, 0], color = 'c', linewidth = 2) # plot current contour
		axe[0, 0].plot(tnc[k-1, :, 1], tnc[k-1, :, 0], linewidth = 0, marker = 's', markersize = 2, color = 'r') # plot previous contour
		for ii in range(2):
			for jj in range(2):
				axe[ii, jj].set_xlim([vcc[1]-l_w, vcc[1]+l_w])
				axe[ii, jj].set_ylim([vcc[0]-l_w, vcc[0]+l_w])
				axe[ii, jj].plot(vcc[1], vcc[0], marker = 'o', color = 'orange', markersize = 3)
				axe[ii, jj].plot(mxc_g[:, 1], mxc_g[:, 0], linewidth = 0, marker = 's', markersize = 2, color = 'r')
				axe[ii, jj].set_xticks([])
				axe[ii, jj].set_yticks([])
		plt.tight_layout()
		plt.savefig('SNAKE/{}/{}.png'.format(nam, k), dpi = 100, bbox_inches = 'tight', pad_inches = 0)
		plt.close()
	np.save('SNAKE/{}/tnc.npy'.format(nam), tnc)
	np.save('SNAKE/{}/tnn.npy'.format(nam), tnn)
	np.save('SNAKE/{}/tnr.npy'.format(nam), tnr)
	np.save('SNAKE/{}/mxt.npy'.format(nam), mxt)
	t_e = time.time() # stop time
	dt = np.int16(round(t_e-t_s)) # calculate runtime
	f = open('SNAKE/{}/run_time.txt'.format(nam), 'w'); f.write(str(dt)); f.close() # save runtime
