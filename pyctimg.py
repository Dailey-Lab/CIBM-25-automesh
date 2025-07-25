import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
	
class Image():
	
	def __init__(self, tnj):
		if not isinstance(tnj, np.ndarray):
			raise ValueError('Input boundary must be a numpy array!')
		if len(tnj.shape) != 3:
			raise ValueError('Input image must be 3 dimensional!')
		self.value = tnj
		self.shape = np.array(self.value.shape, dtype = np.int16)

	def blur_by_Gaussian(self, sig_blu, s_bk): 
		# blurs each axial image individually using the input sigma and kernel size
		for k in range(self.shape[0]): # iterate over the number of axial images 
			self.value[k] = np.round(cv.GaussianBlur(self.value[k], (2*s_bk+1, 2*s_bk+1), sig_blu))
	
	def segment_by_threshold(self, tau): 
		# segments the entire image by the input threshold
		self.value[self.value < tau] = 0

	def detect_frame(self): 
		# detects a bounding box for bone at each axial image
		self.frame = np.zeros((self.shape[0], 2, 2), dtype = np.int16) # initialize as a 3D array with the same length as image
		for k in range(self.shape[0]): # iterate over the number of axial images
			# Axis 0
			vci_rowsum = np.sum(self.value[k], axis = 1) # sum values of axial image at row
			vci = [0, self.shape[1]-1] # initialize axial frame
			while(vci_rowsum[vci[1]] == 0): # iterate backward from last row until 'vci_rowsum' is nonzero
				vci[1] = vci[1]-1 # update upper value of frame
				if vci[1] < 0:
					raise ValueError('Lower edge of frame could not be detected along axis 0!')
			while(vci_rowsum[vci[0]] == 0): # iterate foreward from first row until 'vci_rowsum' is nonzero
				vci[0] = vci[0]+1 # update lower value of frame
				if vci[0] > self.shape[1]-1:
					raise ValueError('Upper edge of frame could not be detected along axis 0!')
			self.frame[k, 0] = np.copy(vci)
			# Axis 1
			vcj_colsum = np.sum(self.value[k], axis = 0) # sum values of axial image at column
			vcj = [0, self.shape[2]-1] # initialize axial frame
			while(vcj_colsum[vcj[1]] == 0): # iterate backward from last column until 'vci_colsum' is nonzero
				vcj[1] = vcj[1]-1 # update upper value of frame
				if vcj[1] < 0:
					raise ValueError('Lower edge of frame could not be detected along axis 1!')
			while(vcj_colsum[vcj[0]] == 0): # iterate foreward from first row until 'vci_rowsum' is nonzero
				vcj[0] = vcj[0]+1 # update lower value of frame
				if vcj[0] > self.shape[2]-1:
					raise ValueError('Upper edge of frame could not be detected along axis 1!')
			self.frame[k, 1] = np.copy(vcj)
		return self.frame

	def detect_contour_simple_col(self): 
		# detects contour of bone at each axial image by searching for it at each column
		tnj_bin = np.copy(self.value)
		tnj_bin[tnj_bin != 0] = 1 # create binary mask
		tnj_bin = np.uint8(tnj_bin)
		tnc = [] # initialize contour as a list
		for k in range(self.shape[0]): # iterate over number of images
			vcj = np.arange(self.frame[k, 1, 0], self.frame[k, 1, 1]+1) # create list of columns withing frame
			n_j = np.shape(vcj)[0]
			mxc = [] # initialize contour
			for j in range(n_j):
				vci = tnj_bin[k, :, vcj[j]] # take column of mask
				vc = np.where(vci == 1)[0] # take indices of nonzero pixels in 'vci'
				if len(vc) < 2:
					continue
				mxc.append([vcj[j], vc[0], vc[-1]]) # add point to contour
			mxc = np.array(mxc)
			mxc_t = np.concatenate((mxc[:, 1:2], mxc[:, 0:1]), axis = 1)
			mxc_b = np.concatenate((mxc[:, 2:3], mxc[:, 0:1], ), axis = 1)
			mxc_b = np.flip(mxc_b, axis = 0)
			mxc = np.concatenate((mxc_t, mxc_b), axis = 0)
			tnc.append(mxc)
		return tnc

	def detect_contour_simple_row(self): 
		# detects boundary of bone at each axial image by searching for it at each row
		tnj_bin = np.copy(self.value)
		tnj_bin[tnj_bin != 0] = 1 # create binary mask
		tnj_bin = np.uint8(tnj_bin)
		tnc = [] # initialize contour as a list
		for k in range(self.shape[0]): # iterate over number of images
			vci = np.arange(self.frame[k, 0, 0], self.frame[k, 0, 1]+1)
			n_i = np.shape(vci)[0]
			mxc = []
			for i in range(n_i):
				vcj = tnj_bin[k, vci[i], :] # take row of mask
				vc = np.where(vcj == 1)[0] # take indices of nonzero pixels in 'vci'
				if len(vc) < 2:
					continue
				mxc.append([vci[i], vc[0], vc[-1]]) # add point to contour
			mxc = np.array(mxc)
			mxc_t = np.concatenate((mxc[:, 0:1], mxc[:, 1:2]), axis = 1)
			mxc_b = np.concatenate((mxc[:, 0:1], mxc[:, 2:3]), axis = 1)
			mxc_b = np.flip(mxc_b, axis = 0)
			mxc = np.concatenate((mxc_t, mxc_b), axis = 0)
			tnc.append(mxc)
		return tnc

	def calc_derivative(self, s_dk):
		# obtains image derivative for each axial image using the input kernel size
		for k in range(self.shape[0]): # iterate over number of images
			mxj_dx = cv.Sobel(self.value[k], ddepth = cv.CV_64F, dx = 1, dy = 0, ksize = s_dk) # calculation x-direction derivative
			mxj_dy = cv.Sobel(self.value[k], ddepth = cv.CV_64F, dx = 0, dy = 1, ksize = s_dk) # calculation y-direction derivative
			self.value[k] = np.round(np.sqrt(mxj_dx**2+mxj_dy**2))
		
	def erode(self, mxk_de, n_iter):
		# erodes each axial image using the input kernel and the number of iterations
		for k in range(self.shape[0]):
			self.value[k] = cv.erode(self.value[k], kernel = mxk_de, iterations = n_iter)
	
	def dilate(self, mxk_de, n_iter):
		# dilates each axial image using the input kernel and the number of iterations
		for k in range(self.shape[0]):
			self.value[k] = cv.dilate(self.value[k], kernel = mxk_de, iterations = n_iter)
	
	def adjust_to_frame(self, mxm_pad, mxf):
		# adds the input padding to the image and crops it to the input frame
		tno_top = np.zeros((-mxm_pad[0, 0], self.shape[1], self.shape[2]), dtype = np.int16) # define padding before k = 0
		tno_bot = np.zeros((mxm_pad[0, 1], self.shape[1], self.shape[2]), dtype = np.int16) # define padding after k = -1
		self.value = np.concatenate((tno_top, self.value, tno_bot), axis = 0)
		self.value = self.value[:, mxf[1, 0]:mxf[1, 1]+1, mxf[2, 0]:mxf[2, 1]+1]
		self.shape = np.array(self.value.shape, dtype = np.int16)

	def downsample(self, tnk, f_ds, vcn_ds):
		# lowers the resolution of the entire image through a 3D convolution using the input kernel, the input downsampling factor, and the input dimension of the downsampled image
		if np.sum((self.shape%(2*f_ds+1))**2) != 0:
			raise ValueError('Image dimension does not allow for downsampling!')
		tnk = np.flip(tnk, axis = 0) # flip kernel for convolution instead of correlation
		tnk = np.flip(tnk, axis = 1) # flip kernel for convolution instead of correlation
		tnj_temp = np.zeros(vcn_ds, dtype = np.int16)
		for k in range(f_ds, f_ds+vcn_ds[0]*(2*f_ds+1), 2*f_ds+1): # iterate over image
			for i in range(f_ds, f_ds+vcn_ds[1]*(2*f_ds+1), 2*f_ds+1): # iterate over image
				for j in range(f_ds, f_ds+vcn_ds[2]*(2*f_ds+1), 2*f_ds+1): # iterate over image
					tn = np.copy(self.value[k-f_ds:k+(f_ds+1), i-f_ds:i+(f_ds+1), j-f_ds:j+(f_ds+1)]) # extract image value at certain location
					v = round(sum(np.ndarray.flatten(tn*tnk))) # perform convolution and round before float-to-int conversion
					vc = np.array([(k-f_ds)/(2*f_ds+1), (i-f_ds)/(2*f_ds+1), (j-f_ds)/(2*f_ds+1)], dtype = np.int16) # calculate indices at downsampled image
					tnj_temp[vc[0], vc[1], vc[2]] = v # update downsampled image
		self.value = tnj_temp
		self.shape = np.array(self.value.shape, dtype = np.int16)
	
	def downsample_w_mask(self, tnk, f_ds, vcn_ds, tnj_mask):
		# lowers the resolution of the entire image through a 3D convolution using the input kernel, the input downsampling factor, and the input dimension of the downsampled image
		# applies a positional condition on assigning nonzero value to a voxel
		if np.sum((self.shape%(2*f_ds+1))**2) != 0:
			raise ValueError('Image dimension does not allow for downsampling!')
		tnk = np.flip(tnk, axis = 0)
		tnk = np.flip(tnk, axis = 1)
		tnj_temp = np.zeros(vcn_ds, dtype = np.int16)
		for k in range(f_ds, f_ds+vcn_ds[0]*(2*f_ds+1), 2*f_ds+1): # iterate over image
			for i in range(f_ds, f_ds+vcn_ds[1]*(2*f_ds+1), 2*f_ds+1): # iterate over image
				for j in range(f_ds, f_ds+vcn_ds[2]*(2*f_ds+1), 2*f_ds+1): # iterate over image
					if tnj_mask[k, i, j] == 1: # check if point is located inside contour
						tn = np.copy(self.value[k-f_ds:k+(f_ds+1), i-f_ds:i+(f_ds+1), j-f_ds:j+(f_ds+1)]) # extract image value at certain location
						v = round(sum(np.ndarray.flatten(tn*tnk))) # perform convolution and round before float-to-int conversion
						vc = np.array([(k-f_ds)/(2*f_ds+1), (i-f_ds)/(2*f_ds+1), (j-f_ds)/(2*f_ds+1)], dtype = np.int16) # calculate indices at downsampled image
						tnj_temp[vc[0], vc[1], vc[2]] = v # update downsampled image
		self.value = tnj_temp
		self.shape = np.array(self.value.shape, dtype = np.int16)
	
	def plot_3d_stack(self, axe, vck):
		# plots in 3D space multiple axial images specified by the input indices in the input axis
		mxi = np.ones((self.shape[2]+1, 1), dtype = np.int16)@[np.arange(self.shape[1]+1)] # create mesh grid
		mxi = np.transpose(mxi)
		mxj = np.ones((self.shape[1]+1, 1), dtype = np.int16)@[np.arange(self.shape[2]+1)] # create mesh grid
		tnj = np.copy(self.value[vck])
		tnj = tnj-np.min(tnj)+1 # scale image for plotting
		tnj_max = np.max(tnj) # scale image for plotting
		for k in range(tnj.shape[0]):
			tnc_face = np.array([tnj[k]])
			tnc_face = np.concatenate((tnc_face, tnc_face, tnc_face, tnc_face), axis = 0)
			tnc_face = np.transpose(tnc_face, axes = (1, 2, 0))
			tnc_face = tnc_face/tnj_max
			axe.plot_surface(mxj, mxi, vck[k]*np.ones(np.shape(mxi)), rstride = 1, cstride = 1, facecolors = tnc_face, antialiased = False)
		return axe

class Contour():
	
	def __init__(self, tnc):
		if not isinstance(tnc, np.ndarray):
			raise ValueError('Input contour must be a numpy array!')
		if len(tnc.shape) != 3:
			raise ValueError('Input contour must be 3 dimensional!')
		self.contour = tnc
		self.shape = np.array(self.contour.shape, dtype = np.int16)
		self.center = np.int16(np.round(np.mean(self.contour, axis = 1))) # round before float-to-integer conversion
		self.axis = np.int16(np.round(np.mean(self.center, axis = 0))) # round before float-to-integer conversion
	
	def convert_to_polar(self, mxc_con = None):
		# converts cartesian representation of contour to polar representation at each axial image
		#if mxc_con != None:
		if mxc_con is not None:
			self.center = mxc_con
		self.contour = np.float32(self.contour)
		for k in range(self.shape[0]): # iterate over the number of axial contours
			vcx = np.int32(self.contour[k, :, 1]-self.center[k, 1])
			vcy = np.int32(self.contour[k, :, 0]-self.center[k, 0])
			vcr = np.sqrt(vcx**2+vcy**2) # calculate radii of points on contour with respect to contour center
			vctht = np.zeros(len(vcr), dtype = np.float32) # initialize vector of angles
			for i in range(self.shape[1]): # iterate over number of points on contour
				x = vcx[i]; y = vcy[i]
				if x > 0 and y >= 0:
					vctht[i] = np.arctan(y/x)
				elif x < 0 and y >= 0:
					vctht[i] = np.pi-np.arctan(-y/x)
				elif x < 0 and y < 0:
					vctht[i] = np.pi+np.arctan(y/x)
				elif x > 0 and y < 0:
					vctht[i] = 2*np.pi-np.arctan(-y/x)
				elif x == 0:
					if y > 0:
						vctht[i] = np.pi/2
					elif y < 0:
						vctht[i] = 3*np.pi/2
			self.contour[k, :, 0] = vctht # update contour by adding angle
			self.contour[k, :, 1] = vcr # update contour by adding radius

	def regulerize_angularly(self, n_bp):
		# resamples contour points at each axial contour for regular spacing angularwise
		vcthe = np.linspace(0, 2*np.pi, n_bp, endpoint = False) # define equally spaced angles
		tnc_reg = np.zeros((self.shape[0], n_bp, 2), dtype = np.float32) # initialize regularized contour
		for k in range(self.shape[0]): # iterate over number of axial contours
			vci = np.argsort(self.contour[k, :, 0])
			self.contour[k] = self.contour[k, vci]
			mxc_1 = np.array([[self.contour[k, -1, 0]-2*np.pi, self.contour[k, -1, 1]]], dtype = np.float32)
			mxc_2 = np.array([[self.contour[k, 0, 0]+2*np.pi, self.contour[k, -1, 1]]], dtype = np.float32)
			mxc_pol = np.concatenate((mxc_1, self.contour[k], mxc_2), axis = 0)
			vcr = np.interp(vcthe, mxc_pol[:, 0], mxc_pol[:, 1]) # interpolate radii at regular angles
			tnc_reg[k, :, 0] = vcthe # update contour 
			tnc_reg[k, :, 1] = vcr # update contour
		self.contour = tnc_reg
		self.shape = np.array(self.contour.shape, dtype = np.int16)

	def convert_to_cartesian(self):
		# converts polar representation of contour to cartesian representation at each axial image
		for k in range(self.shape[0]): # iterate over number of axial contours
			vcx = self.center[k, 1]+self.contour[k, :, 1]*np.cos(self.contour[k, :, 0])
			vcy = self.center[k, 0]+self.contour[k, :, 1]*np.sin(self.contour[k, :, 0])
			self.contour[k, :, 0] = np.round(vcy) # round before float-to-int conversion
			self.contour[k, :, 1] = np.round(vcx) # round before float-to-int conversion
		self.contour = np.int16(self.contour)
		self.shape = np.array(self.contour.shape, dtype = np.int16)
		self.center = np.int16(np.round(np.mean(self.contour, axis = 1)))
		self.axis = np.int16(np.round(np.mean(self.center, axis = 0)))

	def detect_frame(self):
		# detects a bounding box for bone at each axial image
		self.frame = np.zeros((self.shape[0], 2, 2), dtype = np.int16)
		for k in range(self.shape[0]): # iterate over number of axial contours
			self.frame[k, 0] = np.array([np.min(self.contour[k, :, 0]), np.max(self.contour[k, :, 0])])
			self.frame[k, 1] = np.array([np.min(self.contour[k, :, 1]), np.max(self.contour[k, :, 1])])
		return self.frame

	def create_full_mask(self, vcs):
		# creates a full mask at each axial contour with the specified dimension
		self.mask = np.zeros((self.shape[0], vcs[0], vcs[1]))
		for k in range(self.shape[0]): # iterate over number of axial contours
			mxc_ = np.concatenate((self.contour[k, :, 1:2], self.contour[k, :, 0:1]), axis = 1)
			mxc_ = np.int32(mxc_)
			cv.drawContours(self.mask[k], [mxc_], 0, color = [1], thickness = -1)	

	def plot_3d_stack(self, axe, vck, alph):
		# plots in 3D space multiple axial contours specified by the input indices in the input axis
		for k in vck:
			mxc = self.contour[k]
			mxc = np.concatenate((mxc, [mxc[0]]), axis = 0)
			axe.plot(mxc[:, 1], mxc[:, 0], zs = k, zdir = 'z', color = 'black', alpha = alph)
		return axe

class Point():

	def __init__(self, vcp):
		self.position = vcp

	def form_square_neighborhood(self, s_nei):
		# creates a neighborhood of size s_nei of pixels centered at the point
		vci = np.arange(self.position[0]-s_nei, self.position[0]+s_nei+1); mxi = np.reshape(vci, (-1, 1))
		vcj = np.arange(self.position[1]-s_nei, self.position[1]+s_nei+1); mxj = np.reshape(vcj, (-1, 1))
		mxj = np.ones((2*s_nei+1, 1), dtype = np.int16)@np.transpose(mxj)
		mxi = mxi@np.ones((1, 2*s_nei+1), dtype = np.int16)
		vcj = np.ndarray.flatten(mxj); vci = np.ndarray.flatten(mxi)
		mxq = np.concatenate((np.array([vci]), np.array([vcj])), axis = 0)
		mxq = np.transpose(mxq)
		self.neighborhood = np.int16(mxq)

class Mesh():

	def __init__(self, tnj):
		self.image = tnj
		self.shape = np.array(self.image.shape, dtype = np.int32)

	def create_mesh(self):
		# defines nodal and elemental information
		tnn = (-1)*np.ones(self.shape+1, dtype = np.int32) # initialize nodal space
		self.value = [] # initialize list of values of elements
		self.nodes_on_element = [] # initialize list of nodes on elements
		self.nodal_position = [] # initialize nodal position
		mx = np.concatenate(([[0, 0, 0, 0, 1, 1, 1, 1]], [[0, 0, 1, 1, 0, 0, 1, 1]], [[0, 1, 0, 1, 0, 1, 0, 1]]), axis = 0) # define matrix to be used for node number definition
		mx = np.transpose(np.int32(mx))
		cou = 0
		for k in range(self.shape[0]): # iterate over image 
			for i in range(self.shape[1]): # iterate over image
				for j in range(self.shape[2]): # iterate over image
					if self.image[k, i, j] != 0: # check if pixel is nonzero
						self.value.append(self.image[k, i, j]) # update element value
						mxc_ele = np.ones((8, 1), dtype = np.int32)@np.array([[k, i, j]], dtype = np.int32)
						mxc_ele = mxc_ele+mx # define nodes numbers for nodes forming element
						for r in range(8): # iterate over number of nodes on element
							if tnn[mxc_ele[r, 0], mxc_ele[r, 1], mxc_ele[r, 2]] == -1: # check if node in nodal space has not been assigned with node number
								tnn[mxc_ele[r, 0], mxc_ele[r, 1], mxc_ele[r, 2]] = cou # assign node number to node in nodal space
								self.nodal_position.append(np.ndarray.flatten(mxc_ele[r])) # update nodal position
								cou = cou+1	
						self.nodes_on_element.append(np.ndarray.flatten(tnn[mxc_ele[:, 0], mxc_ele[:, 1], mxc_ele[:, 2]])) # update nodes forming element
		self.value = np.array(self.value)
		self.nodes_on_element = np.array(self.nodes_on_element)
		self.nodal_position = np.array(self.nodal_position)
		
	def convert_mesh(self, f_ds, r_cts, e_cut, e_min, vcc_valHU, vcc_HUrho, vcc_duazon):
		# converts positions from indice to length and pixel values to elastic modulus following Dual-zone material model
		self.nodal_position = self.nodal_position*(2*f_ds+1)*r_cts*10**-3 # convert nodal position from indices to lengths
		if np.shape(self.nodes_on_element)[0] == len(self.value):
			self.num_elements = np.shape(self.nodes_on_element)[0]
		else:
			print('Dimensional mismatch between self.nodes_on_element and self.value!')
		self.Hounsfield_unit = vcc_valHU[0]*self.value+vcc_valHU[1] # convert pixel values to HU
		self.density = vcc_HUrho[0]*self.Hounsfield_unit+vcc_HUrho[1] # convert HU to bone mineral density
		self.elastic_modulus = (vcc_duazon[0]*self.density)/vcc_duazon[1] # convert bone mineral density to elastic modulus
		self.elastic_modulus[self.elastic_modulus < e_cut] = e_min # apply cut-off to modulus of elasticity
		self.elastic_modulus = e_min*np.int32(np.round(self.elastic_modulus/e_min)) # regulerize elastic modulus
		vc = np.argsort(self.elastic_modulus)
		self.elastic_modulus = self.elastic_modulus[vc]; self.nodes_on_element = self.nodes_on_element[vc] # sort elemental information based on elastic modulus
		self.unique_elastic_modulus = np.unique(self.elastic_modulus) # keep one instance of each elastic modulus
		return self.nodal_position, self.nodes_on_element, self.elastic_modulus
	
	def write_Ansys_input(self, f_ds, nam):
		# writes an input file for model creation in Ansys
		vc = np.zeros((self.num_elements, ))
		for i in range(len(self.unique_elastic_modulus)):
			vc = vc+i*(self.elastic_modulus == self.unique_elastic_modulus[i])
		mxr = np.concatenate(([vc], [self.elastic_modulus]), axis = 0); mxr = np.int32(np.transpose(mxr))
		fil = open('{}_MatAssignment.txt'.format(nam), 'w')
		for i in range(len(self.unique_elastic_modulus)):
			fil.write('MP, EX, {}, {}\n'.format(i+1, self.unique_elastic_modulus[i]))
			fil.write('MP, PRXY, {}, 0.3\n'.format(i+1))
			fil.write('MP, DENS, {}, 0.01\n'.format(i+1))
		for i in range(np.shape(mxr)[0]):
			fil.write('MPCHG, {}, {}\n'.format(mxr[i, 0]+1, i+1))
		for i in range(np.shape(mxr)[0]):
			fil.write('BFE, {}, TEMP,  , {}\n'.format(i+1, mxr[i, 1]))
		fil.close()	
		fil = open('{}.cdb'.format(nam), 'w')
		fil.write('/COM,ANSYS RELEASE\n')
		fil.write('/TITLE,Ansys Workbench export from Mimics Innovation Suite\n')
		fil.write('/Units: mm\n')
		fil.write('/nopr\n')
		fil.write('/prep7\n')
		fil.write('/com,*********** Nodes ***********\n')
		fil.write('nblock,3,solid\n')
		fil.write('(3i8,3e20.9e3)\n')
		for n in range(np.shape(self.nodal_position)[0]):
			fil.write('{:>8}{:>8}        {:<20}{:<20}{:<20}\n'.format(n+1, 0, self.nodal_position[n, 1], self.nodal_position[n, 2], self.nodal_position[n, 0]))
		fil.write('N,R5.3,LOC,     -1,\n')
		fil.write('/com,*********** Hex8 ***********\n')
		fil.write('et,1,185\n')
		fil.write('eblock,19,solid\n')
		fil.write('(19i8)\n')
		for e in range(np.shape(self.nodes_on_element)[0]):
			fil.write('{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}'.\
			format(1, 1, 1, 0, 0, 0, 0, 0, 8, 0, e+1))
			for c in [0, 1, 3, 2, 4, 5, 7, 6]:
				fil.write('{:>8}'.format(self.nodes_on_element[e, c]+1))
			fil.write('\n')
		fil.close()

	def write_Abaqus_input(self, f_ds, nam, da):
		# writes an input file for model creation in Abaqus
		fil = open('{}_.inp'.format(nam), 'w')
		fil.write('Written by Materialise Abaqus export filter\n')
		fil.write('*HEADING\n')
		fil.write('** Units: mm\n\n')
		vcp_max = np.max(self.nodal_position, axis = 0)
		vcp_min = np.min(self.nodal_position, axis = 0)
		fil.write('!min_xyz_{}_{}_{}\n'.format(vcp_min[1], vcp_min[2], vcp_min[0]))
		fil.write('!max_xyz_{}_{}_{}\n'.format(vcp_max[1], vcp_max[2], vcp_max[0]))
		vca_minz = self.nodal_position[:, 0] == vcp_min[0]
		vca_maxz = self.nodal_position[:, 0] == vcp_max[0]
		vcc_minz = np.mean(self.nodal_position[vca_minz], axis = 0)
		vcc_maxz = np.mean(self.nodal_position[vca_maxz], axis = 0)
		fil.write('!cen_fix_{}_{}_{}\n'.format(vcc_minz[1], vcc_minz[2], vcc_minz[0]))
		fil.write('!cen_twist_{}_{}_{}\n'.format(vcc_maxz[1], vcc_maxz[2], vcc_maxz[0]))
		fil.write('!elementsize_{}\n\n'.format(da))
		fil.write('*NODE\n')
		for n in range(self.nodal_position.shape[0]): # iterate over number of nodes
			fil.write('\t{}, {}, {}, {}\n'.format(n+1, self.nodal_position[n, 1], self.nodal_position[n, 2], self.nodal_position[n, 0])) # print nodal positions
		fil.write('*ELEMENT, TYPE=C3D8\n')
		for e in range(self.nodes_on_element.shape[0]): # iterate over number of elements
			fil.write('\t{}, '.format(e+1)) # print element number 
			for c in [4, 5, 7, 6, 0, 1, 3, 2]:
				fil.write('{}, '.format(self.nodes_on_element[e, c]+1)) # print nodes located on element
			fil.write('\n')
		vcr = np.arange(1, self.num_elements+1)
		for i in range(len(self.unique_elastic_modulus)): # iterate over unique elastic modulus
			vcs = vcr*(self.elastic_modulus == self.unique_elastic_modulus[i]) # classify element numbers based on whether they correspond to certain unique elastic modulus
			vcs = vcs[vcs > 0] # keep element numbers corresponding to certain elastic modulus
			fil.write('*Elset, elset=Set_{}\n'.format(i+1))
			n_fulrow = np.int32(np.floor(len(vcs)/16)) # calculate number of rows with to write 16 element numbers in
			for j in range(n_fulrow):
				vcs_row = vcs[j*16:(j+1)*16]
				vcs_row = ', '.join(vcs_row.astype(str))
				fil.write('{}\n'.format(vcs_row)) # print element numbers belonging to certain elastic modulus
			vcs_row = vcs[n_fulrow*16:]
			vcs_row = ', '.join(vcs_row.astype(str))
			fil.write('{}\n'.format(vcs_row)) # print remaining element numbers belonging to certain elastic modulus
		for i in range(len(self.unique_elastic_modulus)): # iterate over number of unique elastic modulus
			fil.write('*Solid Section, elset=Set_{}, material=Material-{}\n,\n'.format(i+1, i+1)) # print section definitions
		for i in range(len(self.unique_elastic_modulus)): # iterate over number of unique elastic modulus
			fil.write('*Material, name=Material-{}\n'.format(i+1)) # print material definitions
			fil.write('*Elastic\n')
			fil.write('{}, 0.3\n'.format(self.unique_elastic_modulus[i])) # print elastic modulus
		fil.close()

def makekernel(m):
	# creates a custom kernel from the input size
	tnk = np.zeros((2*m+1, 2*m+1, 2*m+1), dtype = np.float32)
	for k in range(2*m+1):
		for i in range(2*m+1):
			for j in range(2*m+1):
				tnk[k, i, j] = np.sqrt((k-m)**2+(i-m)**2+(j-m)**2)
	tnk = (-np.arctan(tnk)+np.pi/2)
	return tnk #np.round(tnk, 2)

def find_3d_frame(tnf, f_ds):
	# identifies a 3D frame from a stack of 2D frames
	vcl = np.argmin(tnf[:, 0:2, 0], axis = 0)
	vch = np.argmax(tnf[:, 0:2, 1], axis = 0)
	mxf_key = np.concatenate(([[vcl[0], vch[0]]], [[vcl[1], vch[1]]]), axis = 0)	
	vcl = np.min(tnf[:, 0:2, 0], axis = 0)
	vch = np.max(tnf[:, 0:2, 1], axis = 0)
	mxf = np.concatenate(([[0, np.shape(tnf)[0]-1]], [[vcl[0], vch[0]]], [[vcl[1], vch[1]]]), axis = 0)
	vcn_ds = np.int16(np.ceil((mxf[:, 1]-mxf[:, 0]+1)/(2*f_ds+1)))
	vcm = vcn_ds*(2*f_ds+1)-(mxf[:, 1]-mxf[:, 0]+1)
	vcm = 0.5*vcm
	mxm_pad = np.concatenate(([-np.ceil(vcm)], [np.floor(vcm)]), axis = 0)
	mxm_pad = np.int16(np.transpose(mxm_pad))
	mxf = mxf+mxm_pad
	mxf[0] = mxf[0]-mxf[0, 0]
	return mxf_key, vcn_ds, mxm_pad, mxf

def find_first_pixel(mx):
	# finds the position of the first non-zero pixel in an image
	cross = Image(np.array([mx]))
	tnf = cross.detect_frame()[0]
	i = tnf[0, 0]
	vc = mx[i]
	j = np.where(vc == 1)[0][0]
	return i, j

def make2dgrid(vcl_i, vcl_j):
	# creates a mesh grid to be used for plotting from the input limits
	mxi = np.ones((vcl_j[1]-vcl_j[0]+1, 1), dtype = np.int16)@[np.arange(vcl_i[0], vcl_i[1]+1)]
	mxi = np.transpose(mxi)
	mxj = np.ones((vcl_i[1]-vcl_i[0]+1, 1), dtype = np.int16)@[np.arange(vcl_j[0], vcl_j[1]+1)]
	return mxi, mxj

def calculate_normal_vector(vcp_pp, vcp_p, mxp, vcp_n, vcp_nn):
	# calculates normal vector at a point on a contour by finite difference using the input previous and next points
	mxn = (-1/12)*vcp_pp+(4/3)*vcp_p+(-5/2)*mxp+(4/3)*vcp_n+(-1/12)*vcp_nn
	return mxn

def calculate_angle(vcu, vcv):
	# calculates the angle between the two unit vectors
	a = vcu@np.transpose(vcv)
	a = np.sign(a)*min(1, np.abs(a))
	tht = np.arccos(a)
	return tht

def creatcols():
	# creats a custom color palette
	mxp_col = [[216, 202, 173], \
			  [220, 208, 182], \
			  [225, 214, 192], \
			  [220, 221, 202], \
			  [220, 221, 202], \
			  [207, 217, 221], \
			  [172, 190, 200], \
			  [134, 159, 173], \
			  [87, 115, 130], \
			  [192, 0, 0], \
			  [248, 225, 225], \
			  [242, 242, 242]]
	mxp_col = np.array(mxp_col, dtype = np.float32)
	mxp_col = mxp_col/256
	return mxp_col