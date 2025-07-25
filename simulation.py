from abaqus import *
from abaqusConstants import *
from odbAccess import openOdb
import numpy as np

class AbaqusSimulation():

	def __init__(self, nam, loc_wd, meth):
		self.name = nam+'_{}'.format(meth)
		self.loc_wd = loc_wd
		inp_fil = open('{}{}_.inp'.format(self.loc_wd, self.name), 'r')
		inp = inp_fil.readlines()
		mxf = np.zeros((2, 3), dtype = float)
		for i in range(2):
			mxf[i] = np.array(inp[4+i].split('_')[2:], dtype = float)
		mxf = np.transpose(mxf)
		mxc = np.zeros((2, 3), dtype = float)
		for i in range(2):
			mxc[i] = np.array(inp[6+i].split('_')[2:], dtype = float)
		mxc = np.transpose(mxc)
		self.frame = mxf
		self.center = mxc
		self.element_size = float(inp[8].split('_')[1])
		inp_fil.close()
		

	def preprocess(self, n_cpu):
		mdb.ModelFromInputFile(inputFileName = '{}{}_.inp'.format(self.loc_wd, self.name), name = self.name)
		if 'Model-1' in mdb.models:
			del mdb.models['Model-1']
		mod = mdb.models[self.name]
		mod.parts.changeKey(fromName = 'PART-1', toName = 'bone')
		part = mod.parts['bone']
		ass = mod.rootAssembly
		ass.features.changeKey(fromName = 'PART-1-1', toName = 'bone')
		ins = ass.instances['bone']
		for i, nam in enumerate(['fix', 'twist']):
			nodes = ins.nodes.getByBoundingBox(self.frame[0, 0], self.frame[1, 0], self.frame[2, i]-0.5*self.element_size, self.frame[0, 1], self.frame[1, 1], self.frame[2, i]+0.5*self.element_size)
			ass.Set(nodes = nodes, name = nam)
		ass.ReferencePoint(point = self.center[:, 1])
		key = ass.referencePoints.keys()[0]
		rot_cen = ass.Set(referencePoints = (ass.referencePoints[key], ), name = 'rot_cen')
		mod.Coupling(controlPoint = rot_cen, couplingType = KINEMATIC, influenceRadius = WHOLE_SURFACE, name = 'twist', surface = ass.sets['twist'])
		mod.StaticStep(previous = 'Initial', name = 'Load')
		mod.EncastreBC(createStepName = 'Initial', name = 'fix', region = ass.sets['fix'])
		mod.DisplacementBC(createStepName = 'Load', name = 'twist', region = ass.sets['rot_cen'], u1 = 0, u2 = 0, ur3 =  0.0174532925)
		mdb.Job(model = self.name, name = self.name, numCpus = n_cpu, numDomains = n_cpu)
		mdb.jobs[self.name].writeInput()


	def create_bash(self, n_cpu, fold):
		fil = open('{}.sh'.format(self.name), 'w')
		fil.write('#!/bin/bash\n')
		fil.write('#SBATCH --partition=enge\n')
		fil.write('#SBATCH --time=1-00:00:00\n')
		fil.write('#SBATCH --job-name="{}"\n'.format(self.name))
		fil.write('#SBATCH --nodes=1\n')
		fil.write('#SBATCH --ntasks-per-node={}\n'.format(n_cpu))
		fil.write('#SBATCH --mail-type=ALL\n')
		fil.write('#SBATCH --output={}_out.log\n'.format(self.name))
		fil.write('#SBATCH --error={}_err.log\n'.format(self.name))
		fil.write('module load abaqus\n')
		fil.write('abaqus job={} input=/home/ala222/{}/{}.inp cpus={} scratch=/home/ala222 interactive\n'.format(self.name, fold, self.name, n_cpu))
		fil.close()

	def postprocess(self):
		odb = openOdb('{}.odb'.format(self.name))
		ass = odb.rootAssembly
		fram = odb.steps['Load'].frames[-1]
		nodes_fix = ass.nodeSets['FIX'].nodes[0]
		nodes_twi = ass.nodeSets['TWIST'].nodes[0]
		mxf_fix = np.zeros((len(nodes_fix), 3), dtype = float)
		mxp_fix = np.zeros((len(nodes_fix), 3), dtype = float)
		mxu_twi = np.zeros((len(nodes_twi), 3), dtype = float)
		mxp_twi = np.zeros((len(nodes_twi), 3), dtype = float)
		for i in range(len(nodes_fix)):
			mxf_fix[i] = fram.fieldOutputs['RF'].getSubset(region = nodes_fix[i], position = NODAL).values[0].data
			mxp_fix[i] = nodes_fix[i].coordinates
		for i in range(len(nodes_twi)):
			mxu_twi[i] = fram.fieldOutputs['U'].getSubset(region = nodes_twi[i], position = NODAL).values[0].data
			mxp_twi[i] = nodes_twi[i].coordinates
		self.center[:, 0] = np.mean(mxp_fix, axis = 0)
		self.center[:, 1] = np.mean(mxp_twi, axis = 0)
		mxp_fix = mxp_fix-self.center[:, 0]
		mxp_twi = mxp_fix-self.center[:, 1]
		m_fix = np.sum(mxp_fix[:, 1]*mxf_fix[:, 0])-np.sum(mxp_fix[:, 0]*mxf_fix[:, 1])
		vtr = m_fix*(self.frame[2, 1]-self.frame[2, 0])/1e6#; vtr = round(vtr, 4)
		np.save('{}_mxf_fix.npy'.format(self.name), mxf_fix)
		np.save('{}_mxp_fix.npy'.format(self.name), mxp_fix)
		fil = open('{}_summary.txt'.format(self.name), 'w')
		fil.write('Reaction Force: {}\n'.format(np.sum(mxf_fix, axis = 0)))
		fil.write('Reaction Moment: {}\n'.format(m_fix))
		fil.write('VTR: {}\n'.format(vtr))
		fil.close()
		odb.close()

n_cpu = 32
method = 'SRT'; fold = method
dire_cur = os.getcwd()
dire_parent = os.path.dirname(dire_cur)
loc_wd = '{}/Abaqus_WD/'.format(dire_parent)
pre = True
for nam in os.listdir('{}/{}/'.format(dire_parent, method)):
	abaqusSimulation = AbaqusSimulation(nam, loc_wd, method)
	if pre == True:
		abaqusSimulation.preprocess(n_cpu)
		abaqusSimulation.create_bash(n_cpu, fold)
	else:
		abaqusSimulation.postprocess()
