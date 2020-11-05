import tensorflow as tf
import pickle
import numpy as np
import pdb
import time
import os
import cv2
from numpy import dot
from numpy.linalg import norm



def tf_init(tf_model_file):
	print("\n\ninit\n\n")
	device_name = "GPU:0"
	f = tf.gfile.GFile(tf_model_file, "rb")
	graph_def = tf.GraphDef()
	str1 = f.read()
	graph_def.ParseFromString(str1)
	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def)
		inp = tf.get_default_graph().get_tensor_by_name('import/data:0') 
		out = tf.get_default_graph().get_tensor_by_name('import/softmax:0') 
		tf.global_variables_initializer()
		tf.constant_initializer()
		tf.local_variables_initializer()
		#init = graph.get_operation_by_name("import/init")

		print(inp, out)
		cfg = dict({'allow_soft_placement': True,'log_device_placement': False})
		utility = 0.3
		cfg['gpu_options'] = tf.GPUOptions(per_process_gpu_memory_fraction = utility,allow_growth=True)
		cfg['allow_soft_placement'] = True
		cfg['device_count'] = {'GPU': 1}
		cfg['use_per_session_threads'] = True
		cfg['intra_op_parallelism_threads'] = 1
		cfg['inter_op_parallelism_threads'] = 1
		sess = tf.Session(config = tf.ConfigProto(**cfg))
		#########
		init = graph.get_operation_by_name("import/init")
		sess.run(init)
		#########
	print("\n\ninit-exit\n\n")
	return sess,inp,out


def tf_processing(im_pp,net):
	inp_feed = list()
	for imsz1 in im_pp:
		this_inp = np.expand_dims(imsz1, 0)
		print(this_inp.shape)
		inp_feed.append(this_inp)
	feed_dict = {net['tf_inp'] : np.concatenate(inp_feed, 0)}
	output = net['sess'].run(net['tf_out'], feed_dict)
	out = output[0]

	
	
	return output

#Takes 224x224 rgb images that have undergone wraping by lm model
if __name__ == '__main__':

	model_path = "./tf_frozen_resnet101.pb"
	net_r = 224 
	net_c = 224 
	batch_size =1

	sess,tf_inp,tf_out = tf_init(model_path)

	net = {
	'sess':sess,\
	'tf_inp':tf_inp,\
	'tf_out':tf_out,\
	'batsiz':batch_size}
	# model specific
	dummyimg = np.zeros((224, 224, 3), dtype = "uint8") 
	dummyimg = np.transpose(dummyimg,(2,0,1))

	im_pp = []
	im_orig  = cv2.imread("./0.jpg")
	# print(im_orig.shape)
       
	h1,w1,_ = im_orig.shape
	maxdim = max(h1,w1)
	tempimg = np.zeros((maxdim, maxdim, 3), dtype = "uint8")
	tempimg[0:h1,0:w1] = im_orig
	Scaling = float(maxdim) / float(net_c)
	imsz1 = cv2.resize(tempimg, (net_c,net_r))
	imsz1 = imsz1[:,:,(2,1,0)]
	im_pp = []
	for b in range(batch_size):
		im_pp.append(imsz1)	
	print(tf.__version__)
	tf_feats = tf_processing(im_pp,net)
	



