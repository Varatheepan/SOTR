import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import math 

sys.path.append('../../utils/gwr_old')
from quantization import *
from distance_measures import *

class GWR:
	def __init__(self, activity_thr=0.99, firing_counter=0.3, epsn=0.01, epsb=0.1, max_edge_age=50, random_state=42, enable_quantization=False, quantization_thresh=True):
		self.nodes = None
		self.conns = dict()
		self.activity_thr = activity_thr
		self.firing_counter = firing_counter or np.inf
		self.random_state = random_state
		self.eps_n = epsn
		self.eps_b = epsb
		self.max_edge_age = max_edge_age
		self.delta_ws = None
		self.mins = None
		self.deltas = None
		self.enable_quantization = enable_quantization
		self.quantization_thresh = quantization_thresh

		self.winner_firing_counters = np.array([1-(1-np.exp(-1.05*k/3.33))/1.05 for k in range(100)])
		self.neighbor_firing_counters = np.array([1-(1-np.exp(-1.05*k/14.3))/1.05 for k in range(100)])

		self.on_train_sample_callback = lambda *_, **__: None
		self.on_test_sample_callback = lambda *_, **__: None

		self.dist_pred_list = [] #for test purposes

	def register_on_train_sample_callback(self, on_train_sample_callback) : 
		self.on_train_sample_callback = on_train_sample_callback
		
	def register_on_test_sample_callback(self, on_test_sample_callback) : 
		self.on_test_sample_callback = on_test_sample_callback

	def _get_next_winner_firing_counter(self, cur_value):
		if cur_value == 0:
			return self.winner_firing_counters[0]
		if cur_value > self.winner_firing_counters[-1]:
			return self.winner_firing_counters[self.winner_firing_counters < cur_value][0]
		return self.winner_firing_counters[-1]

	def _get_next_neighbor_firing_counter(self, cur_value):
		if cur_value == 0:
			return self.neighbor_firing_counters[0]
		if cur_value > self.neighbor_firing_counters[-1]:
			return self.neighbor_firing_counters[self.neighbor_firing_counters < cur_value][0]
		return self.neighbor_firing_counters[-1]

	def _init(self, arg_xs, normalize):
		if normalize:
			self.mins = np.min(arg_xs, axis=0)
			self.deltas = np.max(arg_xs, axis=0) - self.mins

		self.nodes = []
		for k in range(2):
			random_idxs = np.unique(np.random.choice(len(arg_xs), int(len(arg_xs) / 4)))
			sub_xs = arg_xs[random_idxs, :]
			node_w = np.mean(sub_xs, axis=0)
			if normalize:
				node_w = (node_w - self.mins) / self.deltas
			self.nodes.append(Node(id=k, w=node_w))
		self.conns = dict()

	def fit(self, arg_xs, y=None, normalize=True, iters=None, verbose=False, delta_thr=None, warm_start=False):
		np.random.seed(self.random_state)

		if iters is None:
			iters = 1

		if delta_thr is None:
			delta_thr = 1e-4

		if not warm_start or self.nodes is None:
			self._init(arg_xs, normalize)
		# print(arg_xs)
		xs = arg_xs
		if self.mins is not None and self.deltas is not None:
			xs = (arg_xs - self.mins) / self.deltas
		deltas = []

		if verbose:
			print("delta thr:", delta_thr)
			print("activity thr:", self.activity_thr)
		for t in range(iters):
			if len(self.nodes) > 0.1*len(xs) and self.activity_thr > 0.8:
				self.activity_thr *= 0.95
				# self.activity_thr = max(self.activity_thr, 0.8)
				self.activity_thr = max(self.activity_thr, 0.85)

			cur_deltas = []
			num_xs = xs.shape[0]
			# for x in xs:
			for i in range(num_xs) :
				x = xs[i]
				label = y[i]
				cur_dw = self._fit_sample(x, label)
				cur_deltas.append(np.sqrt(np.dot(cur_dw, cur_dw) / len(cur_dw)))
			if len(cur_deltas) > 0:
				deltas.append(np.mean(cur_deltas))

			if verbose:
				print("{:3}:".format(t+1), len(cur_deltas), deltas[-1], len(self.nodes), len(self.conns))
			if deltas[-1] < delta_thr:
				break
		if self.delta_ws is None:
			self.delta_ws = deltas
		else:
			self.delta_ws.extend(deltas)

	def normalize_sample(self, x):
		min = np.min(x)
		max = np.max(x)
		delta = max - min
		# self.deltas = np.max(arg_xs, axis=0) - self.mins
		x_temp = (x-min)
		sum_x_temp = np.sum(x_temp)
		norm_x = x_temp/sum_x_temp
		return norm_x

	''' 
	quantize to int8 before calculating distance
	'''
	def _check_activity(self, x , w, dequant=False):
		if(self.enable_quantization) : 
			# approximated activity calculation
			# dist = np.linalg.norm(x - w)
			x_quant = quantize_vector(x, self.quantization_thresh)
			w_quant = quantize_vector(w, self.quantization_thresh)
			dist_quant   = manhattan_distance(x_quant, w_quant, enable_quantization=True)
			dist         = dequantize_vector(dist_quant, self.quantization_thresh)
			activity     = np.exp(-dist / np.sqrt(len(x)))
			# activity     = dist
			# activity_thr = 123 #122 #122 #125 #activity for exponential
			activity_thr = 0.00075 #0.00075     #activity for distance
			print("dist : ", dist, " activity : ", activity, " custom thresh : 234.89")
			#if(activity < activity_thr) : 
			#duvindu test
			if(dist > 205) : #> 234.891183287): (core 50)
			# if(dist > 5) : #> 234.891183287): (mnist)
				return True
			else : 
				return False
		else :  
			dist         = np.linalg.norm(x - w)
			# dist         = manhattan_distance(x, w)
			activity     = np.exp(-dist / np.sqrt(len(x)))
			activity_thr = self.activity_thr
			print("dist : ", dist, " activity : ", activity, " activity thresh : ", activity_thr)
			if(activity < activity_thr) : 
				return True
			else : 
				return False
		#original calculation
		# return np.exp(-np.linalg.norm(x - w) / np.sqrt(len(x)))

	def _learn_function(self, x, node, learning_rate) : 
		if(self.enable_quantization) : 
			dw = self._learn_function_quant(x, node, learning_rate)
		else :
			dw = self._learn_function_fp(x, node, learning_rate)
		return dw

	'''
	quantize the diff calculation, lr multiplication
	convert back to float before saving
	'''
	def _learn_function_quant(self, x, node, learning_rate) : 
		effective_lr           = learning_rate * node.firing_counter
		# print("lr : ", effective_learning_rate)
		lr_quantiztion_thresh   = learning_rate 
		i16_effective_lr        = quantize_vector_pow2(effective_lr, lr_quantiztion_thresh)
		i16_x                   = quantize_vector_pow2(x, self.quantization_thresh)
		i16_w                   = quantize_vector_pow2(node.w, self.quantization_thresh)

		i16_diff                = i16_x - i16_w
		i16_diff                = i16_effective_lr * i16_diff

		# print("i16_effective lr : ", i16_effective_lr, " i16_effective_lr exp : ", i16_effective_lr_exp)
		i16_diff                = dequantize_vector_pow2(i16_diff, lr_quantiztion_thresh)
		f32_diff                = dequantize_vector(i16_diff, self.quantization_thresh)
		i16_new_w               = i16_w + i16_diff
		f32_w                   = dequantize_vector(i16_new_w, self.quantization_thresh)
		node.w                  = f32_w
		return f32_diff

	def _learn_function_fp(self, x, node, learning_rate) : 
		dist_neg = np.where( (x-node.w) < 0 )
		dw1     = learning_rate * node.firing_counter * (x - node.w)
		node.w += dw1
		return dw1

	def _get_next_node_id(self):
		return 1+max([n.id for n in self.nodes])
	
	''' 
	quantize to int16 before calculating new vector weight. convert back to float before saving
	'''
	def _add_new_node(self, x, winner_node_1, winner_node_2,winner_conn) : 
		if(self.enable_quantization) : 
			i16_w = quantize_vector_pow2(winner_node_1.w, self.quantization_thresh)
			i16_x = quantize_vector_pow2(x, self.quantization_thresh)
			i16_new_node_w = i16_w + i16_x                      # w = w + x
			i16_new_node_w = np.right_shift(i16_new_node_w, 1)  # w = w * 0.5
			f32_w       = dequantize_vector(i16_new_node_w, self.quantization_thresh)
			new_node_w  = f32_w
		else : 
			new_node_w = 0.5*(winner_node_1.w + x)
		new_node = Node(id=self._get_next_node_id(), w=new_node_w)
		self.nodes.append(new_node)
		#add new hebbian connection
		self.conns[(winner_node_1.id, new_node.id)] = 0
		self.conns[(winner_node_2.id, new_node.id)] = 0
		del self.conns[winner_conn]
		dw1 = new_node.w - x
		return new_node, dw1

	def _fit_sample(self, x, y=None):
		"""

		:param x:
		:return:    
		"""
		# def _get_next_node_id():
		#     return 1+max([n.id for n in self.nodes])

		# find the best and second best matching nodes.
		node1, node2, min_dist1, min_dist2 = self._find_best_matching_nodes(x)
		cur_conn = (node1.id, node2.id) if node1.id < node2.id else (node2.id, node1.id)
		self.conns[cur_conn] = 0
		# add a new node or update node1 and node2.
		# activity1 = self._get_activity(x, node1.w)
		# activity1 = np.exp(-np.linalg.norm(x - node1.w) / np.sqrt(len(x)))
		# if activity1 < self.activity_thr and node1.firing_counter < self.firing_counter and node2.firing_counter < self.firing_counter:
		if self._check_activity(x, node1.w) and node1.firing_counter < self.firing_counter and node2.firing_counter < self.firing_counter:
			# new_node = Node(id=_get_next_node_id(), w=0.5*(node1.w + x))
			# self.nodes.append(new_node)
			# self.conns[(node1.id, new_node.id)] = 0
			# self.conns[(node2.id, new_node.id)] = 0
			# del self.conns[cur_conn]
			# dw1 = new_node.w - x
			new_node, dw1 = self._add_new_node(x, node1, node2, cur_conn)
			node1 = new_node
			# print("add a new node", node1.id, self.conns)
		else:
			# dw1     = self.eps_b * node1.firing_counter * (x - node1.w)
			# node1.w += dw1
			#custom train function
			dw1 = self._learn_function(x, node1, self.eps_b)
			#duvindu_fix : instead of 2nd best all neighbor weights should be updated
			# dw2 = self.eps_n * node2.firing_counter * (x - node2.w)
			# node2.w += dw2

		# age connections incident to node1 and update firing counters of its neighborhs.
		node1.firing_counter = self._get_next_winner_firing_counter(node1.firing_counter) #  h0 - 1/alpha_b * (1 - np.exp(-alpha_b * t / tau_b))
		for conn in self.conns:
			if node1.id not in conn:
				continue
			
			self.conns[conn] += 1
			neighbor_id = (set(conn) - set([node1.id])).pop()
			neighbor_node = None
			for n in self.nodes:
				if n.id == neighbor_id:
					neighbor_node = n
					break
			#duvindu_fix : instead of 2nd best all neighbor weights should be updated
			# dwn = self.eps_n * neighbor_node.firing_counter * (x - neighbor_node.w)
			# neighbor_node.w += dwn
			#custom train function
			self._learn_function(x, neighbor_node, self.eps_n)
			neighbor_node.firing_counter = self._get_next_neighbor_firing_counter(neighbor_node.firing_counter)   # h0 - 1 / alpha_n * (1 - np.exp(-alpha_n * t / tau_n))

		self._remove_dangling_nodes()
		self._remove_too_old_edges()

		self.on_train_sample_callback(node1.id, y)
		# self.get_neighbor_counts()
		# self.get_connection_ages()

		return dw1

	def _remove_dangling_nodes(self):
		node_ids = {node.id: node for node in self.nodes}
		# del node ids with active connections
		for c in self.conns:
			if c[0] in node_ids:
				del node_ids[c[0]]
			if c[1] in node_ids:
				del node_ids[c[1]]
		# delete remaining nodes
		for node_id, node in node_ids.items():
			# print("Delete node id : ", node_id)
			# raw_input("Enter")
			self.nodes.remove(node)

	def _remove_too_old_edges(self):
		conn_ids = list(self.conns)
		for conn_id in conn_ids:
			if self.conns[conn_id] > self.max_edge_age:
				# print("Delete connection id : ", conn_id)
				# raw_input("Enter")
				del self.conns[conn_id]

	"""
	fvec and weights are quantized before calculating distance
	"""
	def _find_best_matching_nodes(self, argx):
		min_dist1, min_dist2 = np.inf, np.inf
		node1, node2 = None, None

		if(self.enable_quantization) : 
			argx_quant = quantize_vector(argx, self.quantization_thresh)
			# argx_quant = argx
		for node in self.nodes:
			if(self.enable_quantization) : 
				quant_w = quantize_vector(node.w, self.quantization_thresh)
				# quant_w = node.w
				# dist    = np.linalg.norm(argx_quant - quant_w)
				# dist    = manhattan_distance(argx, node.w)
				dist    = manhattan_distance(argx_quant, quant_w, enable_quantization=True)
				# dist    = dequantize_vector(dist, self.quantization_thresh)
			else : 
				# dist = manhattan_distance(argx, node.w)
				dist    = np.linalg.norm(argx - node.w)
			if dist < min_dist2:
				min_dist2 = dist
				node2 = node
			if dist < min_dist1:
				node2 = node1
				min_dist2 = min_dist1
				node1 = node
				min_dist1 = dist

		assert(node1.id != node2.id)
		return node1, node2, min_dist1, min_dist2

	"""
	fvec and weights are quantized before calculating distance
	"""
	def _find_best_matching_nodes_test(self, argx):
		min_dist1, min_dist2 = np.inf, np.inf
		node1, node2 = None, None

		if(self.enable_quantization) :
			argx_quant = quantize_vector(argx, self.quantization_thresh)           
		for node in self.nodes:
			if(self.enable_quantization) :
				quant_w  = quantize_vector(node.w, self.quantization_thresh)
				dist     = manhattan_distance_test(argx_quant, quant_w, enable_quantization=True)
			else :
				# dist     = manhattan_distance(argx, node.w)
				dist    = np.linalg.norm(argx - node.w)
			if dist < min_dist2:
				min_dist2 = dist
				node2 = node
			if dist < min_dist1:
				node2 = node1
				min_dist2 = min_dist1
				node1 = node
				min_dist1 = dist

		assert(node1.id != node2.id)
		return node1, node2, min_dist1, min_dist2

	def fit_predict(self, xs, y=None):
		self.fit(xs, y)
		return self.predict(xs)

	def predict_samples(self, x, y=None): 
		node1, node2, min_dist1, min_dist2 = self._find_best_matching_nodes_test(x)
		prediction = self.on_test_sample_callback(node1.id, y)

		#append input and predictions to an array
		if(self.enable_quantization):
			fvec_quant = quantize_vector(x, self.quantization_thresh)        
			fvec_quant = np.append(fvec_quant, node1.id)
			fvec_quant = np.append(fvec_quant, node2.id)
			fvec_quant = np.append(fvec_quant, min_dist1)
			fvec_quant = np.append(fvec_quant, min_dist2)
			self.dist_pred_list.append(fvec_quant)
		
		return prediction
	
	def predict(self, xs, y=None, labeldim = 1):
		num_xs = xs.shape[0]
		prediction_sample = []
		for i in range(num_xs):
			x_sample = xs[i]
			y_sample = y[i]
			prediction_sample.append(self.predict_samples(x_sample, y_sample))
		   
		return prediction_sample

	# def predict_samples(self, x, y=None): 
	#     # x = self.normalize_sample(x)
	#     node1, node2, min_dist1, min_dist2 = self._find_best_matching_nodes(x)
	#     is_correct = self.on_test_sample_callback(node1.id, y)
	#     return is_correct

	# def predict(self, xs, y=None, labeldim = 1):
	#     # pass
	#     if(labeldim == 1) : 
	#         correct_count = 0
	#     else : 
	#         correct_count = [0] * labeldim
	#     num_xs = xs.shape[0]
	#     for i in range(num_xs):
	#         x_sample = xs[i]
	#         y_sample = y[i]
	#         is_correct = self.predict_samples(x_sample, y_sample)

	#         if(labeldim == 1) : 
	#             if(is_correct) : 
	#                 correct_count += 1
	#         else : 
	#             for i in range(labeldim) : 
	#                 if(is_correct[i]) : 
	#                     correct_count[i] += 1
	#     return correct_count

	def get_weights(self):
		res = []
		for node in self.nodes:
			if self.mins is not None and self.deltas is not None:
				res.append(node.w * self.deltas + self.mins)
			else:
				res.append(node.w)
		return res

	#for hw testing
	def dump_weights(self,filename):
		weights = []
		for node in self.nodes:
			if(self.enable_quantization) : 
				quant_w = quantize_vector(node.w, self.quantization_thresh)
				weights.append(quant_w)
			else :
				weights.append(node.w)
		weights_np = np.array(weights)
		weights_np.tofile(filename)

	def dump_dist_predictions(self, filename) : 
		dist_pred_np = np.array(self.dist_pred_list)
		print("dist pred np : ", dist_pred_np.shape)
		print(dist_pred_np)
		dist_pred_np.tofile(filename)

	def dump(self, filename):
		with open(filename, mode='w') as fd:
			fd.write(jsonpickle.dumps(self))

	def get_neighbor_counts(self):
		neighbor_counts = []
		for node in self.nodes:
			neighbor_count = 0
			for conn in self.conns:
				if node.id not in conn:
					continue
				else:
					neighbor_count+=1
			neighbor_counts.append(neighbor_count)
		fig = plt.figure()
		plt.ion()
		plt.show()
		plt.gca().set_title('GwR Neighbour counts', pad=15, fontsize=20)
		plt.plot(neighbor_counts)
		return neighbor_counts
	
	def get_connection_ages(self) : 
		connection_ages = []
		conn_ids = list(self.conns)
		for conn_id in conn_ids:
			connection_ages.append(self.conns[conn_id])
		fig = plt.figure()
		plt.ion()
		plt.show()
		plt.gca().set_title('GwR Connection ages', pad=15, fontsize=20)
		plt.plot(connection_ages)
		return connection_ages
	
	@staticmethod
	def load(filename):
		with open(filename, mode='r') as fd:
			s = fd.read()
		res = jsonpickle.loads(s)
		if res.conns:
			conns = dict()
			for str_key, value in res.conns.items():
				conns[eval(str_key)] = value
			res.conns = conns
		return res

class Node:
	def __init__(self, id=None, w=None):
		self.id = id
		self.w = w
		self.firing_counter = 1

	def __eq__(self, other):
		if type(other) != type(self):
			return False

		return (other.id == self.id) and np.all(other.w == self.w) and (other.firing_counter == self.firing_counter)
