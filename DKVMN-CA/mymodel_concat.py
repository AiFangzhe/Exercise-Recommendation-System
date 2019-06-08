import numpy as np
import os, time
import tensorflow as tf
import operations
import shutil
from memory import DKVMN
from sklearn import metrics
import math
import pickle
from sklearn.metrics import classification_report

class Model():
	def __init__(self, args, sess, name='KT'):
		self.args = args
		self.name = name
		self.sess = sess
		self.sess.run(tf.global_variables_initializer())
		self.create_model()
		# if self.load():
		# 	print('CKPT Loaded')
		# else:
		# 	raise Exception('CKPT need')



	def create_model(self):
		# 'seq_len' means question sequences
		self.q_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='q_data')
		self.qa_data = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len], name='qa_data')
		self.target = tf.placeholder(tf.float32, [self.args.batch_size, self.args.seq_len], name='target')
		self.kg = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len, 3], name='knowledge_tag')
		self.kg_hot = tf.placeholder(tf.float32, [self.args.batch_size, self.args.seq_len, 188], name='knowledge_hot')
		self.timebin = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len])
		self.diff = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len])
		self.guan = tf.placeholder(tf.int32, [self.args.batch_size, self.args.seq_len])

		with tf.variable_scope('Memory'):
			init_memory_key = tf.get_variable('key', [self.args.memory_size, self.args.memory_key_state_dim], \
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			init_memory_value = tf.get_variable('value', [self.args.memory_size,self.args.memory_value_state_dim], \
				initializer=tf.truncated_normal_initializer(stddev=0.1))
		with tf.variable_scope('time'):
			time_embed_mtx = tf.get_variable('timebin', [12, self.args.memory_value_state_dim],\
				initializer=tf.truncated_normal_initializer(stddev=0.1))
		with tf.variable_scope('diff'):
			guan_embed_mtx = tf.get_variable('diff', [12, self.args.memory_value_state_dim],\
				initializer=tf.truncated_normal_initializer(stddev=0.1))

		with tf.variable_scope('gate'):
			diff_embed_mtx = tf.get_variable('gate', [12, self.args.memory_value_state_dim],\
				initializer=tf.truncated_normal_initializer(stddev=0.1))

		init_memory_value = tf.tile(tf.expand_dims(init_memory_value, 0), tf.stack([self.args.batch_size, 1, 1]))
		print(init_memory_value.get_shape())

		self.memory = DKVMN(self.args.memory_size, self.args.memory_key_state_dim, \
				self.args.memory_value_state_dim, init_memory_key=init_memory_key, init_memory_value=init_memory_value, batch_size=self.args.batch_size, name='DKVMN')


		with tf.variable_scope('Embedding'):
			# A
			q_embed_mtx = tf.get_variable('q_embed', [self.args.n_questions+1, self.args.memory_key_state_dim],\
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			# B
			qa_embed_mtx = tf.get_variable('qa_embed', [2*self.args.n_questions+1, self.args.memory_value_state_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))


		q_embed_data = tf.nn.embedding_lookup(q_embed_mtx, self.q_data)
		slice_q_embed_data = tf.split(q_embed_data, self.args.seq_len, 1)

		qa_embed_data = tf.nn.embedding_lookup(qa_embed_mtx, self.qa_data)
		slice_qa_embed_data = tf.split(qa_embed_data, self.args.seq_len, 1)

		time_embedding = tf.nn.embedding_lookup(time_embed_mtx, self.timebin)
		slice_time_embedding = tf.split(time_embedding, self.args.seq_len, 1)

		guan_embedding = tf.nn.embedding_lookup(diff_embed_mtx, self.diff)
		slice_guan_embedding = tf.split(guan_embedding, self.args.seq_len, 1)

		diff_embedding = tf.nn.embedding_lookup(diff_embed_mtx, self.diff)
		slice_diff_embedding = tf.split(diff_embedding, self.args.seq_len, 1)

		slice_kg = tf.split(self.kg, self.args.seq_len, 1)

		slice_kg_hot = tf.split(self.kg_hot, self.args.seq_len, 1)

		reuse_flag = False

		prediction = list()

		# Logics
		for i in range(self.args.seq_len):
			# To reuse linear vectors
			if i != 0:
				reuse_flag = True

			q = tf.squeeze(slice_q_embed_data[i], 1)
			qa = tf.squeeze(slice_qa_embed_data[i], 1)
			kg = tf.squeeze(slice_kg[i], 1)
			kg_hot = tf.squeeze(slice_kg_hot[i], 1)
			dotime = tf.squeeze(slice_time_embedding[i], 1)
			dodiff = tf.squeeze(slice_diff_embedding[i], 1)
			doguan = tf.squeeze(slice_guan_embedding[i], 1)

			self.correlation_weight = self.memory.attention(q, kg, kg_hot)

			# # Read process, [batch size, memory value state dim]
			self.read_content = self.memory.read(self.correlation_weight)

			mastery_level_prior_difficulty = tf.concat([self.read_content, q, doguan], 1)

			# f_t
			summary_vector = tf.tanh(operations.linear(mastery_level_prior_difficulty, self.args.final_fc_dim, name='Summary_Vector', reuse=reuse_flag))
			# p_t
			pred_logits = operations.linear(summary_vector, 1, name='Prediction', reuse=reuse_flag)

			prediction.append(pred_logits)

			qa_time = tf.concat([qa, dotime], axis=1)


			self.new_memory_value = self.memory.write(self.correlation_weight, qa_time, reuse=reuse_flag)

# 'prediction' : seq_len length list of [batch size ,1], make it [batch size, seq_len] tensor
		# tf.stack convert to [batch size, seq_len, 1]
		self.pred_logits = tf.reshape(tf.stack(prediction, axis=1), [self.args.batch_size, self.args.seq_len])

		# Define loss : standard cross entropy loss, need to ignore '-1' label example
		# Make target/label 1-d array
		target_1d = tf.reshape(self.target, [-1])
		pred_logits_1d = tf.reshape(self.pred_logits, [-1])
		index = tf.where(tf.not_equal(target_1d, tf.constant(-1., dtype=tf.float32)))
		# tf.gather(params, indices) : Gather slices from params according to indices
		filtered_target = tf.gather(target_1d, index)
		filtered_logits = tf.gather(pred_logits_1d, index)

		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_target))
		self.pred = tf.sigmoid(self.pred_logits)

		# Optimizer : SGD + MOMENTUM with learning rate decay
		self.global_step = tf.Variable(0, trainable=False)
		self.lr = tf.placeholder(tf.float32, [], name='learning_rate')

		optimizer = tf.train.MomentumOptimizer(self.lr, self.args.momentum)
		grads, vrbs = zip(*optimizer.compute_gradients(self.loss))
		grad, _ = tf.clip_by_global_norm(grads, self.args.maxgradnorm)
		self.train_op = optimizer.apply_gradients(zip(grad, vrbs), global_step=self.global_step)
		self.tr_vrbs = tf.trainable_variables()
		self.params = {}
		for i in self.tr_vrbs:
			print(i.name)
			self.params[i.name] = tf.get_default_graph().get_tensor_by_name(i.name)
		self.saver = tf.train.Saver()

	def getParam(self):
		"""
		Get parameters of DKVMN-CA model
		:return:
		"""
		params = self.sess.run([self.params])
		with open('good_future.pkl', 'wb') as f:
			pickle.dump(params, f)



	def train(self, train_q_data, train_qa_data, valid_q_data, valid_qa_data, train_kg_data,
			  valid_kg_data, train_kgnum_data, valid_kgnum_data, traintime, validtime, trainguan, validguan, traindiff, validdiff):
		"""

		:param train_q_data: exercises ID
		:param train_qa_data: exercises ID and answer result
		:param train_kg_data: one-hot form of knowledge concepts
		:param train_kgnum_data: knowledge concepts tags
		:param traintime: completion time
		:param trainguan: the gate of exercise
		:param traindiff: the difficulty of exercise
		"""
		shuffle_index = np.random.permutation(train_q_data.shape[0])
		q_data_shuffled = train_q_data[shuffle_index]
		qa_data_shuffled = train_qa_data[shuffle_index]
		kg_shuffled = train_kgnum_data[shuffle_index]
		kghot_shuffled = train_kg_data[shuffle_index]
		time_shuffled = traintime[shuffle_index]
		guan_shuffled = trainguan[shuffle_index]
		diff_shuffled = traindiff[shuffle_index]
		training_step = train_q_data.shape[0] // self.args.batch_size
		self.sess.run(tf.global_variables_initializer())



		self.train_count = 0
		if self.args.init_from:
			if self.load():
				print('Checkpoint_loaded')
			else:
				print('No checkpoint')
		else:
			if os.path.exists(os.path.join(self.args.checkpoint_dir, self.model_dir)):
				try:
					shutil.rmtree(os.path.join(self.args.checkpoint_dir, self.model_dir))
					shutil.rmtree(os.path.join(self.args.log_dir, self.mode_dir+'.csv'))
				except(FileNotFoundError, IOError) as e:
					print('[Delete Error] %s - %s' % (e.filename, e.strerror))

		best_valid_auc = 0

		# Training
		for epoch in range(0, self.args.num_epochs):
			if self.args.show:
				bar.next()

			pred_list = list()
			target_list = list()
			epoch_loss = 0


			for steps in range(training_step):
				# [batch size, seq_len]
				q_batch_seq = q_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
				qa_batch_seq = qa_data_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
				kg_batch_seq = kg_shuffled[steps*self.args.batch_size:(steps+1)*self.args.batch_size, :]
				kghot_batch_seq = kghot_shuffled[steps * self.args.batch_size:(steps + 1) * self.args.batch_size, :]
				time_batch_seq = time_shuffled[steps * self.args.batch_size:(steps + 1) * self.args.batch_size, :]
				guan_batch_seq = guan_shuffled[steps * self.args.batch_size:(steps + 1) * self.args.batch_size, :]
				diff_batch_seq = diff_shuffled[steps * self.args.batch_size:(steps + 1) * self.args.batch_size, :]
				# qa : exercise index + answer(0 or 1)*exercies_number
				# right : 1, wrong : 0, padding : -1
				target = qa_batch_seq[:,:]
				# Make integer type to calculate target
				target = target.astype(np.int)
				target_batch = (target - 1) // self.args.n_questions
				target_batch = target_batch.astype(np.float)

				feed_dict = {self.kg:kg_batch_seq, self.q_data:q_batch_seq, self.qa_data:qa_batch_seq, self.target:target_batch,
							 self.kg_hot:kghot_batch_seq, self.lr:self.args.initial_lr, self.timebin:time_batch_seq, self.diff:diff_batch_seq,
							 self.guan: guan_batch_seq}


				loss_, pred_, _, = self.sess.run([self.loss, self.pred, self.train_op], feed_dict=feed_dict)

				right_target = np.asarray(target_batch).reshape(-1,1)
				right_pred = np.asarray(pred_).reshape(-1,1)

				right_index = np.flatnonzero(right_target != -1.).tolist()



				pred_list.append(right_pred[right_index])
				target_list.append(right_target[right_index])

				epoch_loss += loss_



			if self.args.show:
				bar.finish()

			all_pred = np.concatenate(pred_list, axis=0)
			all_target = np.concatenate(target_list, axis=0)

			# Compute metrics
			self.auc = metrics.roc_auc_score(all_target, all_pred)
			# Extract elements with boolean index
			# Make '1' for elements higher than 0.5
			# Make '0' for elements lower than 0.5
			all_pred[all_pred > 0.5] = 1
			all_pred[all_pred <= 0.5] = 0


			self.accuracy = metrics.accuracy_score(all_target, all_pred)

			epoch_loss = epoch_loss / training_step
			print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f' % (epoch+1, self.args.num_epochs, epoch_loss, self.auc, self.accuracy))
			self.write_log(epoch=epoch+1, auc=self.auc, accuracy=self.accuracy, loss=epoch_loss, name='training_')

			valid_steps = valid_q_data.shape[0] // self.args.batch_size
			valid_pred_list = list()
			valid_target_list = list()
			for s in range(valid_steps):
				# Validation
				valid_q = valid_q_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
				valid_qa = valid_qa_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
				valid_kg = valid_kgnum_data[s*self.args.batch_size:(s+1)*self.args.batch_size, :]
				valid_hot_kg = valid_kg_data[s * self.args.batch_size:(s + 1) * self.args.batch_size, :]
				valid_time = validtime[s * self.args.batch_size:(s + 1) * self.args.batch_size, :]
				valid_guan = validguan[s * self.args.batch_size:(s + 1) * self.args.batch_size, :]
				valid_diff = validdiff[s * self.args.batch_size:(s + 1) * self.args.batch_size, :]
				# right : 1, wrong : 0, padding : -1
				valid_target = (valid_qa - 1) // self.args.n_questions
				valid_feed_dict = {self.kg: valid_kg, self.q_data : valid_q, self.qa_data : valid_qa,
								   self.kg_hot: valid_hot_kg, self.target : valid_target, self.timebin:valid_time,
								   self.guan: valid_guan, self.diff:valid_diff}
				valid_loss, valid_pred = self.sess.run([self.loss, self.pred], feed_dict=valid_feed_dict)
				# Same with training set
				valid_right_target = np.asarray(valid_target).reshape(-1,)
				valid_right_pred = np.asarray(valid_pred).reshape(-1,)
				valid_right_index = np.flatnonzero(valid_right_target != -1).tolist()
				valid_target_list.append(valid_right_target[valid_right_index])
				valid_pred_list.append(valid_right_pred[valid_right_index])

			all_valid_pred = np.concatenate(valid_pred_list, axis=0)
			all_valid_target = np.concatenate(valid_target_list, axis=0)

			valid_auc = metrics.roc_auc_score(all_valid_target, all_valid_pred)
			# For validation accuracy
			stop = 0
			all_valid_pred[all_valid_pred > 0.5] = 1
			all_valid_pred[all_valid_pred <= 0.5] = 0
			valid_accuracy = metrics.accuracy_score(all_valid_target, all_valid_pred)
			print('Epoch %d/%d, valid auc : %3.5f, valid accuracy : %3.5f' %(epoch+1, self.args.num_epochs, valid_auc, valid_accuracy))
			# Valid log
			self.write_log(epoch=epoch+1, auc=valid_auc, accuracy=valid_accuracy, loss=valid_loss, name='valid_')

			if valid_auc > best_valid_auc:
				print('%3.4f to %3.4f' % (best_valid_auc, valid_auc))
				best_valid_auc = valid_auc
				best_acc = valid_accuracy
				best_epoch = epoch + 1
				# self.save(best_epoch)
			else:
				if epoch - best_epoch >= 2:


					with open(self.args.dataset+'concat', 'a') as f:
						f.write('auc:'+str(best_valid_auc)+',acc:'+str(best_acc)+'\n')
					self.args.count += 1
					break


	@property
	def model_dir(self):
		return '{}_{}batch_{}epochs'.format(self.args.dataset+str(self.args.count), self.args.batch_size, self.args.num_epochs)

	def load(self):
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.train_count = int(ckpt_name.split('-')[-1])
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def save(self, global_step):
		model_name = 'DKVMN'
		checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)
		self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
		print('Save checkpoint at %d' % (global_step+1))

	# Log file
	def write_log(self, auc, accuracy, loss, epoch, name='training_'):
		log_path = os.path.join(self.args.log_dir, name+self.model_dir+'.csv')
		if not os.path.exists(log_path):
			self.log_file = open(log_path, 'w')
			self.log_file.write('Epoch\tAuc\tAccuracy\tloss\n')
		else:
			self.log_file = open(log_path, 'a')

		self.log_file.write(str(epoch) + '\t' + str(auc) + '\t' + str(accuracy) + '\t' + str(loss) + '\n')
		self.log_file.flush()

