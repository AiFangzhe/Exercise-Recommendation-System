import numpy as np
class DATA_LOADER():
	def __init__(self, number_AllConcepts, number_concepts, seqlen):
		"""
		Preprocessing data
		:param number_AllConcepts: the number of all unique knowledge concepts
		:param number_concepts: the number of knowledge concepts for an exercise
		:param seqlen: exercises sequence length
		"""
		self.number_AllConcepts = number_AllConcepts
		self.number_concepts = number_concepts
		self.seq_len = seqlen


	def load_dataes(self, q_data, qa_data, kg_data, kg_num, time, guan, diff):

		q_data_array = np.zeros((len(q_data), self.seq_len))
		for i in range(len(q_data)):
			data = q_data[i]
			q_data_array[i, :len(data)] = data

		qa_data_array = np.zeros((len(qa_data), self.seq_len))
		for i in range(len(qa_data)):
			data = qa_data[i]
			qa_data_array[i, :len(data)] = data

		kg_data_array = np.zeros((len(kg_data), self.seq_len, self.number_AllConcepts))
		for i in range(len(kg_data)):
			data = np.array(kg_data[i])
			kg_data_array[i, :len(data)] = data

		kgnum_data_array = np.zeros((len(kg_num), self.seq_len, self.number_concepts))
		for i in range(len(kg_num)):
			data = np.array(kg_num[i])
			kgnum_data_array[i, :len(data)] = data

		time_data_array = np.zeros((len(time), self.seq_len))
		for i in range(len(time)):
			data = np.array(time[i])
			time_data_array[i, :len(data)] = data
		time_data_array = time_data_array.astype(int)

		guan_data_array = np.zeros((len(guan), self.seq_len))
		for i in range(len(guan)):
			data = np.array(guan[i])
			guan_data_array[i, :len(data)] = data
		guan_data_array = guan_data_array.astype(int)

		diff_data_array = np.zeros((len(diff), self.seq_len))
		for i in range(len(diff)):
			data = np.array(diff[i])
			diff_data_array[i, :len(data)] = data
		diff_data_array = diff_data_array.astype(int)

		return q_data_array, qa_data_array, kg_data_array, kgnum_data_array,  time_data_array, guan_data_array, diff_data_array