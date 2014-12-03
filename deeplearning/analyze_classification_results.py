
NUM_UNIQUE_LABELS = 4


#================================================
# Main function
#================================================

def main():
	stats = Stats()
	results_file = open('tmp_class_results.csv', 'r') # Provide the input file name here
	for row in results_file:
		exp_label, pred_label = map(lambda x: int(x), row.split(','))
		stats.addExpAndPredLabelsPair(exp_label, pred_label)

	stats.computeConfusionMatrix()
	printSquareMatrix("confusion_matrix.csv", stats.confusion_matrix) # Provide the output file name here

#================================================
# Stats class
#================================================

class Stats():
	def __init__(self):
		self.exp_label_to_pred_label_freqs = createSquareMatrix(NUM_UNIQUE_LABELS)
		self.confusion_matrix = []

	def addExpAndPredLabelsPair(self, exp_label, pred_label):
		d = self.exp_label_to_pred_label_freqs[exp_label]
		d[pred_label] = d[pred_label] + 1

	def computeConfusionMatrix(self):
		matrix = createSquareMatrix(NUM_UNIQUE_LABELS)
		for (exp_label, freq_arr) in enumerate(self.exp_label_to_pred_label_freqs):
			total_freq_of_exp_label = float(sum(freq_arr))
			if total_freq_of_exp_label > 0:
				for pred_label in range(NUM_UNIQUE_LABELS):
					matrix[exp_label][pred_label] = freq_arr[pred_label] / total_freq_of_exp_label
			else:
				for pred_label in freq_arr:
					matrix[exp_label][pred_label] = 0

		self.confusion_matrix = matrix

#================================================
# Helper functions
#================================================

def createSquareMatrix(n):
	matrix = [[]] * n
	for i in range(n):
		matrix[i] = [0] * NUM_UNIQUE_LABELS

	return matrix

def printSquareMatrix(file_name, matrix):
	f = open(file_name, 'w')
	for row in matrix:
		row_str = reduce(lambda x,y: str(x) + "," + str(y), row) + "\n"
		f.write(row_str)

	f.close()

#================================================

main()