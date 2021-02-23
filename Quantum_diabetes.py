import qiskit
import numpy as np
from matplotlib import pyplot as plt
from qiskit.ml.datasets import ad_hoc_data
from qiskit.aqua import QuantumInstance
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import (ErrorCorrectingCode,AllPairs,OneAgainstRest)
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

feature_dim = 2
training_dataset_size = 20
testing_dataset_size =10
random_seed = 10598
shot = 10000

sample_Total, training_input, test_input, class_labels = ad_hoc_data(training_size=training_dataset_size,
                                                                     test_size=testing_dataset_size,
                                                                     gap=0.3,
                                                                     n=feature_dim,
                                                                     plot_data=True)
                                                                     
  
dataponints, class_to_label = split_dataset_to_data_and_labels(test_input)
print(class_to_label)


backend =BasicAer.get_backend('qasm_simulator')
feature_map = ZZFeatureMap(feature_dim,reps=2)
svm = QSVM(feature_map, training_input, test_input, None)
svm.random_seed = random_seed
quantum_instance = QuantumInstance(backend, shots=shot,
                                   seed_simulator = random_seed,
                                   seed_transpiler = random_seed)
result = svm.run(quantum_instance)


print("kernal matrix during the training")
kernel_matrix = result['kernel_matrix_training']
img = plt.imshow(np.asmatrix(kernel_matrix),
                 Interpolation ='nearest',
                 origin='upper',
                 cmap='bone_r')
plt.show()



predicted_labels = svm.predict(dataponints[0])
predicted_classes = map_label_to_class_name(predicted_labels, 
                                             svm.label_to_class)
print("ground truth : {}".format(dataponints[1]))
print("predictions:{}".format(predicted_labels) )
print("testing success ratio: ", result["testing_accuracy"])



from sklearn.model_selection import train_test_split
import pandas as pd
df= pd.read_csv("/content/drive/MyDrive/Datasets/diabetes.csv")
print(df.head())

feature_dim = 9
y = df["Outcome"]
x_dia = df[df["Outcome"]==1]
x_no_dia = df[df["Outcome"]==0]

x_dia_train, x_dia_test = train_test_split(x_dia)
x_no_dia_train, x_no_dia_test = train_test_split(x_no_dia)

x_dia = x_dia.drop(columns=["Outcome"], axis=1)
x_no_dia = x_no_dia.drop(columns=["Outcome"], axis=1)
print(x_dia.head())
print(x_no_dia.head())

x_train = {'1':np.asanyarray(x_dia_train),
           '0':np.asanyarray(x_no_dia_train)}
x_test = {'1':np.asanyarray(x_no_dia_train),
          '0':np.asanyarray(x_no_dia_test)}
          
backend =BasicAer.get_backend('qasm_simulator')
feature_map = ZZFeatureMap(feature_dim,reps=2)
svm = QSVM(feature_map, x_train, x_test, None)
svm.random_seed = random_seed
quantum_instance = QuantumInstance(backend = backend, shots=shot,
                                   seed_simulator = random_seed,
                                   seed_transpiler = random_seed)
result = svm.run(quantum_instance)

seed = 10598  # Setting seed to ensure reproducable results

feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
qsvm = QSVM(feature_map, x_train, x_test, data_pts[0])

num_qubits = 9

feature_map = SecondOrderExpansion(feature_dimension=num_qubits,depth=2,entanglement='full')

svm = QSVM(feature_map, training_data,testing_data) # Creation of QSVM

quantum_instance = QuantumInstance(backend,shots=shots,skip_qobj_validation=False)

print('Running....\n')

result = svm.run(quantum_instance) # Running the QSVM and getting the accuracy

data = np.array([[1.453],[1.023],[0.135],[0.266]]) #Unlabelled data

prediction = svm.predict(data,quantum_instance) # Predict using unlabelled data 

print('Prediction of Smoker or Non-Smoker based upon gene expression of CDKN2A\n')
print('Accuracy: ' , result['testing_accuracy'],'\n')
print('Prediction from input data where 0 = Non-Smoker and 1 = Smoker\n')
print(prediction)


backend =BasicAer.get_backend('qasm_simulator')
feature_map = ZZFeatureMap(feature_dim,reps=2)
svm = QSVM(feature_map, x_dia_train, x_dia_test, None)
svm.random_seed = random_seed
quantum_instance = QuantumInstance(backend = backend, shots=shot,
                                   seed_simulator = random_seed,
                                   seed_transpiler = random_seed)
result = svm.run(quantum_instance)
