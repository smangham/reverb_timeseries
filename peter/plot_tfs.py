import tfpy
import pickle

print("Unpickling Ψ...")
picklefile = open('pickle_tf_sey_line_tf.pickle', 'rb')
tf_sey_line = pickle.load(picklefile)
picklefile.close()
picklefile = open('pickle_tf_qso_line_tf.pickle', 'rb')
tf_qso_line = pickle.load(picklefile)
picklefile.close()

tf_sey_line.plot(response_map=True, name='resp')
tf_qso_line.plot(response_map=True, name='resp')
