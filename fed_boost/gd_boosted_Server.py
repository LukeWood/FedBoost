import h5py, random
import numpy as np
from fed_boost.server import Server
from scipy.optimize import line_search
from parameters import client_size, output_class_size, client_epochs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.utils import to_categorical

class GDBoostServer(Server):

    def get_objective(self,z):
        g = to_categorical(z,output_class_size)
        def objective(w):
            return (g-w)**2
        return objective

    def get_gradient(self,z):
        g = to_categorical(z,output_class_size)
        def gradient(w):
            return -2*(g-w)
        return gradient

    def __init__(self,alpha,path):
        print(f'Creating GDBoost Server')
        print(f'Initializing weak learners')
        super().__init__(alpha,path)
        self.af = np.full(client_size, 0.0)
        self.model = Sequential([
            Dense(output_class_size,input_shape=(output_class_size,)),
        ])
        self.model.compile(
            'adam',
            loss='lse',
            metrics=['accuracy'],
        )
        print(f'Weak learners are loaded')
        print(f'server model ready')

    def f(self,x):
        self.alpha_producer = np.linalg.pinv(np.array(list(map(lambda y: y.predict_without_softmax(x),self.weak_learners))).T)
        return np.sum(np.array(list(map(lambda y: y[1].predict_without_softmax(x)*self.af[y[0]],enumerate(self.weak_learners)))),axis=0)

    def w(self,x,z):
        f_x = self.f(x)
        result = np.exp(-1*np.abs(f_x - f_x[z])/2)
        result[z] = np.sum(result)
        return result

    def alpha(self,x,z):
        w = self.w(x,z)
        f = self.f(x)
        return np.min(f/w)

    def g(self,x,z):
        w_curr = self.w(x,z)
        return self.model.predict(np.array([w_curr]))[0]

    def calculate_w_matrix(self):
        self.w_matrix = []
        for i in range(len(self.train_labels)):
            x,z = self.train_images[i],self.train_labels[i]
            self.w_matrix.append(self.w(x,z))
        self.w_matrix = np.array(self.w_matrix)

    def train_g_model(self):
        self.history = self.model.fit(
            self.w_matrix,
            to_categorical(self.train_labels,output_class_size),
            epochs=client_epochs,
            validation_data=(self.test_images, to_categorical(self.test_labels,output_class_size)),
        )

    def one_iteration(self,v):
        self.calculate_w_matrix()
        self.train_g_model()
        random_test_index = random.sample(range(len(self.test_labels)),1)
        x,z = self.test_images[random_test_index], self.test_labels[random_test_index]
        result = line_search(self.get_objective(z),self.get_gradient(z),self.f(x),self.g(x,z))
        alpha = result[0]
        new_f = self.f(x) + v*alpha*self.g(x)
        self.af = np.matmul(self.alpha_producer,new_f)

    def predict(self,x):
        return np.argmax(self.f(x), axis=0)

    def save_server(self,path):
        h5f = h5py.File(f'{path}/gdboost_server_extra_data_al{self.data_alpha}.h5', 'w')
        h5f.create_dataset('alpha', data=self.af)
        h5f.create_dataset('history_iter', data=self.history_iter)
        h5f.close()
        self.model.save(f'{path}/gdboost_server_model_al{self.data_alpha}.h5')
        for i in range(len(self.weak_learners)):
            self.weak_learners[i].save_model(f'{path}/gdboost_server')

    def load_server(self,path):
        h5f = h5py.File(f'{path}/gdboost_server_extra_data_al{self.data_alpha}.h5','r')
        self.af = h5f['alpha'][:]
        self.history_iter = h5f['history_iter'][:]
        h5f.close()
        self.model.load_weights(f'{path}/gdboost_server_model_al{self.data_alpha}.h5')
        for i in range(len(self.weak_learners)):
            self.weak_learners[i].load_model(f'{path}/gdboost_server')


    def train(self,Nb,v):
        self.history_iter = []
        print(f'Training Server')
        for i in range(Nb):
            self.one_iteration(v)
            self.history_iter.append(self.history)
            print('Training done')
            print(f"Loss after {i} iteration is {self.history.history['loss']}")
        print(f'Training Server Done')
