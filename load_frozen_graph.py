import tensorflow as tf
import numpy as np
import cv2
import time


class CNN(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        print('[INFO]: Loading model...')
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.InteractiveSession(graph=self.graph)
        self.load_graph(model_filepath=self.model_filepath)

    def load_graph(self, model_filepath):

        with tf.compat.v1.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        # Define input tensor
        self.input = tf.compat.v1.placeholder(np.float32, shape=[None, 224, 224, 3], name='input')
        self.output = tf.compat.v1.placeholder(tf.float32, shape=[], name='output')

        tf.import_graph_def(graph_def, {'input_1': self.input, 'output': self.output})

        print('Model loading complete!')

        # Get layer names
        # layers = [op.name for op in self.graph.get_operations()]
        # for layer in layers:
        #     print(layer)


        '''
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            print("Value - " )
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        '''

    def test(self, data):

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/output:0")
        output = self.sess.run(output_tensor, feed_dict={self.input: data, self.output: 0})

        return output


if __name__ == '__main__':
    model = CNN('model.pb')
    image = cv2.imread('test.jpg')
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    t1 = time.perf_counter()
    print(model.test(image))
    t2 = time.perf_counter()
    print(f'time: {t2 - t1}')
