import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16

class VggNet(models.Model):
    """
    加载预训练VGG16网络作为StyleTransfer特征提取网络
    """
    def __init__(self, model_path, **kwargs):
        """
        model_path: (None, imagenet, model_path)
        """
        super(VggNet, self).__init__(**kwargs)
        self.model_path = model_path
        vgg16 = VGG16(weights=model_path, include_top=False)
        print('Model Loading Finished!!!')
        output_layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'block4_conv2', 'block5_conv3']
        output_layers = [vgg16.get_layer(name).output for name in output_layer_names]
        # build feature network
        self.feature = models.Model(inputs=vgg16.input, outputs=output_layers)
        # print(self.feature.summary())

    def call(self, xs, training=True):
        outputs = self.feature(xs)
        return outputs

if __name__ == "__main__":
    import skimage.io as io
    import matplotlib.pyplot as plt
    from skimage import transform
    
    xs = io.imread('dog.jpg')
    xs = transform.resize(xs, (224, 224))
    # convert to BGR
    # xs = tf.keras.applications.vgg16.preprocess_input(xs)
    # print(np.min(xs), np.max(xs))
    # xs += np.array([[[103.939, 116.779, 123.68]]])
    
    xs = tf.convert_to_tensor(xs, dtype=tf.float32)
    
    xs = tf.expand_dims(xs, axis=0)
    model = VggNet(model_path='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    outputs = model(xs, training=False)
    for output in outputs:
        # print(output.shape)
        plt.figure()
        output = output[0]
        fig = tf.reduce_mean(output,axis=[-1]).numpy()
        io.imshow(fig)
        io.show()
    
    