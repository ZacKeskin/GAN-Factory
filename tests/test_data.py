import glob
import os
import tensorflow as tf

# Unit test to ensure each image can pass through the TF jpg decorder
def test_imagetype():
  base_path = os.getcwd()

  for i, image_name in enumerate(glob.glob(os.path.join(base_path, 'target',  '*'))):
      print(i, image_name)
      with tf.Graph().as_default():
        image_contents = tf.read_file(image_name)
        image = tf.image.decode_image(image_contents, channels=3)
        init_op = tf.tables_initializer()
        with tf.Session() as sess:
          sess.run(init_op)
          tmp = sess.run(image)
