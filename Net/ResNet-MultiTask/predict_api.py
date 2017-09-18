# -*- coding=utf-8 -*-
from resnet import *
import tensorflow as tf
import sys
from DataSet.DataSetBridger import generate_batch_data
MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/resnet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('save_model_path', './models', 'the saving path of the model')
tf.app.flags.DEFINE_string('log_dir', './log/train',
                           """The Summury output directory""")
tf.app.flags.DEFINE_string('log_val_dir', './log/val',
                           """The Summury output directory""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


'''
    识别一个文件夹想下面所有的图片
'''
def predict_images_api(image_dir):
    batch_images, batch_labels = generate_batch_data(image_dir)
    logits = inference(batch_images,
                       num_classes=36 * 4,
                       is_training=True,
                       bottleneck=False, )
    is_training = tf.placeholder('bool', [], name='is_training')
    predict_res = predict_vc(is_training, logits)
    label = np.argmax(predict_res)
    print predict_res

'''
    对于图片进行预测分类
'''
def predict_vc(is_training, logits):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    predictions = tf.nn.softmax(logits)


    saver = tf.train.Saver(tf.all_variables())


    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)


    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.save_model_path)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)
    step = sess.run(global_step)

    predictions_value = sess.run(predictions, { is_training: False })
    print 'predictions_value shape is ', np.shape(predictions_value)
    return predictions_value

if __name__ == '__main__':
    predict_images_api('/home/give/Documents/dataset/VerficationCode/Captcha/test_images_gray')