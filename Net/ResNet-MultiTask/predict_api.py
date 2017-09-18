# -*- coding=utf-8 -*-
from resnet import *
import tensorflow as tf
import sys
from DataSet.DataSetBridger import generate_batch_data
from tools.ImageProcessing import vec2txt, copy_result_to
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
tf.app.flags.DEFINE_integer('max_single_vc_length', 36, 'the length of vc code, contain 0-9 a-z')
tf.app.flags.DEFINE_integer('letter_num_per_vc', 4, 'the count of letter for per vc')
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


'''
    识别一个文件夹想下面所有的图片
'''
def predict_images_api(image_dir):
    image_pathes = [os.path.join(image_name) for image_name in os.listdir(image_dir)]
    batch_images, batch_labels = generate_batch_data(image_dir, num_processing_threads=1,num_epochs=1, shuffle=False)
    logits_multi_task = []
    for i in range(FLAGS.letter_num_per_vc):
        logits = inference(batch_images,
                           task_name='task_' + str(i),
                           num_classes=FLAGS.max_single_vc_length,
                           is_training=True,
                           bottleneck=False, )
        logits_multi_task.append(logits)
    is_training = tf.placeholder('bool', [], name='is_training')
    print 'batch labels shapie is ', batch_labels
    predictes_res, batch_label_values = predict_vc(is_training, logits_multi_task, batch_labels)
    shape = list(np.shape(predictes_res))
    predictes_res = np.array(predictes_res)
    res_dict = {}
    for sample_id in range(shape[0]):
        image_path = image_pathes[sample_id]
        print 'cur image name is ', os.path.basename(image_path)
        print batch_label_values[sample_id]
        vc_txt = vec2txt(predictes_res[sample_id])
        print vc_txt
        res_dict[os.path.basename(image_path)] = vc_txt
        print '\n'
    return res_dict

'''
    对于图片进行预测分类
'''
def predict_vc(is_training, logits, batch_label):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    predictions = []
    for i in range(FLAGS.letter_num_per_vc):
        predictions.append(tf.nn.softmax(logits[i]))

    saver = tf.train.Saver(tf.all_variables())


    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)


    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.save_model_path)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)
    step = sess.run(global_step)
    ops = []
    ops.extend(predictions)
    ops.append(batch_label)
    predictions_value = []
    batch_label_value = []
    while True:
        try:
            ops_res = sess.run(ops, { is_training : False})
            print np.shape(ops_res[:4])
            predictions_value_single = np.concatenate(ops_res[:4], axis=1)
            predictions_value.extend(predictions_value_single)
            batch_label_value.extend(ops_res[4])
        except Exception, error:
            print 'error msg is ', error.message
            break
    print 'predictions_value shape is ', np.shape(predictions_value)
    print 'predictions_label shape is ', np.shape(batch_label_value)
    return predictions_value, batch_label_value

if __name__ == '__main__':
    res_dict = predict_images_api('/home/give/Documents/dataset/VerficationCode/Captcha/test_images_gray')
    copy_result_to(
        result_dict=res_dict,
        org_dir='/home/give/Documents/dataset/VerficationCode/Captcha/test_images_gray',
        target_dir='/home/give/Documents/dataset/VerficationCode/Captcha/predict_result'
    )