# -*- coding=utf-8 -*-
from resnet import *
import tensorflow as tf
import sys
from DataSet.DataSetBridger import generate_batch_data
from tools.ImageProcessing import vec2txt, copy_result_to, cal_accuracy
from PIL import Image
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
tf.app.flags.DEFINE_integer('batch_size', 256, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_integer('max_single_vc_length', 36, 'the length of vc code, contain 0-9 a-z')
tf.app.flags.DEFINE_integer('letter_num_per_vc', 4, 'the count of letter for per vc')
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


'''
    识别一个文件
    :param image_path 文件的路径
'''
def predict_image_api(image_path, show=False):
    image_pathes = [image_path]
    if show:
        img = Image.open(image_path)
        img.show()
    print predict_images_api(image_pathes, is_pathes=True)


'''
    识别一个文件夹想下面所有的图片
    :param image_dir 文件夹的路径
    :param has_label 就是文件的名字是否可以作为ｌａｂｅｌ
    :param is_pathes　决定了ｉｍａｇｅ_ｄｉｒ是文件夹还是图像路劲的数组
'''
def predict_images_api(image_dir, has_label=False, is_pathes=False):
    if is_pathes:
        image_pathes = image_dir
    else:
        image_pathes = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
    batch_images, batch_labels = generate_batch_data(image_dir, batch_size=FLAGS.batch_size, num_processing_threads=1,num_epochs=1, shuffle=False, is_pathes=is_pathes)
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
    label_txt = []
    predict_txt = []
    for sample_id in range(shape[0]):
        image_path = image_pathes[sample_id]
        vc_txt = vec2txt(predictes_res[sample_id])
        label_txt.append(vec2txt(batch_label_values[sample_id]))
        predict_txt.append(vc_txt)
        res_dict[os.path.basename(image_path)] = vc_txt
    if has_label:
        print label_txt
        print predict_txt
        print cal_accuracy(labels=label_txt, predict_res=predict_txt)
    return res_dict

'''
    预测验证码的值，没有ｌａｂｅｌ
'''
def predict_vc_nonlabel(image_tensor, image_value, logits):
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
    predictions_value = []

    try:
        ops_res = sess.run(ops, feed_dict={image_tensor: image_value})
        predictions_value_single = np.concatenate(ops_res[:4], axis=1)
        predictions_value.extend(predictions_value_single)
    except Exception, error:
        print 'error msg is ', error.message
    print 'predictions_value shape is ', np.shape(predictions_value)
    return predictions_value

'''
    对于图片进行预测分类
'''
def predict_vc(is_training, logits, batch_label=None):
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
    # predict_dir = '/home/give/Documents/dataset/VerficationCode/Captcha/new_dataset/data'
    predict_dir = '/home/give/Documents/dataset/VerficationCode/Captcha/test_images_gray'
    save_dir = '/home/give/Documents/dataset/VerficationCode/Captcha/predict_result'
    res_dict = predict_images_api(predict_dir, has_label=True)
    copy_result_to(
        result_dict=res_dict,
        org_dir=predict_dir,
        target_dir=save_dir
    )
    # ['2da0', '3g13', 'd21r', '8703', 'deh1', '3m2w', 'i8xi', '8v23', '3747', '7395', '1316', 'h394', 'sn35', 'k409',
    #  'd245', 'r0z8', '73l1', 'co9l', 'rw18', 'ma0t', '6x13', '26zz', '3836', '2513', '3be2', '7w88', 'r9zd', '6549',
    #  '5at4', 'j9go', 'csg2', '8vz9', '9o73', 'p89f', '7z3p', '6264', '24eh', '5q5r', '6w64', '998g', '4x91', '8b8i',
    #  '2790', '3856', 'jd46', '21yq', '053r', '75rp', '97pf', 'dx95', '1356', 'hlo1', '49sz', '3835', 'ghl8', '1539',
    #  'mt67', '7352', '6175', '2xt8', '8699', '1558', '6mhk', '12kb', 'k32v', '6v9u', 'rxz1', '2898', 'o948', 'm467',
    #  '4h7p', '4g39', 'q454', '22y3', '6f34', '5320', '46n2', 'd129', '56g8', '0327', 's45v', '2i1j', '89n8', '4503',
    #  '6q39', 'c794', '07i0', '303q', '3665', '44d3', '8k70', '8t78', 'egdk', '7p68', '1e1l', '0y7q', 'so24', '4113',
    #  '8212', '7227', 'tr1k', 'izc4', '5s49', '1700', 'ah75', 'm54u', 'f558', 'h96q', '7ba8', '740a', 'u5ao', '1609',
    #  '58f0', '9z36', '24jo', '084h', '0840', '5163', 'u199', '66b1', '1748', 'r1r4', 'n9k4', '17g4', '10u5', '9100',
    #  '1y7u', '6k94', 'w716', '8zil', 'lxib', '56p0', '44u1', '7732', '0y10', '8137', 'w931', 'j37u', '298j', '33c1',
    #  '676l', '12t6', '07r2', 'n220', 'ze75', 'v029', '456m', 'p63g', '61kv', 'v6li', '8237', 'en74', '225i', '9406',
    #  '21w5', '51n8', 'gt7h', '4067', '46j1', '169s', 'v4u3', '7v33', '2yp4', '20q2', '3275', 'bldt', '78c2', 'g288',
    #  '9303', 'tk72', '2839', 'ce6t', 'n02z', 'yoeh', 'x060', 'f6x5', '3645', '95j9', '640o', '5795', '6739', '5783',
    #  '24n3', '0e63', '1183', '7u58', 'a463', '6550', '221c', '6730', '798y', '70pe', '9920', '489c', 'm50l', '6c49',
    #  'z60n', 'x306', '3219', '108s', 'y7zo', 'w742', 'z9i0', '4hg7', '793g', '1j0v', '1o09', 're1n', '8f9c', '3h57',
    #  '7p56', '09x0', '652t', '444o', '1o8l', '3hsm', 'km37', 'q118', '288y', '5x89', '693k', 'vjtj', '11r9', 'u17d',
    #  '7au3', '9254', 'em3b', '5f9p', '31p8', '31p1', '5203', '8487', '00ek', 'l981', 'p4c5', '83x2', '208x', '584o',
    #  'u63b', 'c0tj', '19u7', '9uw3', '38b4', '1171', '2xgj', '9gd7', 'v885', 't31l', '5ak2', '3010', 'd9u1', 'qb9v',
    #  '36vz', 'z7o4', '4213', '9f1l', '76i5', '0e4h', '0136', '4gm3', 'pxq2', '2102', '2o3d', 'h379', '963n', '63fe',
    #  'j918', '6116', '83l7', 'g552', 'a3j6', '391t', '2cl3', '3n00', '2500', '9f01', 'xw9g', '8d2s', '3nyq', '4691',
    #  '73s3', 'a264', '2i30', '3022', 'p9x4', '579z', '2i28', 's7n1', '507c', 'co0n', '7889', '235a', '3206', '766t',
    #  '8642', '5209', '1546', 'p6u8', '13bq', '6454', 'p823', '8l2t', '4n72', '9z4u', '7091', '1692', 'qm25', '4m36',
    #  'z594', 'j84d', 'fq5p', '4w4o', 'uck7', 'ga15', 'ly46', '1684', 'yn23', '4888', '3bni', '3925', 'j646', '1idu',
    #  '598r', 'n59i', 'pou3', '0030', '684k', 'q31x', '5985', '489m', '1m6m', '24s4', 'tz47', '2063', 'w40a', '4748',
    #  '59ta', '1s93', 'hk08', 'y7e3', '93zn', 'k166', 'tqj5', 'e0vj', 'mp48', 'a928', 'zx68', '406r', '22d4', 'c5y1',
    #  '3h6z', 'b882', 'f695', '90sl', 'yh03', 'f9rz', '6086', 'gi4q', '4rq6', 't456', '9954', '9932', '2em2', '25q2',
    #  '1f3d', '3221', '1c87', '5g79', '7972', '046j', '41f3', '4066', 'x42o', '8877', '07p1', '7g00', 'n343', '90u9',
    #  'qz8s', '5u57', '5qt9', '61z4', '8284', 'i5p3', '230q', '96h7', '7vt8', '07gh', 'w648', 'e377', '53ti', '36m7',
    #  '0419', 'oub0', '7820', '77gl', '4fwg', '9tq1', '88ga', 'r432', '7v21', 'w277', '9j69', '0302', 'mfjn', '78n5',
    #  'e8j2', '1e0a', 'iapp', '49du', '365b', 'x9q3', 'l3q5', '4l4x', '691n', '87qc', '75x0', '3es7', '26z9', 'rphe',
    #  '408e', '5318', '9606', 'p368', '1ib4', '675l', '3408', '1no3', '57m6', '1o57', '6wck', '5mv2', '942c', 'o8gr',
    #  '5067', 've59', 'n168', '6g2l', '961u', '1832', '7940', '0yrp', '76y8', 'tc3b', 'lpq4', 'yd87', 's8v8', '0137',
    #  '2oof', 'yy11', '5560', 'h9t6', '71b2', 'b9c4', '4wsm', 'z408', 'ygm0', '0e25', '9896', '5r9x', '3746', '0315',
    #  'sgv2', '62wl', '089a', '2cu5', '2495', '3v47', '9137', 'i0kl', '8p64', '7220', '3544', '4078', 'gos0', 't478',
    #  'zr66', '1378', '6k8e', '61qm', '778v', '04aq', 'rf40', '8j99', '68t9', 'ux3i', '93q5', 'dv29', '012y', '6508',
    #  '9t89', 'h9b0', 'w305', 'zy7w', '7845', 'm1e2', 'ro68', 'l721', '8p7a', '5hc3', '7208', '1981', '46s3', '1439',
    #  '9649', '7oj3', '39w4', '1p92', '0i48', '2x75', '671i', 'dd1s', '7g68', 'i194', 'c494', '8889', 'ph08', 'y452',
    #  '27ch', '9z24', '5510', 'c73m', '612c', '95e6', '1im2', '3824', 'b081', '78oy', 'x26p', '4dk8', '6q6t', '02ly',
    #  '5960', '9053', 'j9g2', '12b9', '5me4', 'p87v', '3601', '8j0j', 'a12f', '4nf7', '4wh2', '779g', '585s', '1s0d',
    #  'x1i2', '7pvq', 'u8xt', '3n22', 'm13q', '8747', '0s15', 'g115', 'v908', '3421', 'd062', '3360', '3040', '4rho',
    #  '7450', '034m', '6x9p', '5285', '80a9', 'ig8g', '1c3d', '7j11', 'tj19', '92bd', '8p88', '90d1', 'bd44', '8181',
    #  'ap2p', 'a952', '61i6', '321b', 'v2yw', '8b98', '0k8y', 'k8um', '04m1', '4876', '8fle', '6wyh', 'le29', '0y62',
    #  '90i9', '4l51', '0e37', '657t', '6491', 't319', 'ynzj', '75c7', '5lyg', '97q1', '9tf0', '8692', '94bc', '099s',
    #  'u05h', '8414', 'm8e0', 'lq47', '5646', 'mhwq', '92y1', '1c99', 'o0ba', '8vir', '2j15', '321k', '4sw1', '856e',
    #  '282f', '05c2', '32v0', '4x03', 'o770', '672v', '003z', 'qu5y', 'm925', '674e', '8402', 'v9u2', '7486', '2613',
    #  '9pl0', 'lw20', 'b9fu', '1y68', '5973', '4i4z', 'w5h3', 'tw57', '3641', '6551', 'm504', '95f5', '3e52', 's55w',
    #  'v4t7', '4781', 'uusc', 'n367', '6e2k', '8hr8', '8t8u', 'o187', 'm7w3', '5g27', '6701', 'm96r', '0s87', 'g45r',
    #  'h033', 'u976', '526k', '4646', '603c', '540x', '9ye9', '360l', '5w9i', '002s', '07x7', '60r6', '7br6', 'wm1b',
    #  '62t4', 'f8jy', 'x3r7', '2088', '63o4', 'x75c', '22uw', '0p7x', 'r271', 'i899', '993w', '32wt', 'e772', '0076',
    #  '7132', '73wp', '5508', 'i78o', '5k8a', 'qnl8', '2x13', '4ff6', '3372', 'h23w', 'e178', '6673', 'v793', 'zqa3',
    #  'y35i', '95of', 'u472', 'zk60', '29h5', '9184', '7946', 'd410', '207q', '8d54', 'h07x', '1z93', '75a1', '5s77']
    # ['2da0', '3g13', 'o21r', '8703', 'deh1', '3m2w', 'l8xi', '8v23', '3747', '7395', '1316', 'h394', 'sn35', 'k409',
    #  '0245', 'r0z8', '73l1', 'c09l', 'rw18', 'ma0t', '6x13', '26zz', '3836', '2513', '3be2', '7w88', 'r9zd', '6549',
    #  '5at4', 'j9go', 'csg2', '8vz9', '9o73', 'p89f', '7z3p', '6264', '24eh', '5q5r', '6w64', '998g', '4x91', '8b8i',
    #  '2790', '3856', 'jd46', '21yq', '053r', '75rp', '97pf', 'dx95', '1356', 'hlo1', '49sz', '3835', 'ghl8', '1539',
    #  'mt67', '7352', '6175', '2xt8', '8699', '1558', '6mhk', '12kb', 'k32v', '6v9u', 'rxz1', '2898', '0948', 'm467',
    #  '4h7p', '4g39', 'q454', '22y3', '6f34', '5320', '46n2', 'd129', '56g8', '0327', 's45v', '2i1j', '89n8', '4503',
    #  '6q39', 'c794', '07i0', '303q', '3665', '44d3', '8k70', '8t78', 'egdk', '7p68', '1e1l', '0y7q', 's024', '4113',
    #  '8212', '7227', 'tr1k', 'izc4', '5s49', '1700', 'ah75', 'm54u', 'f558', 'h96q', '7ba8', '740y', 'u5ao', '1609',
    #  '58f0', '9z36', '24jo', '084h', '0840', '5163', 'u199', '66b1', '1748', 'r1r4', 'n9k4', '17g4', '10u5', '9100',
    #  '1y7u', '6k94', 'w716', '8z1l', 'lxib', '56p0', '44u1', '7732', '0y10', '8137', 'w931', 'j37u', '298j', '33c1',
    #  '676l', '12t6', '07k2', 'n220', 'ze75', 'v029', '456m', 'p63g', '61kv', 'v6lt', '8237', 'en74', '225i', '94o6',
    #  '21w5', '51n8', 'gt7h', '4067', '46j1', '169s', 'v4u3', '7v33', '2yp4', '20q2', '3275', 'bldt', '78c2', 'g288',
    #  '9303', 'tk72', '2839', 'oe6t', 'n02z', 'yoeh', 'x060', 'f6x5', '3645', '95j9', '640o', '5795', '6739', '5783',
    #  '24n3', '0e63', '1183', '7u58', 'a463', '6550', '221c', '6730', '798y', '70pe', '9920', '489c', 'm50l', '6c49',
    #  'z60n', 'x306', '3219', '108s', 'y7zo', 'w742', 'z9i0', '4hg7', '793g', '1jov', '10o9', 're1n', '8f9c', '3h57',
    #  '7p56', '09x0', '652t', '444o', '108l', '3hsm', 'km37', 'q118', '288y', '5x89', '693k', 'vjtj', '11r9', 'u17d',
    #  '7au3', '9254', 'em3b', '5f9p', '31p8', '31p1', '5203', '8487', '00ek', 'l981', 'p4c5', '83x2', '208x', '584o',
    #  'u63b', 'c0tj', '19u7', '9uw3', '38b4', '1771', '2xgj', '9gd7', 'v885', 't31l', '5ak2', '3010', 'd9u1', 'qb9v',
    #  '36vz', 'z7o4', '4213', '9f1l', '76i5', '0e4h', '0136', '4gm3', 'pxq2', '21o2', '203d', 'h379', '963n', '63fe',
    #  'j918', '6116', '83l7', 'g552', 'a3j6', '391t', '2cl3', '3z00', '2500', '9f01', 'xw9g', '8d2s', '3nyq', '4691',
    #  '73s3', 'a264', '2i30', '3022', 'p9x4', '579z', '2i28', 's7n1', '507c', 'c0on', '7889', '235a', '3206', '766t',
    #  '8642', '5209', '1546', 'p6u8', '13bq', '6454', 'p823', '8l2t', '4n72', '9z4u', '7091', '1692', 'qm25', '4m36',
    #  'z594', 'j84d', 'fq5p', '4w4o', 'uck7', 'ga15', 'ly46', '1684', 'yn23', '4888', '38ni', '3925', 'j646', '1idu',
    #  '598r', 'n59i', 'p0u3', '0030', '684k', 'o31x', '5985', '489m', '1m6m', '24s4', 'tz47', '2063', 'w40a', '4748',
    #  '59ta', '1s93', 'hk08', 'y7e3', '93zn', 'k166', 'tqj5', 'e0vj', 'mp48', 'a928', 'zx68', '406r', '22d4', 'c5y1',
    #  '3h6z', 'b882', 'f695', '90sl', 'yh03', 'f9rz', '6086', 'gi4q', '4rq6', 't456', '9954', '9932', '2em2', '25q2',
    #  '1f3d', '3221', '1c87', '5q79', '7972', '046j', '41f3', '4066', 'x42o', '8877', '07p1', '7g00', 'n343', '90u9',
    #  'qz8s', '5u57', '5qt9', '61z4', '8284', 'i5p3', '230q', '96h7', '7vt8', '07gh', 'w648', 'e377', '53ti', '36m7',
    #  '0419', 'oub0', '7820', '77gl', '4fwg', '9tq1', '88ga', 'r432', '7v21', 'w277', '9j69', '0302', 'mfjn', '78n5',
    #  'e8j2', '1e0a', 'iapp', '49du', '365b', 'x9q3', 'l3q5', '4l4x', '691n', '87qc', '75x0', '3es7', '26z9', 'rphe',
    #  '408e', '5318', '96o6', 'p368', '1ib4', '675l', '3408', '1n03', '57m6', '1057', '6wck', '5mv2', '942c', 'o8gr',
    #  '5067', 've59', 'n168', '6g2l', '961u', 'i832', '7940', '0yrp', '76y8', 'tc3b', 'lpq4', 'yd87', 's8v8', '0137',
    #  '200f', 'yy11', '5560', 'h9t6', '71b2', 'b9c4', '4wsm', 'z408', 'ygm0', '0e25', '9896', '5r9x', '3746', '0315',
    #  'sgv2', '62wl', '089a', '2cu5', '2495', '3v47', '9137', 'i0kl', '8p64', '7220', '3544', '4078', 'g0s0', 't478',
    #  'zr66', '1378', '6k8e', '61qm', '778v', '04aq', 'rf40', '8j99', '68t9', 'ux3i', '93q5', 'dv29', '012y', '6508',
    #  '9t89', 'h980', 'w305', 'zy7w', '7845', 'm1e2', 'r068', 'l721', '8p7a', '5hc3', '7208', '1981', '46s3', '1439',
    #  '9649', '70j3', '39w4', '1p92', '0i48', '2x75', '671i', 'od1s', '7g68', 'i194', 'c494', '8889', 'ph08', 'y452',
    #  '27ch', '9z24', '5510', 'c73m', '612c', '95e6', '1im2', '3824', 'b081', '78oy', 'x26p', '4dk8', '6q6t', '02ly',
    #  '5960', '9053', 'j9g2', '12b9', '5me4', 'p87v', '3601', '8joj', 'a12f', '4nf7', '4wh2', '779g', '585s', '1s0d',
    #  'x1t2', '7pvq', 'u8xt', '3n22', 'm13q', '8747', '0s15', 'g115', 'v908', '3421', 'o062', '3360', '3040', '4rho',
    #  '7450', '034m', '6x9p', '5285', '80a9', 'ig8g', '1c3d', '7j11', 'tj19', '92bd', '8p88', '90d1', 'bd44', '8181',
    #  'ap2p', 'a952', '6116', '321b', 'v2yw', '8b98', '0k8y', 'k8um', '04m1', '4876', '8fle', '6wyh', 'le29', '0y62',
    #  '90i9', '4l51', '0e37', '657t', '6491', 't319', 'ynzj', '75c7', '5lyg', '97q1', '9tf0', '8692', '94bc', '099s',
    #  'u05h', '8414', 'm8e0', 'lq47', '5646', 'mhwq', '92y1', '1c99', 'o0ba', '8v1r', '2j15', '321k', '4sw1', '856e',
    #  '282f', '05c2', '32v0', '4x03', 'o770', '672v', '003z', 'qu5y', 'm925', '674e', '8402', 'v9u2', '7486', '2613',
    #  '9pl0', 'lw20', 'b9fu', '1y68', '5973', '4i4z', 'w5h3', 'tw57', '3641', '6551', 'm504', '95f5', '3e52', 's55m',
    #  'v4t7', '4781', 'uusc', 'n367', '6e2k', '8hr8', '8t8u', 'o187', 'm7w3', '5g27', '6701', 'm96r', '0s87', 'g45r',
    #  'h033', 'u976', '526k', '4646', '603c', '540x', '9ye9', '360l', '5w9i', '002s', '07x7', '60r6', '7br6', 'wm1b',
    #  '62t4', 'f8jy', 'x3r7', '2088', '63o4', 'x75c', '22uw', '0p7x', 'r271', 'i899', '993w', '32wt', 'e772', '0076',
    #  '7132', '73wp', '5508', '1780', '5k8a', 'qnl8', '2x13', '4ff6', '3372', 'h23w', 'e178', '6673', 'v793', 'zqa3',
    #  'y35i', '95of', 'u472', 'zk60', '29h5', '9184', '7946', 'd410', '207q', '8d54', 'h07x', '1z93', '75a1', '5s77']

    # print predict_image_api('/home/give/Documents/dataset/VerficationCode/Captcha/test_images_gray/2DA0.jpg', show=True)