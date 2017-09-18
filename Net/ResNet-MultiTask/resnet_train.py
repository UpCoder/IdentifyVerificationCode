# -*- coding=utf-8 -*-
from resnet import *
import tensorflow as tf
import sys

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
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 32, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 500000, "max steps")
tf.app.flags.DEFINE_integer('max_single_vc_length', 36, 'the length of vc code, contain 0-9 a-z')
tf.app.flags.DEFINE_integer('letter_num_per_vc', 4, 'the count of letter for per vc')
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')

'''
    计算准确率
'''
def cal_accuracy(predictions, labels):
    correct_equal_and = None
    for i in range(FLAGS.letter_num_per_vc):
        reshaped_labels = tf.reshape(labels, [-1, FLAGS.letter_num_per_vc, FLAGS.max_single_vc_length])
        correct_equal = tf.equal(
            tf.argmax(reshaped_labels[:, i], axis=1),
            tf.argmax(predictions[i], axis=1)
        )
        if correct_equal_and is None:
            correct_equal_and = correct_equal
        else:
            # only if all label are right, it can be right.
            correct_equal_and = tf.logical_and(correct_equal_and, correct_equal)
    accuracy = tf.reduce_mean(tf.cast(correct_equal_and, tf.float32))
    return accuracy


def train(is_training, logits, images, labels, save_model_path=None):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    loss_ = None
    predictions = []
    for i in range(FLAGS.letter_num_per_vc):
        start = i * FLAGS.max_single_vc_length
        end = (i+1) * FLAGS.max_single_vc_length
        if loss_ is None:
            # the ith logits 对应的ｌａｂｅｌ返回
            loss_ = loss(logits[i], labels[:, start:end])
        else:
            loss_ += loss(logits[i], labels[:, start:end])
        predictions.append(tf.nn.softmax(logits[i]))

    top1_error = cal_accuracy(predictions, labels)


    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        for var in tf.trainable_variables():
            tf.summary.image(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.all_variables())

    summary_op = tf.summary.merge_all()

    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(FLAGS.log_val_dir, sess.graph)
    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.save_model_path)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.train_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)

    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)
        o = sess.run(i, { is_training: True })

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            top1_error_value= sess.run(top1_error, feed_dict={is_training: True})
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d loss = %.2f, accuracy = %g (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, top1_error_value, examples_per_sec, duration))
        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(FLAGS.save_model_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        if step > 1 and step % 100 == 0:
            _, top1_error_value, summary_value = sess.run([val_op, top1_error, summary_op], { is_training: False })
            print('Validation accuracy %.2f' % top1_error_value)
            val_summary_writer.add_summary(summary_value, step)