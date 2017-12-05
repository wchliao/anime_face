import argparse
import numpy as np
import tensorflow as tf
from DataSet import DataSet
import utils
import os
import time
import csv
import skipthoughts
import scipy.misc


##### Global Constants #####

image_path = 'faces/'
tag_file = 'tags_clean.csv'
model_file = './model/DCGAN/model.ckpt'
sample_path = 'samples/'

############################


##### Parameters #####

batch_size = 64
image_size = 64
caption_vec_size = 4800
noise_dim = 100
channel_dim = 64
reduced_text_dim = 256
learning_rate = 0.0002
momentum = 0.5
max_epoch = 300
max_epoch = 10

######################


##### GPU Options #####

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

#######################


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--generate', action='store_true', help='Run testing')
    parser.add_argument('-t', '--testing_file', help='Should give testing text file')
    parser.add_argument('-n', '--generate_num', help='How many images to generate for each caption')

    return parser.parse_args()


class DCGAN():
    def __init__(self, batch_size, image_size, caption_vec_size, 
            noise_dim, channel_dim, reduced_text_dim, momentum):
        
        self.batch_size = batch_size
        self.image_size = image_size 
        self.caption_vec_size = caption_vec_size
        self.noise_dim = noise_dim
        self.channel_dim = channel_dim
        self.reduced_text_dim = reduced_text_dim

        self.g_bn0 = utils.batch_norm(momentum=momentum, name='g_bn0')
        self.g_bn1 = utils.batch_norm(momentum=momentum, name='g_bn1')
        self.g_bn2 = utils.batch_norm(momentum=momentum, name='g_bn2')
        self.g_bn3 = utils.batch_norm(momentum=momentum, name='g_bn3')

        self.d_bn1 = utils.batch_norm(momentum=momentum, name='d_bn1')
        self.d_bn2 = utils.batch_norm(momentum=momentum, name='d_bn2')
        self.d_bn3 = utils.batch_norm(momentum=momentum, name='d_bn3')
        self.d_bn4 = utils.batch_norm(momentum=momentum, name='d_bn4')

        return


    def build_model(self):
        real_image = tf.placeholder('float32', [self.batch_size,
            self.image_size, self.image_size, 3], name = 'read_image')
        wrong_image = tf.placeholder('float32', [self.batch_size,
            self.image_size, self.image_size, 3], name = 'wrong_image')
        caption = tf.placeholder('float32', [self.batch_size, self.caption_vec_size], name = 'caption')
        noise = tf.placeholder('float32', [self.batch_size, self.noise_dim])

        fake_image = self.generator(noise, caption)

        d_real_image, d_real_image_logits = self.discriminator(real_image, caption)
        d_wrong_image, d_wrong_image_logits = self.discriminator(wrong_image, caption, reuse=True)
        d_fake_image, d_fake_image_logits = self.discriminator(fake_image, caption, reuse=True)

        g_loss = tf.reduce_mean(tf.square(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_image_logits, labels=tf.ones_like(d_fake_image))))

        d_loss1 = tf.reduce_mean(tf.square(d_real_image_logits - tf.ones_like(d_real_image)))
        d_loss2 = tf.reduce_mean(tf.square(d_wrong_image_logits))
        d_loss3 = tf.reduce_mean(tf.square(d_fake_image_logits))

        d_loss = d_loss1 + d_loss2 + d_loss3

        all_var = tf.trainable_variables()
        d_vars = [var for var in all_var if 'd_' in var.name]
        g_vars = [var for var in all_var if 'g_' in var.name]

        inputs = {
                'real_image': real_image,
                'wrong_image': wrong_image,
                'caption': caption,
                'noise': noise
        }

        variables = {
                'd_vars': d_vars,
                'g_vars': g_vars
        }

        loss = {
                'g_loss': g_loss,
                'd_loss': d_loss,
        }

        return inputs, variables, loss
             

    def build_generator(self):
        caption = tf.placeholder('float32', [self.batch_size, self.caption_vec_size], name = 'caption')
        noise = tf.placeholder('float32', [self.batch_size, self.noise_dim])

        fake_image = self.sampler(noise, caption)

        inputs = {
                'caption': caption,
                'noise': noise
        }

        return inputs, fake_image


    def sampler(self, noise, caption):
        tf.get_variable_scope().reuse_variables()

        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        reduce_caption = utils.lrelu(utils.linear(caption, self.reduced_text_dim, 'g_embedding'))
        noise_concat = tf.concat([noise, reduce_caption], 1)
        new_noise = utils.linear(noise_concat, self.channel_dim*8*s16*s16, 'g_h0_lin')

        h0 = tf.reshape(new_noise, [-1, s16, s16, self.channel_dim*8])
        h0 = tf.nn.relu(self.g_bn0(h0, is_training = False))

        h1 = utils.deconv2d(h0, [self.batch_size, s8, s8, self.channel_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, is_training = False))

        h2 = utils.deconv2d(h1, [self.batch_size, s4, s4, self.channel_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, is_training = False))

        h3 = utils.deconv2d(h2, [self.batch_size, s2, s2, self.channel_dim], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, is_training = False))

        h4 = utils.deconv2d(h3, [self.batch_size, s, s, 3], name='g_h4')

        return (tf.tanh(h4)/2. + 0.5)


    def generator(self, noise, caption):
        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        reduced_caption = utils.lrelu(utils.linear(caption, self.reduced_text_dim, 'g_embedding'))
        noise_concat = tf.concat([noise, reduced_caption], 1)
        new_noise = utils.linear(noise_concat, self.channel_dim*8*s16*s16, 'g_h0_lin')

        h0 = tf.reshape(new_noise, [-1, s16, s16, self.channel_dim*8])
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = utils.deconv2d(h0, [self.batch_size, s8, s8, self.channel_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = utils.deconv2d(h1, [self.batch_size, s4, s4, self.channel_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = utils.deconv2d(h2, [self.batch_size, s2, s2, self.channel_dim], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = utils.deconv2d(h3, [self.batch_size, s, s, 3], name='g_h4')

        return (tf.tanh(h4)/2. + 0.5)


    def discriminator(self, image, caption, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = utils.lrelu(utils.conv2d(image, self.channel_dim, name='d_h0_conv'))
        h1 = utils.lrelu(self.d_bn1(utils.conv2d(h0, self.channel_dim*2, name = 'd_h1_conv')))
        h2 = utils.lrelu(self.d_bn2(utils.conv2d(h1, self.channel_dim*4, name = 'd_h2_conv')))
        h3 = utils.lrelu(self.d_bn3(utils.conv2d(h2, self.channel_dim*8, name = 'd_h3_conv')))

        reduced_caption = utils.lrelu(utils.linear(caption, self.reduced_text_dim, 'd_embedding'))
        reduced_caption = tf.expand_dims(reduced_caption, 1)
        reduced_caption = tf.expand_dims(reduced_caption, 2)
        tiled_caption = tf.tile(reduced_caption, [1,4,4,1], name='tiled_embedding')

        h3_concat = tf.concat([h3, tiled_caption], 3, name='h3_concat')
        h3_new = utils.lrelu(self.d_bn4(utils.conv2d(h3_concat, self.channel_dim*8, 1,1,1,1, name='d_h3_conv_new')))

        h4 = utils.linear(tf.reshape(h3_new, [self.batch_size, -1]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4


    def save_model(self, sess, model_file):
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir(os.path.dirname(model_file))
        saver = tf.train.Saver()
        saver.save(sess, model_file)
        return


    def restore_model(self, sess, model_file):
        if os.path.isdir(os.path.dirname(model_file)):
            saver = tf.train.Saver()
            saver.restore(sess, model_file)


def train():
    model = DCGAN(
            batch_size = batch_size, 
            image_size = image_size, 
            caption_vec_size = caption_vec_size, 
            noise_dim = noise_dim, 
            channel_dim = channel_dim, 
            reduced_text_dim = reduced_text_dim,
            momentum = momentum
    )
    inputs, variables, loss = model.build_model()

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        d_optimizer = tf.train.AdamOptimizer(learning_rate, 
            beta1 = momentum).minimize(loss['d_loss'], var_list=variables['d_vars'])
        g_optimizer = tf.train.AdamOptimizer(learning_rate, 
            beta1 = momentum).minimize(loss['g_loss'], var_list=variables['g_vars'])

    data = DataSet(image_path, tag_file, image_size)
    
    init = tf.global_variables_initializer()

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        model.restore_model(sess, model_file)
        
        epoch = -1
        start_time = time.time()
        while epoch < max_epoch:
            real_image, wrong_image, caption = data.next_batch(batch_size=batch_size)
            noise = np.random.uniform(-1, 1, [batch_size, noise_dim])

            sess.run(d_optimizer, feed_dict={
                inputs['real_image']: real_image,
                inputs['wrong_image']: wrong_image,
                inputs['caption']: caption,
                inputs['noise']: noise
            })

            sess.run(g_optimizer, feed_dict={
                inputs['real_image']: real_image,
                inputs['wrong_image']: wrong_image,
                inputs['caption']: caption,
                inputs['noise']: noise
            })

            sess.run(g_optimizer, feed_dict={
                inputs['real_image']: real_image,
                inputs['wrong_image']: wrong_image,
                inputs['caption']: caption,
                inputs['noise']: noise
            })

            if epoch != data.N_epoch:
                epoch = data.N_epoch
                model.save_model(sess, model_file)
                used_time = time.time() - start_time
                start_time = time.time()

                d_loss, g_loss = sess.run([loss['d_loss'], loss['g_loss']], feed_dict={
                    inputs['real_image']: real_image,
                    inputs['wrong_image']: wrong_image,
                    inputs['caption']: caption,
                    inputs['noise']: noise
                })

                print(str(epoch) + '/' + str(max_epoch) + ' epoch: ' +
                        'd_loss = ' + str(d_loss) + ' ' +
                        'g_loss = ' + str(g_loss) + ' ' +
                        'time = ' + str(used_time) + ' secs')

    return


def generate(testing_file, generate_num):
    model = DCGAN(
            batch_size = generate_num, 
            image_size = image_size, 
            caption_vec_size = caption_vec_size, 
            noise_dim = noise_dim, 
            channel_dim = channel_dim, 
            reduced_text_dim = reduced_text_dim,
            momentum = momentum
    )
    _, _, _ = model.build_model()

    inputs, image = model.build_generator()

    captions = []

    with open(testing_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            captions.append(line[1])

    sent2vec = skipthoughts.load_model()
    caption_vecs = skipthoughts.encode(sent2vec, captions)
    
    generated_images = []

#    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        model.restore_model(sess, model_file)
        
        for vec in caption_vecs:
            noise = np.random.uniform(-1, 1, [generate_num, noise_dim])
            caption = [vec] * generate_num
            generated_images.append(sess.run(image, feed_dict={
                inputs['caption']: caption,
                inputs['noise']: noise
            }))

    if not os.path.isdir(sample_path):
        os.mkdir(sample_path)

    for i, images in enumerate(generated_images, start=1):
        for j, image in enumerate(images, start=1):
            scipy.misc.imsave(os.path.join(sample_path, 'sample_{}_{}.jpg'.format(i, j)), image)

    return


if __name__ == '__main__':
    args = parse_args()
    if args.train:
        train()
    if args.generate:
        generate(args.testing_file, int(args.generate_num))

