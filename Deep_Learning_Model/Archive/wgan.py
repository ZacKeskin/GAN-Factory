#Copied from Lena's branch 11am 1/12/17
import tensorflow as tf
import main


def train(BATCH_SIZE,EPOCH,input_data,target_data):
 
    gen1 = main.Generator()
    gen2 = main.Generator()

    with tf.variable_scope('input'):
        #real image placeholder
        real_female_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_female_image')
        #fake image place holder
        real_male_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_male_image')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    #feed the fake image into the generator.
    fake_male_image = generator(real_female_image, is_train)
    reconstructed_female_image = generator2(fake_male_image, is_train)

    fake_female_image = generator2(real_male_image, is_train)
    reconstructed_male_image = generator(fake_female_image, is_train)
    
    #feed the real female image into discriminator (1)
    real_female_result = discriminator(real_female_image, is_train)
    #feed the output of generator2 into disccriminator (1)
    fake_female_result = discriminator(fake_female_image, is_train, reuse=True)

    #feed the real male image into the discriminator (2)
    real_male_result = discriminator2(real_male_image, is_train)
    #feed the output of generator (1) into disccriminator (2)
    fake_male_result = discriminator2(fake_male_image, is_train, reuse=True)

    #reconstructed loss
    #pdb.set_trace()
    reconstructed_female_loss=tf.metrics.mean_squared_error(real_female_image, reconstructed_female_image)
    reconstructed_male_loss=tf.metrics.mean_squared_error(real_male_image, reconstructed_male_image)


    #calculate the loss between the generated image and real image
    d_female_loss = tf.reduce_mean(fake_female_result) - tf.reduce_mean(real_female_result)  # This optimizes discriminator (1).
    d_male_loss = tf.reduce_mean(fake_male_result) - tf.reduce_mean(real_male_result)  # This optimizes discriminator (2).
    
    d_loss = d_female_loss + d_male_loss

    generator_loss = -tf.reduce_mean(fake_male_result)  +tf.reduce_mean(fake_female_result) # This optimizes the generators.
    
    reconstructed_loss = tf.reduce_mean(reconstructed_female_loss) + tf.reduce_mean(reconstructed_male_loss) # This optimizes the generators.

    g_loss= generator_loss + reconstructed_loss
    #returns a list of trainable variables (likes weights and biases)
    t_vars = tf.trainable_variables()
    #trainable discriminator variables are stored in d_vars
    d_female_vars = [var for var in t_vars if 'dis' in var.name ]
    d_male_vars = [var for var in t_vars if 'dis2' in var.name ]
    
    d_vars = [d_female_vars,d_male_vars]
    #trainable generator variables are stored in g_vars
    g_female_vars = [var for var in t_vars if 'gen' in var.name]
    g_male_vars = [var for var in t_vars if 'gen2' in var.name]
    
    g_vars= [g_female_vars,g_male_vars]
   
    #minmise the d_loss using the rms optimizer by altering the discriminator weights and biases      
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-3).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights between the values -0.01 and 0.01
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    
    batch_size = BATCH_SIZE
    image_batch, samples_num = target_data()
    image_batch2, samples_num2 = input_data()
    

    batch_num = int(samples_num / batch_size)
    total_batch = 0
    #start the tf session
    #pdb.set_trace()
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    #saver = tf.train.import_meta_graph('4500.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('./'))
    print(tf.train.latest_checkpoint('./'))
    #initialise the variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('start training...')

    for i in range(EPOCH):
        print('Epoch: ' + str(i))
        for j in range(batch_num):
            print('Batch: ' + str(j))
            d_iters = 2
            g_iters = 1

            
            for k in range(d_iters):
                print(k)
                print(image_batch.shape)
                train_image = sess.run(image_batch)
                train_image2 = sess.run(image_batch2)
                
                sess.run(d_clip)
            # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={people_image: train_image2, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                
                _, gLoss = sess.run([trainer_g, g_loss],
                                   feed_dict={people_image: train_image2, is_train: True})                      
        # save check point every 500 epoch
        if i%500 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))  
        if i%5 == 0:
            # save images
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            imgtest = sess.run(fake_image, feed_dict={people_image: train_image2, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
           
            save_images(imgtest, [8,8] ,output_path + '/epoch' + str(i) + '.jpg')
            
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)

