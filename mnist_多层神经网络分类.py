
# coding: utf-8

# In[15]:


import tensorflow as tf


# In[16]:


from tensorflow.examples.tutorials.mnist import input_data


# In[17]:


mnist = input_data.read_data_sets("第03周/MNIST_data",one_hot=True)


# ### 参数设置

# In[18]:


learning_rate=0.001
training_epochs=100
batch_size=100
display_step=5


# ### 网络参数

# In[19]:


n_hidden_1=256
n_hidden_2=256
n_input=784
n_classes=10


# In[20]:


X=tf.placeholder('float',[None,n_input])
Y=tf.placeholder('float',[None,n_classes])


# ### 创建模型

# In[21]:


def multilayer_percetron(x,weights,biases):
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.relu(layer_1)
    
    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2=tf.nn.relu(layer_2)
    
    out_layer=tf.matmul(layer_2,weights['out']) + biases['out']
    
    return out_layer


# In[22]:


weights={'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
         'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
         'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases={'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
}


# In[23]:


pred=multilayer_percetron(X,weights,biases)


# In[24]:


cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[25]:


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch= int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_x,Y:batch_y})
            avg_cost += c / total_batch
        
        if epoch % display_step == 0:
            print('Epoch: %04d'%(epoch+1),'cost: ','{:.9f}'.format(avg_cost))
    
    print('Finished!')
    
    correct_prediction=tf.equal(tf.arg_max(pred,1),tf.arg_max(Y,1))
    
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print('Accuracy:',sess.run(accuracy,feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    

