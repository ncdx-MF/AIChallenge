import tensorflow as tf
import keras.backend as K
import numpy as np
import os
import cv2

from keras.layers import Input, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
#from denseyolo import DenseYolo
from dense121_yolo_refine import DenseYolo,loss
from keras.backend import mean
from dataset import *


###############     Net_config   #################
# Mode = 'train' or 'visualize' or 'drawModel'
Mode = 'drawModel'
weights_path='model_weights/'


input_shape = (448,448)
num_classes = 10
seg_classes = 2
anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
anchors = np.array(anchors)
batch_size = 32


logging = TensorBoard(log_dir=weights_path)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss', save_weights_only=True,
                             save_best_only=True, period=3)  # 只存储weights，
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)  # 当评价指标不在提升时，减少学习率
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)  # 测试集准确率，下降前终止
##########################################################

#gpu configure
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


model = create_model(input_shape, 
                     anchors, 
                     num_classes)

if Mode == 'train':
    print(Mode)
    
    # use custom yolo_loss Lambda layer.
    model.compile(optimizer=Adam(lr=1e-3), loss={'loss': lambda y_true, y_pred: y_pred})
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    name = ('./labels/bdd100k_labels_images_val.json')
    name2 = ('./labels/bdd100k_labels_images_train.json')
    annotation_lines1 = json.load(open(name))
    annotation_lines2 = json.load(open(name2))
    lines = annotation_lines2 + annotation_lines1
    
    model.fit_generator(data_generator_wrapper(lines[:70000], 
                                               batch_size, input_shape, 
                                               anchors, 
                                               num_classes,
                                               seg_classe), #数据集生成器
                        steps_per_epoch=max(1, 70000//batch_size),
                        validation_data=data_generator_wrapper(lines[70000:], 
                                                               batch_size, 
                                                               input_shape, 
                                                               anchors, 
                                                               num_classes,
                                                               seg_classe),#验证集生成器
                        alidation_steps=max(1, 10000//batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint])
        
    model.save_weights(weight_path + 'trained_weights.h5')
    
elif Mode == 'visualize':
    print(Mode)
    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

elif Mode == 'drawModel':
    print(Mode)
    plot_model(model,to_file='./model.png',show_shapes=True)
    
    

###################################################
    
def create_model(input_shape, anchors, num_classes=10,seg_classe=2, load_pretrained=True, freeze_body=2):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(448, 448, 3))
    h, w = input_shape
    num_anchors = len(anchors)


    #create model
    model_body = DenseYolo()

    
    yolo_y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], 
                                w // {0: 32, 1: 16, 2: 8}[l],
                                num_anchors // 3, 
                                num_classes + 5)) for l in range(3)]
    refine_y_true = [Input(shape=(224,224,seg_classe))]
    
    outputs = model_body.outputs
    y_trues = yolo_y_true+refine_y_true
    
    model_loss = Lambda(loss,
                        output_shape=(1,),
                        name='loss',
                        arguments={'anchors':anchors,
                                   'num_classes':num_classes}
                       )(outputs+y_trues)
    
    model = Model(inputs=[model_body.input] + y_trues, outputs=model_loss)

    return model
#################################################################################    
    
    
    
    
    
