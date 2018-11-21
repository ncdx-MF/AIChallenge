import keras
import tensorflow as tf

from keras.layers import Input, concatenate, ZeroPadding2D,multiply,Conv2DTranspose,Add
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.models import Model

from custom_layers import Scale
IMAGE_ORDERING = 'channels_last'




##############################     Total Net Body   ################################################
def DenseYolo(num_classes=10,seg_classes=2,nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, classes=1000, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            num_classes: number of yolo classes
            seg_classes: number of refineNet classes
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    eps = 1.1e-5
    #input_mask = Input(shape=(448//8, 448//8, 1000), name='mask')
    #input_ohem_mask = Input(shape=(448//8, 448//8, 1000), name='ohem_mask')
    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      #img_input = Input(shape=(224, 224, 3), name='data')
      img_input = Input(shape=(448, 448, 3), name='data')
    else:
      concat_axis = 1
      #img_input = Input(shape=(3, 224, 224), name='data')
      img_input = Input(shape=(3,448, 448), name='data')

    #outputs init
    output = [None]*4
    #concatenet layers init
    concat1 = [None]*4
    concat2 = [None]*4

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,24,16]#For DenseNet-121

    ##########  Initial convolution layers#############
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), use_bias=False, name="conv1")(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    #x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    
    ########### Add dense blocks ###############
    for block_idx in range(nb_dense_block):
        stage = block_idx+2
        x, nb_filter = dense_block(x, 
                                   stage, 
                                   nb_layers[block_idx], 
                                   nb_filter, 
                                   growth_rate, 
                                   dropout_rate=dropout_rate, 
                                   weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, 
                             stage, 
                             nb_filter, 
                             compression=compression, 
                             dropout_rate=dropout_rate, 
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

        i = block_idx
        concat1[i] = x        
        x = AveragePooling2D((2, 2), strides=(2, 2), name='Pooling'+str(i))(x)        
        concat2[i] = x

    #banckend
    
    #concat2[0] ==> 112*112*128
    #concat2[1] ==> 56*56*256
    #concat2[2] ==> 28*28*512 
    #concat2[3] ==> 14*14*1024
    

    ########### refine decoder ##########
    g8x, g4x, g2x = create_global_net_dilated((concat1),seg_classes)
    s8x, s4x, s2x = g8x, g4x, g2x
    
    nStack = 2
    #outputs_refine =  [g2x_mask]
    for i in range(nStack):
        s8x, s4x, s2x =  create_stack_refinenet((s8x, s4x, s2x),seg_classes, 'stack_'+str(i))
     
    
    s2x = Conv2D(seg_classes, (3, 3),padding='same', strides=1, use_bias=False, name="refine_cov1")(s2x)
    s2x = BatchNormalization(epsilon=eps, axis=concat_axis, name='refine_bn1')(s2x)
    s2x = Activation('softmax', name='refine_sofxmax1')(s2x)
    
    output[3] = s2x
    
    
    ############ yolo decoder  ###########
    output[0] = concat2[3]
    
    x = concat2[3]
    x = Conv2D(256, (1, 1), strides=1, use_bias=False, name="conv_end1")(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_end1_bn')(x)
    x = Activation('relu', name='relu1_end1')(x)
    
    x = UpSampling2D(size=(2, 2), data_format=None,name='up_end1')(x)#change 13*13 to 26*26 

    x = concatenate([concat2[2], x], axis=concat_axis,name='concat_end1')
    
    x = Conv2D(256, (1, 1), strides=1, use_bias=False, name="conv1_end2")(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_end2_bn1')(x)
    x = Activation('relu', name='relu1_end2')(x)
    x = Conv2D(512, (1, 1), strides=1, use_bias=False, name="conv2_end2")(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_end2_bn2')(x)
    x = Activation('relu', name='relu2_end2')(x)
    x = Conv2D(256, (1, 1), strides=1, use_bias=False, name="conv3_end2")(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_end2_bn3')(x)
    x = Activation('relu', name='relu3_end2')(x)
    x = Conv2D(512, (1, 1), strides=1, use_bias=False, name="conv4_end2")(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_end2_bn4')(x)
    x = Activation('relu', name='relu4_end2')(x)
    x = Conv2D(256, (1, 1), strides=1, use_bias=False, name="conv5_end2")(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_end2_bn5')(x)
    x = Activation('relu', name='relu5_end2')(x)

    output[1] = x

    x = Conv2D(256, (1, 1), strides=1, use_bias=False, name="conv6_end2")(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_end2_bn6')(x)
    x = Activation('relu', name='relu6_end2')(x)
    x = UpSampling2D(size=(2, 2), data_format=None,name='up_end2')(x)     #change 26*26 to 52*52
    x = concatenate([concat2[1], x], axis=concat_axis,name='concat_end2')
    output[2] = x
    


    #yolo outputs
    #output[2] ==> 56*56*128==>56*56*75
    #output[1] ==> 28*28*256==>28*28*75
    #output[0] ==> 14*14*512==>14*14*75
    for i in range(3):
        output[i] = Conv2D(526, (1, 1), strides=1, use_bias=False, name='conv_ou'+str(i)+'_1')(output[i])
        output[i] = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_out'+str(i)+'_bn1')(output[i])
        output[i] = Activation('relu', name='relu_'+str(i)+'_1')(output[i])
        output[i] = Conv2D(128, (1, 1), strides=1, use_bias=False, name='conv_out'+str(i)+'_2')(output[i])
        output[i] = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv_out'+str(i)+'_bn2')(output[i])
        output[i] = Activation('relu', name='relu_'+str(i)+'_2')(output[i])
        output[i] = Conv2D((num_classes+5)*5, (1, 1), strides=1, use_bias=False, name='conv_out'+str(i)+'_3')(output[i])
        output[i] = Activation('relu', name='relu_'+str(i)+'_3')(output[i])
    

    model = Model(img_input, output, name='DenseYolo')

    
    if weights_path is not None:
      model.load_weights(weights_path)

    return model



def loss(args,anchors,num_classes=10):
    '''
        args = [yolo_outputs[0],
                yolo_outputs[1],
                yolo_outputs[2],
                refine_output,
                yolo_y_trues[0],
                yolo_y_trues[1],
                yolo_y_trues[2],
                refine_true]
    '''    
    loss = 0
    yolo_outputs = args[:3]
    yolo_y_trues = args[4:7]
    
    refine_output = args[3]
    refine_true = args[7]

    YOLO_loss = yolo_loss(yolo_outputs+yolo_y_trues, anchors, num_classes) 
    

    Refine_loss = refine_loss(refine_true,refine_output)
    
    loss = YOLO_loss + Refine_loss

    return loss
#####################################################################################



################################ Dense ##############################################
def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, (1, 1), use_bias=False, name=conv_name_base+"_x1")(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), use_bias=False, name=conv_name_base+'_x2')(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), use_bias=False, name=conv_name_base)(x)
    ###x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], 
                                  axis=concat_axis,
                                  name='concat_'+str(stage)+'_'+str(branch)) 


        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

###############################################################################################



################################   refine   ####################################################

def create_refine_net(inputFeatures, n_classes):
    f8x, f4x, f2x = inputFeatures

    # 2 Conv2DTranspose f8x -> fup8x
    fup8x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='refine8x_deconv_1', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING)(f8x)
    fup8x = BatchNormalization()(fup8x)

    fup8x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='refine8x_deconv_2', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING)(fup8x)
    fup8x = BatchNormalization()(fup8x)

    # 1 Conv2DTranspose f4x -> fup4x
    fup4x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='refine4x_deconv', padding='same', activation='relu',
                    data_format=IMAGE_ORDERING)(f4x)

    fup4x = BatchNormalization()(fup4x)

    # 1 conv f2x -> fup2x
    fup2x =  Conv2D(128, (3, 3), activation='relu', padding='same', name='refine2x_conv', data_format=IMAGE_ORDERING)(f2x)
    fup2x =  BatchNormalization()(fup2x)

    # concat f2x, fup8x, fup4x
    fconcat = concatenate([fup8x, fup4x, fup2x], axis=-1, name='refine_concat')

    # 1x1 to map to required feature map
    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='refine2x', data_format=IMAGE_ORDERING)(fconcat)

    return out2x

def create_stack_refinenet(inputFeatures, n_classes, layerName):
    f8x, f4x, f2x = inputFeatures

    # 2 Conv2DTranspose f8x -> fup8x
    fup8x = (Conv2D(256, kernel_size=(1, 1), name=layerName+'_refine8x_1', padding='same', activation='relu'))(f8x)
    fup8x = (BatchNormalization())(fup8x)

    fup8x = (Conv2D(128, kernel_size=(1, 1), name=layerName+'refine8x_2', padding='same', activation='relu'))(fup8x)
    fup8x = (BatchNormalization())(fup8x)

    out8x = fup8x
    fup8x = UpSampling2D((4, 4), data_format=IMAGE_ORDERING)(fup8x)

    # 1 Conv2DTranspose f4x -> fup4x
    fup4x = Conv2D(128, kernel_size=(1, 1), name=layerName+'refine4x', padding='same', activation='relu')(f4x)
    fup4x = BatchNormalization()(fup4x)
    out4x = fup4x
    fup4x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(fup4x)

    # 1 conv f2x -> fup2x
    fup2x = Conv2D(128, (1, 1), activation='relu', padding='same', name=layerName+'refine2x_conv')(f2x)
    fup2x = BatchNormalization()(fup2x)

    # concat f2x, fup8x, fup4x
    fconcat = concatenate([fup8x, fup4x, fup2x], axis=-1, name=layerName+'refine_concat')

    # 1x1 to map to required feature map
    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name=layerName+'refine2x')(fconcat)

    return out8x, out4x, out2x

# create a  global_net of refinenet
def create_global_net_dilated(lowlevelFeatures, n_classes):
    lf2x, lf4x, lf8x, lf16x = lowlevelFeatures

    o = lf16x

    o = Conv2D(256, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='up16x_conv', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)

    o = Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), name='upsample_16x', activation='relu', padding='same',
                    data_format=IMAGE_ORDERING)(o)
    o = concatenate([o, lf8x], axis=-1)
    o = Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='up8x_conv', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    fup8x = o

    o = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='upsample_8x', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING)(o)
    o = concatenate([o, lf4x], axis=-1)
    o = Conv2D(64, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='up4x_conv', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    fup4x = o

    o = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), name='upsample_4x', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING)(o)
    o = concatenate([o, lf2x], axis=-1)
    o = Conv2D(64, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='up2x_conv', data_format=IMAGE_ORDERING)(o)
    o = BatchNormalization()(o)
    fup2x = o

    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out2x', data_format=IMAGE_ORDERING)(fup2x)
    out4x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out4x', data_format=IMAGE_ORDERING)(fup4x)
    out8x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out8x', data_format=IMAGE_ORDERING)(fup8x)

    x4x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(out8x)
    eadd4x = Add(name='global4x')([x4x, out4x])

    x2x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(eadd4x)
    eadd2x = Add(name='global2x')([x2x, out2x])

    return (fup8x, eadd4x, eadd2x)


def refine_loss(refine_true,refine_output):

    loss = 0
    loss = K.sqrt(K.sum(K.square(refine_true - refine_output)))
    
    return loss
#########################################################################################




################################    yolo    #############################################


def yolo_loss(data, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''

    num_layers = len(anchors)//3 # default setting
    yolo_outputs = data[:num_layers]
    y_true = data[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))


    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])

        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))# avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]#2 - y_true[l].w *y_true[l].h 

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
      
        #calculate total loss
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss


def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs
#########################################################################################









