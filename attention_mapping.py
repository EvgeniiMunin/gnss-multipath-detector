from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import multiply, Lambda, LocallyConnected2D, Input, Conv2D, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import models
import numpy as np

def attention_map_model(backbone):
    """ 
    Add attention map branch to CNN
    use DenseNet121, MobileNet as backbone
    
    Example:
    backbone = DenseNet121(weights='imagenet', include_top=False)
    backbone = MobileNet(weights='imagenet', include_top=False)
    """

    in_layer = Input((80,80,2)) # ???? dimesnions question ????
    reshape_layer = Conv2D(3, (3,3), activation='relu', padding='same', input_shape=(80,80,2))
    for layer in backbone.layers:
        layer.trainable = True
    
    reshape_ = reshape_layer(in_layer)
    pt_depth = backbone.get_output_shape_at(0)[-1]
    pt_features = backbone(reshape_)
    bn_features = BatchNormalization()(pt_features)
    
    # attention mech to turn pixels in GAP on/ off
    attn_layer = Conv2D(64, (1,1), padding='same', activation='relu')(bn_features)
    attn_layer = Conv2D(64, (1,1), padding='same', activation='relu')(attn_layer)
    attn_layer = LocallyConnected2D(1, (1,1), padding='valid', activation='sigmoid')(attn_layer)
    
    # insert it to backbone branch
    # initialize weights
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, (1,1), padding='same', activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    
    # get together attn_layer and bn_features branches
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    
    # account for missing values from attention model
    gap = Lambda(lambda x: x[0]/x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.5)(gap)
    dr_steps = Dropout(0.25)(Dense(1024, activation='elu')(gap_dr))
    # linear 16 bit
    out_layer = Dense(1, activation='sigmoid')(dr_steps)
    
    attn_model = models.Model(inputs=[in_layer], outputs=[out_layer])
    
    return attn_model
