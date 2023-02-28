"""
Script to generate a melody for a given lyrics (syllable-lyrics & word-lyrics)."""
import tensorflow as tf

import numpy as np

import pickle

from generator import *
from discriminator import *
from drivers import *
from utils import *

import argparse
from gensim.models import Word2Vec
import time

def main():
    
    """# Data"""
    
    print("Data: \n")
  
    NUM_P_TOKENS = len((pickle.load(open(P_LE_PATH, "rb"))).classes_)
    NUM_D_TOKENS = len((pickle.load(open(D_LE_PATH, "rb"))).classes_)
    NUM_R_TOKENS = len((pickle.load(open(R_LE_PATH, "rb"))).classes_)
    
    NUM_TOKENS= [NUM_P_TOKENS, NUM_D_TOKENS, NUM_R_TOKENS]
    LE_PATHS  = [P_LE_PATH, D_LE_PATH, R_LE_PATH]

    # load train data -- to compute STEPS_PER_EPOCH_TRAIN

    x_train, y_train_dat_attr, y_train = load_data(TRAIN_DATA_PATH,
                                                   LE_PATHS,
                                                   SONG_LENGTH,
                                                   NUM_SONG_FEATURES,
                                                   NUM_META_FEATURES)


    x_test,  y_test_dat_attr,  y_test  = load_data(TEST_DATA_PATH,
                                                   LE_PATHS,
                                                   SONG_LENGTH,
                                                   NUM_SONG_FEATURES,
                                                   NUM_META_FEATURES,
                                                   convert_to_tensor=True)

    TRAIN_LEN = len(x_train)

    STEPS_PER_EPOCH_TRAIN = np.ceil(TRAIN_LEN/BATCH_SIZE)
    print('Steps per epoch train: ', STEPS_PER_EPOCH_TRAIN)

    ## Initialise Model

    # Initialise generator model
    g_model = Generator(
        G_EMB_UNITS, G_PROJ_UNITS, G_EMB_DROPOUT_RATE, G_PROJ_DROPOUT_RATE,
        G_MEM_SLOTS, G_HEAD_SIZE, G_NUM_HEADS, G_NUM_BLOCKS, NUM_TOKENS,
        NUM_META_FEATURES)

    # Initialise discriminator
    d_model = Discriminator(D_EMB_UNITS, D_PROJ_UNITS, D_EMB_DROPOUT_RATE, D_PROJ_DROPOUT_RATE,
                            D_MEM_SLOTS, D_HEAD_SIZE, D_NUM_HEADS, D_NUM_BLOCKS, NUM_TOKENS,
                            NUM_META_FEATURES)

    ## Initialise Optimizer

    # Initialise optimizer for pretraining
    pre_train_g_opt = tf.keras.optimizers.Adam(
        PRE_TRAIN_LR, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Initialise optimizer for adversarial training
    adv_train_g_opt = tf.keras.optimizers.Adam(ADV_TRAIN_G_LR, beta_1=0.9, beta_2=0.999)
    adv_train_d_opt = tf.keras.optimizers.Adam(ADV_TRAIN_D_LR, beta_1=0.9, beta_2=0.999)

    ## Initialise Driver

    # Initialise adversarial driver
    adv_train_driver = AdversarialDriver(g_model,
                                         d_model,
                                         adv_train_g_opt,
                                         adv_train_d_opt,
                                         TEMP_MAX,
                                         STEPS_PER_EPOCH_TRAIN,
                                         ADV_TRAIN_EPOCHS,
                                         MAX_GRAD_NORM, 
                                         NUM_TOKENS,
                                         seed_len=SEED_LENGTH)

    ## Setup Checkpoint

    if not IS_GAN:
        
        # Setup checkpoint for pretraining
        pre_train_ckpt = tf.train.Checkpoint(g_model=g_model,
                                             pre_train_g_opt=pre_train_g_opt)

        pre_train_ckpt_manager = tf.train.CheckpointManager(
            pre_train_ckpt, CKPT_PATH, max_to_keep=PRE_TRAIN_EPOCHS)
        
        # restore the pre-training checkpoint..

        if pre_train_ckpt_manager.latest_checkpoint:
            pre_train_ckpt.restore(pre_train_ckpt_manager.latest_checkpoint).expect_partial()
            print ('Latest pretrain checkpoint restored from {}'.format(pre_train_ckpt_manager.latest_checkpoint))

        # reset the temperature
        adv_train_driver.reset_temp()

        print('Temperature: {}'.format(adv_train_driver.temp.numpy()))
        
    else:
    
        # Setup checkpoint for adversarial training
        adv_train_ckpt = tf.train.Checkpoint(g_model=g_model,
                                             d_model=d_model,
                                             adv_train_g_opt=adv_train_g_opt,
                                             adv_train_d_opt=adv_train_d_opt)

        adv_train_ckpt_manager = tf.train.CheckpointManager(
            adv_train_ckpt, CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)
        
        # restore the adversarial training checkpoint.
        
        if adv_train_ckpt_manager.latest_checkpoint:
            adv_train_ckpt.restore(adv_train_ckpt_manager.latest_checkpoint).expect_partial()
            print ('Latest checkpoint restored from {}'.format(adv_train_ckpt_manager.latest_checkpoint))

        # update the temperature
        adv_train_driver.update_temp(ADV_TRAIN_EPOCHS-1, STEPS_PER_EPOCH_TRAIN-1)
        print('Temperature: {}'.format(adv_train_driver.temp.numpy()))


    y_p, y_d, y_r = y_test
    song_length = y_p.shape[1]
    inp = np.expand_dims(x_test[ID, :, :], 0), (np.expand_dims(y_p[ID, :, :], 0), np.expand_dims(y_d[ID, :, :], 0), np.expand_dims(y_r[ID, :, :], 0))

    # generate
    out = adv_train_driver.seed_generate(inp, output_seed=True)

    # infer generated song attributes
    gen_attr = infer(out, LE_PATHS, is_tune=True)
    
    # gather song attributes
    gen_song = gather_song_attr(gen_attr, (1, song_length, NUM_SONG_FEATURES))
    gen_song = tf.squeeze(gen_song, 0).numpy()
    
    gen_midi = create_midi_pattern_from_discretized_data(gen_song)
    if MIDI_NAME:
        gen_midi.write(f'../../results/c_hybrid_gan/melodies/{MIDI_NAME}.mid')
        print(f"Melody can be found at ../../results/c_hybrid_gan/melodies/{MIDI_NAME}.mid")
    else:
        timestamp = time.time()
        gen_midi.write(f'../../results/c_hybrid_gan/melodies/{timestamp}.mid')
        print(f"Melody can be found at ../../results/c_hybrid_gan/melodies/{timestamp}.mid")


if __name__ == '__main__':
    
    settings = {'settings_file': 'settings'}
    
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", required=True, type=int, help="Id of a song in the test set")
    # parser.add_argument("--SYLL_LYRICS", required=True, help="syllable lyrics for which melody is to be generated.", type=str)
    # parser.add_argument("--WORD_LYRICS", required=True, help="word lyrics for which melody is to be generated.", type=str)
    parser.add_argument("--CKPT_PATH",   help="path to the model checkpoints.", type=str, 
                        default='../../checkpoints/adv_train_c_hybrid_gan')
    parser.add_argument("--MIDI_NAME", help="name of the generated melody", type=str)
    parser.add_argument("--IS_GAN", help="If model checkpoint corresponds to GAN or not.", action='store_false')

    settings.update(vars(parser.parse_args()))
    settings = load_settings_from_file(settings)
    
    print("Settings: \n")
    for (k, v) in settings.items():
        print(v, '\t', k)
    
    locals().update(settings)
    
    print("================================================================ \n " )
    
    main()
    
