"""
Script to evaluate the C-Hybrid-MLE & C-Hybrid-GAN model on test data.
C-Hybrid-MLE is the model obtained at the end of pre-training.
Used to evaluate the last SONG_LENGTH - SEED_LENGTH notes agains ground truth
and models using seed
"""

import pandas as pd

from discriminator import *
from drivers import *
from generator import *
from utils import *


def main(SONG_LENGTH: int):
    """# Data"""

    print("Data: \n")
    SEED_LENGTH = 0

    NUM_P_TOKENS = len((pickle.load(open(P_LE_PATH, "rb"))).classes_)
    NUM_D_TOKENS = len((pickle.load(open(D_LE_PATH, "rb"))).classes_)
    NUM_R_TOKENS = len((pickle.load(open(R_LE_PATH, "rb"))).classes_)

    NUM_TOKENS = [NUM_P_TOKENS, NUM_D_TOKENS, NUM_R_TOKENS]
    LE_PATHS = [P_LE_PATH, D_LE_PATH, R_LE_PATH]

    # load train (to compute STEPS_PER_EPOCH_TRAIN) and test data

    x_train, y_train_dat_attr, y_train = load_data(TRAIN_DATA_PATH,
                                                   LE_PATHS,
                                                   SONG_LENGTH,
                                                   NUM_SONG_FEATURES,
                                                   NUM_META_FEATURES)

    x_test, y_test_dat_attr, y_test = load_data(TEST_DATA_PATH,
                                                LE_PATHS,
                                                SONG_LENGTH,
                                                NUM_SONG_FEATURES,
                                                NUM_META_FEATURES,
                                                convert_to_tensor=True)

    inp = x_test, y_test

    TRAIN_LEN = len(x_train)
    TEST_LEN = len(x_test)

    STEPS_PER_EPOCH_TRAIN = np.ceil(TRAIN_LEN / BATCH_SIZE)
    print('Steps per epoch train: ', STEPS_PER_EPOCH_TRAIN)

    print("================================================================ \n ")

    # Set seed for reproducibility

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

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

    # Initialise pre-train driver
    pre_train_driver = PreTrainDriver(g_model,
                                      pre_train_g_opt,
                                      MAX_GRAD_NORM,
                                      NUM_TOKENS)

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

    # Setup checkpoint for pretraining
    pre_train_ckpt = tf.train.Checkpoint(g_model=g_model,
                                         pre_train_g_opt=pre_train_g_opt)

    pre_train_ckpt_manager = tf.train.CheckpointManager(
        pre_train_ckpt, PRE_TRAIN_CKPT_PATH, max_to_keep=PRE_TRAIN_EPOCHS)

    # Setup checkpoint for adversarial training
    adv_train_ckpt = tf.train.Checkpoint(g_model=g_model,
                                         d_model=d_model,
                                         adv_train_g_opt=adv_train_g_opt,
                                         adv_train_d_opt=adv_train_d_opt)

    adv_train_ckpt_manager = tf.train.CheckpointManager(
        adv_train_ckpt, ADV_TRAIN_CKPT_PATH, max_to_keep=ADV_TRAIN_EPOCHS)

    if REUSE_TEST_RESULT_LOGS:
        # load test result logs from previous run
        try:
            test_result_logs = pd.read_csv(TEST_RESULT_LOGS_FILENAME).to_dict('records')
        except FileNotFoundError:
            test_result_logs = []
    else:
        test_result_logs = []

    """# Evaluation"""

    SEED_LENGTH = 8
    # Compute ground truth data statistics and attribute information

    # Remove the seed melody from test song attributes
    y_test_dat_attr_p = y_test_dat_attr[0][:, SEED_LENGTH:]
    y_test_dat_attr_d = y_test_dat_attr[1][:, SEED_LENGTH:]
    y_test_dat_attr_r = y_test_dat_attr[2][:, SEED_LENGTH:]

    y_test_dat_attr_no_seed = (y_test_dat_attr_p, y_test_dat_attr_d, y_test_dat_attr_r)
    SONG_LENGTH = SONG_LENGTH - SEED_LENGTH
    # gather song attributes
    test_dat_songs = gather_song_attr(y_test_dat_attr_no_seed, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

    # compute stats
    test_dat_stats = gather_stats(y_test_dat_attr_no_seed, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))
    test_dat_mean_stats = get_mean_stats(test_dat_stats, TEST_LEN)

    print('Test data mean stats')
    for key, value in test_dat_mean_stats.items():
        print(f"{key} = {value}")

    # save ground truth attributes, songs, mean stats 
    pickle.dump(y_test_dat_attr, open('../../results/test/y_test_dat_attr.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    np.save('../../results/test/test_dat_songs.npy', test_dat_songs)
    pickle.dump(test_dat_mean_stats, open('../../results/test/test_dat_mean_stats.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    print("================================================================ \n ")

    """## C-Hybrid-MLE"""

    # restore the checkpoint at the end of pre-training.

    if pre_train_ckpt_manager.latest_checkpoint:
        pre_train_ckpt.restore(pre_train_ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest pretrain checkpoint restored from {}'.format(pre_train_ckpt_manager.latest_checkpoint))

    # reset the temperature
    adv_train_driver.reset_temp()

    print('Temperature: {}'.format(adv_train_driver.temp.numpy()))

    # Compute statistics & attribute info. of songs generated using C-Hybrid-MLE

    result = {}

    for run_id in range(EVAL_RUNS):

        print(f"\n ***** --run-{run_id}-- ***** \n")

        logs = {'seed': SEED,
                'run_id': run_id,
                'method': 'C-Hybrid-MLE'}

        # generate test song attr.
        test_g_out = adv_train_driver.seed_generate(inp)
        test_g_out = test_g_out[0][:, SEED_LENGTH:], test_g_out[1][:, SEED_LENGTH:], test_g_out[2][:, SEED_LENGTH:]

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=True)
        y_test_gen_attr_p = y_test_gen_attr[0]
        y_test_gen_attr_d = y_test_gen_attr[1]
        y_test_gen_attr_r = y_test_gen_attr[2]

        # gather song attributes
        test_gen_songs = gather_song_attr(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        # compute self bleu
        test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)
        print('Self-BLEU: {}'.format(test_self_bleu))

        # compute mmd
        test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr_no_seed, y_test_gen_attr)
        test_o_mmd = np.abs(test_p_mmd) + np.abs(test_d_mmd) + np.abs(test_r_mmd)
        print('pMMD: {}\ndMMD: {}\nrMMD: {}\noMMD: {}\n'.format(test_p_mmd, test_d_mmd, test_r_mmd, test_o_mmd))

        # compute stats
        test_gen_stats = gather_stats(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        test_gen_mean_stats = get_mean_stats(test_gen_stats, TEST_LEN)
        print('C-Hybrid-MLE mean stats')
        for key, value in test_gen_mean_stats.items():
            result[key] = result.get(key, []) + [value]
            print(f"{key}: {value}")

        result['oMMD'] = result.get('oMMD', []) + [test_o_mmd]
        result['self-BLEU-4'] = result.get('self-BLEU-4', []) + [test_self_bleu[4]]

        # save test result logs for current run

        for n_gram in N_GRAMS:
            logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

        logs['pMMD'] = test_p_mmd
        logs['dMMD'] = test_d_mmd
        logs['rMMD'] = test_r_mmd
        logs['oMMD'] = test_o_mmd

        logs = {**logs, **test_gen_mean_stats}

        test_result_logs.append(logs)

        print()

    # print average stats across runs

    print(f'\nC-Hybrid-MLE mean stats across {EVAL_RUNS} runs.\n')
    for key, value in result.items():
        print('{}: {} +/- {}'.format(key, round(np.mean(value), 4), round(np.std(value), 4)))

    # save songs, attr & mean stats generated using C-Hybrid-MLE 

    pickle.dump(y_test_gen_attr, open('../../results/c_hybrid_gan/generated/c_hybrid_mle_y_test_gen_attr.pkl', 'wb'),
                pickle.HIGHEST_PROTOCOL)
    np.save('../../results/c_hybrid_gan/generated/c_hybrid_mle_test_gen_songs.npy', test_gen_songs)
    pickle.dump(test_gen_mean_stats,
                open('../../results/c_hybrid_gan/generated/c_hybrid_mle_test_mean_stats.pkl', 'wb'),
                pickle.HIGHEST_PROTOCOL)

    print("================================================================ \n ")

    """## C-Hybrid-GAN"""

    # restore the checkpoint at the end of adversarial training.

    if adv_train_ckpt_manager.latest_checkpoint:
        adv_train_ckpt.restore(adv_train_ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored from {}'.format(adv_train_ckpt_manager.latest_checkpoint))

    # update the temperature
    adv_train_driver.update_temp(ADV_TRAIN_EPOCHS - 1, STEPS_PER_EPOCH_TRAIN - 1)
    print('Temperature: {}'.format(adv_train_driver.temp.numpy()))

    # Compute statistics & attribute info. of songs generated using C-Hybrid-GAN

    result = {}

    for run_id in range(EVAL_RUNS):

        print(f"\n ***** --run-{run_id}-- ***** \n")

        logs = {'seed': SEED,
                'run_id': run_id,
                'method': 'C-Hybrid-GAN'}

        # generate test song attr.
        test_g_out = adv_train_driver.seed_generate(inp)
        test_g_out = test_g_out[0][:, SEED_LENGTH:], test_g_out[1][:, SEED_LENGTH:], test_g_out[2][:, SEED_LENGTH:]

        # infer attributes of generated songs
        y_test_gen_attr = infer(test_g_out, LE_PATHS, is_tune=True)
        y_test_gen_attr_p = y_test_gen_attr[0]
        y_test_gen_attr_d = y_test_gen_attr[1]
        y_test_gen_attr_r = y_test_gen_attr[2]

        # gather song attributes
        test_gen_songs = gather_song_attr(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        # compute self bleu
        test_self_bleu = compute_self_bleu_score(y_test_gen_attr, N_GRAMS)
        print('Self-BLEU: {}'.format(test_self_bleu))

        # compute mmd
        test_p_mmd, test_d_mmd, test_r_mmd = compute_mmd_score(y_test_dat_attr_no_seed, y_test_gen_attr)
        test_o_mmd = np.abs(test_p_mmd) + np.abs(test_d_mmd) + np.abs(test_r_mmd)
        print('pMMD: {}\ndMMD: {}\nrMMD: {}\noMMD: {}\n'.format(test_p_mmd, test_d_mmd, test_r_mmd, test_o_mmd))

        # compute stats
        test_gen_stats = gather_stats(
            y_test_gen_attr, (TEST_LEN, SONG_LENGTH, NUM_SONG_FEATURES))

        test_gen_mean_stats = get_mean_stats(test_gen_stats, TEST_LEN)
        print('C-Hybrid-GAN mean stats')
        for key, value in test_gen_mean_stats.items():
            result[key] = result.get(key, []) + [value]
            print(f"{key}: {value}")

        # save test result logs for current run

        result['oMMD'] = result.get('oMMD', []) + [test_o_mmd]
        result['self-BLEU-4'] = result.get('self-BLEU-4', []) + [test_self_bleu[4]]

        for n_gram in N_GRAMS:
            logs[f'selfBLEU_{n_gram}'] = test_self_bleu[n_gram]

        logs['pMMD'] = test_p_mmd
        logs['dMMD'] = test_d_mmd
        logs['rMMD'] = test_r_mmd
        logs['oMMD'] = test_o_mmd

        logs = {**logs, **test_gen_mean_stats}

        test_result_logs.append(logs)

        print()

    # print average  stats across runs

    print(f'\nC-Hybrid-GAN mean stats across {EVAL_RUNS} runs.\n')
    for key, value in result.items():
        print('{}: {} +/- {}'.format(key, round(np.mean(value), 4), round(np.std(value), 4)))

    # save songs, attr & mean stats generated using C-Hybrid-GAN

    pickle.dump(y_test_gen_attr, open('../../results/c_hybrid_gan/generated/c_hybrid_gan_y_test_gen_attr.pkl', 'wb'),
                pickle.HIGHEST_PROTOCOL)
    np.save('../../results/c_hybrid_gan/generated/c_hybrid_gan_test_gen_songs.npy', test_gen_songs)
    pickle.dump(test_gen_mean_stats,
                open('../../results/c_hybrid_gan/generated/c_hybrid_gan_test_mean_stats.pkl', 'wb'),
                pickle.HIGHEST_PROTOCOL)

    # save test result logs as csv file

    pd.DataFrame.from_records(test_result_logs).to_csv(TEST_RESULT_LOGS_FILENAME, index=False)


if __name__ == '__main__':

    settings = {'settings_file': 'settings'}
    settings = load_settings_from_file(settings)

    print("Settings: \n")
    for (k, v) in settings.items():
        print(v, '\t', k)

    locals().update(settings)

    print("================================================================ \n ")

    main(SONG_LENGTH=SONG_LENGTH)
