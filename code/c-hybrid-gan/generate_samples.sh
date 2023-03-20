CKPT_PATH=../../checkpoints/pre_train_c_hybrid_gan
MIDI_NAME=ch-mle-no-autotune

python generate_from_test_with_seed.py --CKPT_PATH $CKPT_PATH --ID 988 --MIDI_NAME gimme-some-lovin-${MIDI_NAME} --IS_NOT_GAN
python generate_from_test_with_seed.py --CKPT_PATH $CKPT_PATH --ID 889 --MIDI_NAME in-my-life-${MIDI_NAME} --IS_NOT_GAN
python generate_from_test_with_seed.py --CKPT_PATH $CKPT_PATH --ID 774 --MIDI_NAME zombie-${MIDI_NAME} --IS_NOT_GAN
python generate_from_test_with_seed.py --CKPT_PATH $CKPT_PATH --ID 228 --MIDI_NAME dont-stop-me-now-${MIDI_NAME} --IS_NOT_GAN