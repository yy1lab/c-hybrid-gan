Samples used for testing:
228: Queen - Don't stop me now [Tonight, Tonight, gonna, gonna, have, myself, myself, a, real, good, I, feel, feel, and, the, turning, turning, turning, inside, inside, And, floating, floating, around, around, in, ecstasy]
774: The Cranberries - Zombie [family, in, your, head, in, your, head, they, are, fighting, fighting, with, their, tanks, and, their, bombs, and, their, bombs]
889: In My Life - The Beatles [There, are, places, places, I, remember, remember, remember, my, life, though, some, have, changed, Some, forever, forever, forever, not, for, better, better, some, have, gone, and, some, remain]
984: The Beatles - Hey Jude [Hey, Jude, make, it, bad, Take, a, sad, song, and, make, it, better, better, Remember, Remember, Remember, to, let, her, into, into, your, heart, then, you, can, start, to, make, it, better, better]
988: Gimme Some Lovin' - The Blues Brothers [Well, my, rising, rising, and, my, feet, on, the, floor, Twenty, Twenty, people, people, and, they, wanna, wanna, go, more]


========================================================
Conditional Hybrid GAN for Melody Generation from Lyrics
========================================================
Yi Yu,  National Institute of Informatics,  Tokyo, Japan
========================================================
Abhishek Srivastava, Rajiv Ratn Shah, IIIT, Delhi, India
Karol Lasocki, Aalto University, Helsinki, Finland - seed melody modifications
========================================================

Where to find the code ?

    Code related to C-Hybrid-GAN can be found at ./code/c_hybrid_gan

        train.py is be used to train the model
        evaluate.py is used to evaluate the model on test data
        evaluate_with_seed.py is used to evaluate the model on test data for models continuing initial melodies
        evaluate_without_seed_for_last_notes.py is used to evaluate the model not conditioned on initial melodies,
        only on lyrics, but only considering the last SONG_LENGTH - SEED_LENGTH notes, for comparing with seed models
        generate.py is used to generate melody for a given lyrics
        generate_from_test_with_seed.py is used to generate melody for a given test set ID, using the initial melody
        and complete lyrics

        refer to ./code/c_hybrid_gan/run.ipynb for usage.
      
    Code to produce visualisations can be found at ./code/viz, modified for seed melodies

        generated figures can be found at ./code/viz/figures

How to generate a melody for a given lyrics?

    Use the script generate.py found at ./code/c_hyrbid_gan
    
    To generate a melody you need to pass
    
        --SYLL_LYRICS : it corresponds to the syllable lyrics
        --WORD_LYRICS : it corresponds to the word lyrics 
        --CKPT_PATH   : it corresponds to model checkpoint path
        --IS_GAN      : set to generate using C-Hybrid-GAN
        --MIDI_NAME   : (optional) name of generated midi 
        
     refer to ./code/c_hybrid_gan/run.ipynb for usage.
        
How to change model hyper-parameters ? 

    Hyper-parameters can be changed by editing the settings.txt (json) file.
        
        It can be found at ./code/c_hybrid_gan/settings
     
Where to find model checkpoints ?

    Checkpoints for C-Hybrid-MLE can be found at ./checkpoints/c_hybrid_gan/pre_train_c_hybrid_gan
    Checkpoints for C-Hybrid-GAN can be found at ./checkpoints/c_hybrid_gan/adv_train_c_hybrid_gan
    
Where to find the evaluation result ?

    Results for C-Hybrid-MLE & C-Hybrid-GAN can be found at ./results/c_hybrid_gan/
  
How to produce lyrics-synthesized melody ?

    We use Synthesizer V with the voice of Eleanor Forte.
    
Where to find lyrics-synthesized melodies ?

    lyrics-synthesized melodies can be found at ./synthesized_melodies
    
    Explaination of the directory and song codes can be found in ./synthesized_melodies/readme.txt
