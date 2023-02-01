========================================================
Conditional Hybrid GAN for Melody Generation from Lyrics
========================================================
Yi Yu,  National Institute of Informatics,  Tokyo, Japan
========================================================
Abhishek Srivastava, Rajiv Ratn Shah, IIIT, Delhi, India
========================================================

Where to find the code ?

    Code related to C-Hybrid-GAN can be found at ./code/c_hybrid_gan

        train.py is be used to train the model
        evaluate.py is used to evaluate the model on test data
        generate.py is used to generate melody for a given lyrics
        
        refer to ./code/c_hybrid_gan/run.ipynb for usage.
      
    Code to produce visualisations can be found at ./code/viz

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
