### L101: Machine learning for language processing
### CoNLL 2003 shared task NER: A comparison between a standard and a two stage LSTM framework
Name: Georgios Papagiannis   
CRSid: gp494    
   
This is the code for the results included in the coursework report for the L101 module on Machine learning for language processing.
The objective was to compare a standard LSTM framework and Two-stage LSTM framework for the task of NER. The two-stage LSTM framework use a LSTM that first determines whether
a word is an entity, by classifying to IOB. Then, this acts as input to a second LSTM that classifies the word to its corresponding Named Entity class.

1. The file ```standard_lstm_pref.py``` is responsible for training the standard LSTM framework for NER. The utils files are responsible for data pre-processing.
2. The folder ```lstm_bio/``` contains the files for training the first LSTM of the two stage framework. The file ```lstm_bio.py``` trains the LSTM and the utils
files are responsilbe for data pre-processing.
3. The folder ```lstm_entities/``` contains the files for training the second LSTM of the two stage framework. The file ```lstm_entities.py``` trains the LSTM
with IOB inputs to classify words to NEs and the utils files are responsilbe for data pre-processing.
4. The folder ```eval_two_stage_lstm/``` contains the files for evaluating both the standard and two stage LSTMs.
