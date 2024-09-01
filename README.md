# Aspect-Term Polarity Classification in Sentiment Analysis

The goal of this project is to implement a model that predicts opinion polarities (positive, 
negative or neutral) for given aspect terms in sentences. The model takes as input 3 elements: a 
sentence, a term occurring in the sentence, and its aspect category. For each input triple, it produces 
a polarity label: positive, negative or neutral. Note: the term can occur more than once in the same 
sentence, that is why its character start/end offsets are also provided. 
 

# Dataset
 
The dataset is in TSV format, one instance per line. Each line contains 5 tab-separated fields: the polarity of the opinion (the ground truth polarity label), 
the aspect category on which the opinion is expressed, a specific target term, the character offsets 
of the term (start:end), and the sentence in which the term occurs and the opinion is expressed. 
 


# Classifier 
The classifier is a **Bilateral LSTM with an attention layer**.   

The sentence and aspect terms are preprocessed (tokenization, removal of stopwords and padding).   

The word embedding comes from the pre-trained word vector of [Global Vectors for Word Representation (GloVe)](https://nlp.stanford.edu/projects/glove/).   
The following pre-trained word embedding was used: ./glove_embedding/glove.6B.300d.txt

# Results

The results are of the order of 80% accuracy which is interesting for this model. The execution time is reasonable. We know from other works that grabbing percents on accuracy can result in multiplying the execution time.

Dev accs: [78.72, 79.26, 81.38, 81.12, 80.59]  
**Mean Dev Acc.: 80.21 (1.05)**  
Exec time: 229.29 s. ( 45 per run )

# Improvements
Other models like **TD LSTM** or **ATAE LSTM** and corresponding preprocessing technique have been explored but without yielding better results. Further work could focus on finding the optimal parameters for the models to overpass actual results.

