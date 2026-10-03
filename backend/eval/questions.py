"""Questions about eval/data/machinelearning.pdf (LIP2AUDSPEC, ICASSP 2018) with reference answers."""

TARGET_FILE = "machinelearning.pdf"

QUESTIONS = [
    {
        "query": "What is the main objective of the LIP2AUDSPEC paper?",
        "reference": "The main objective is to reconstruct intelligible speech from silent lip movement videos using a deep neural network that combines an autoencoder with a lip reading network.",
    },
    {
        "query": "What type of spectrogram is used as the speech representation?",
        "reference": "Auditory spectrogram is used as the spectral representation of speech which gives a higher quality of re-synthesis than traditional spectrograms.",
    },
    {
        "query": "What is the architecture of the autoencoder used in the network?",
        "reference": "The autoencoder takes a 128 frequency bin auditory spectrogram as input and output with a bottleneck of size 32. It uses Dense layers with LeakyReLU activations and Gaussian noise at the bottleneck with sigma 0.05.",
    },
    {
        "query": "What neural network layers make up the lip reading network?",
        "reference": "The lip reading network consists of a 7-layer 3D convolutional network for spatiotemporal feature extraction followed by a single-layer LSTM with 512 units a fully connected layer and an output layer.",
    },
    {
        "query": "What dataset was used to train the model?",
        "reference": "The GRID audio-visual corpus was used containing audio and video recordings of 34 speakers with 1000 utterances each. Training used two male speakers S1 and S2 and two female speakers S4 and S29 with an 80-10-10 train-validation-test split.",
    },
    {
        "query": "What loss function was used during training?",
        "reference": "The CorrMSE loss function was used which combines mean squared error and correlation with lambda set to 0.5 to balance the two components.",
    },
    {
        "query": "What is the optimal bottleneck size and why?",
        "reference": "The optimal bottleneck size is 32 nodes. Increasing to 64 improves autoencoder output quality but makes reconstruction harder for the lip reading network. 32 nodes balances this trade-off with 98 percent correlation.",
    },
    {
        "query": "How does the proposed method compare to Vid2Speech in PESQ score?",
        "reference": "The proposed method achieves an average PESQ of 1.88 compared to Vid2Speech's 1.76 and outperforms it across speakers S1 S2 and S29.",
    },
    {
        "query": "What were the results of the human evaluation on Amazon Mechanical Turk?",
        "reference": "The model achieved 55.8 percent average word recognition accuracy versus 50.9 for Vid2Speech and scored 1.63 versus 1.35 on natural sound quality on a scale of 1 to 5 and 85.1 percent correct gender identification versus 43.2 percent.",
    },
    {
        "query": "What is the effect of using Gaussian noise at the bottleneck?",
        "reference": "Gaussian noise slightly reduces autoencoder output accuracy but significantly improves overall lip reading performance by allowing the autoencoder to handle variations in input videos.",
    },
    {
        "query": "What evaluation metrics were used to assess the network performance?",
        "reference": "Three metrics were used: 2D correlation between reconstructed and actual auditory spectrogram, Perceptual Evaluation of Speech Quality PESQ for audio quality, and Spectro Temporal Modulation Index STMI for intelligibility.",
    },
    {
        "query": "What were the video preprocessing steps applied to the dataset?",
        "reference": "Each video frame was converted to grayscale and normalized. The face region was extracted and resized. Videos were divided into K non-overlapping slices of length Lv and first and second order temporal derivatives were stacked to form a 4D tensor.",
    },
    {
        "query": "What optimizer and learning rate were used for training?",
        "reference": "The Adam optimizer was used with an initial learning rate of 0.0001.",
    },
    {
        "query": "How does dropout affect autoencoder performance according to the ablation study?",
        "reference": "Using dropout makes results worse. PESQ dropped from 2.81 to 2.33 and Corr2D dropped from 0.98 to 0.95 when dropout was used.",
    },
    {
        "query": "What are the future directions mentioned in the conclusion?",
        "reference": "Future work includes collecting more training data including emotions in reconstructed speech and proposing an end-to-end structure to directly estimate raw waveform from facial speech-related features.",
    },
]

# Nothing in the eval folder answers these; retrieval should report low relevance.
UNANSWERABLE = [
    "What is the boiling point of mercury?",
    "Who won the 2010 FIFA World Cup?",
]
