from tests.evaluator import run_eval_suite, print_eval_report
import os

TEST_CASES = [
    {
        "query": "What is the main objective of the LIP2AUDSPEC paper?",
        "reference_answer": "The main objective is to reconstruct intelligible speech from silent lip movement videos using a deep neural network that combines an autoencoder with a lip reading network.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What type of spectrogram is used as the speech representation?",
        "reference_answer": "Auditory spectrogram is used as the spectral representation of speech which gives a higher quality of re-synthesis than traditional spectrograms.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What is the architecture of the autoencoder used in the network?",
        "reference_answer": "The autoencoder takes a 128 frequency bin auditory spectrogram as input and output with a bottleneck of size 32. It uses Dense layers with LeakyReLU activations and Gaussian noise at the bottleneck with sigma 0.05.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What neural network layers make up the lip reading network?",
        "reference_answer": "The lip reading network consists of a 7-layer 3D convolutional network for spatiotemporal feature extraction followed by a single-layer LSTM with 512 units a fully connected layer and an output layer.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What dataset was used to train the model?",
        "reference_answer": "The GRID audio-visual corpus was used containing audio and video recordings of 34 speakers with 1000 utterances each. Training used two male speakers S1 and S2 and two female speakers S4 and S29 with an 80-10-10 train-validation-test split.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What loss function was used during training?",
        "reference_answer": "The CorrMSE loss function was used which combines mean squared error and correlation with lambda set to 0.5 to balance the two components.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What is the optimal bottleneck size and why?",
        "reference_answer": "The optimal bottleneck size is 32 nodes. Increasing to 64 improves autoencoder output quality but makes reconstruction harder for the lip reading network. 32 nodes balances this trade-off with 98 percent correlation.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "How does the proposed method compare to Vid2Speech in PESQ score?",
        "reference_answer": "The proposed method achieves an average PESQ of 1.88 compared to Vid2Speech's 1.76 and outperforms it across speakers S1 S2 and S29.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What were the results of the human evaluation on Amazon Mechanical Turk?",
        "reference_answer": "The model achieved 55.8 percent average word recognition accuracy versus 50.9 for Vid2Speech and scored 1.63 versus 1.35 on natural sound quality on a scale of 1 to 5 and 85.1 percent correct gender identification versus 43.2 percent.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What is the effect of using Gaussian noise at the bottleneck?",
        "reference_answer": "Gaussian noise slightly reduces autoencoder output accuracy but significantly improves overall lip reading performance by allowing the autoencoder to handle variations in input videos.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What evaluation metrics were used to assess the network performance?",
        "reference_answer": "Three metrics were used: 2D correlation between reconstructed and actual auditory spectrogram, Perceptual Evaluation of Speech Quality PESQ for audio quality, and Spectro Temporal Modulation Index STMI for intelligibility.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What were the video preprocessing steps applied to the dataset?",
        "reference_answer": "Each video frame was converted to grayscale and normalized. The face region was extracted and resized. Videos were divided into K non-overlapping slices of length Lv and first and second order temporal derivatives were stacked to form a 4D tensor.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What optimizer and learning rate were used for training?",
        "reference_answer": "The Adam optimizer was used with an initial learning rate of 0.0001.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "How does dropout affect autoencoder performance according to the ablation study?",
        "reference_answer": "Using dropout makes results worse. PESQ dropped from 2.81 to 2.33 and Corr2D dropped from 0.98 to 0.95 when dropout was used.",
        "relevant_files": ["machinelearning.pdf"],
    },
    {
        "query": "What are the future directions mentioned in the conclusion?",
        "reference_answer": "Future work includes collecting more training data including emotions in reconstructed speech and proposing an end-to-end structure to directly estimate raw waveform from facial speech-related features.",
        "relevant_files": ["machinelearning.pdf"],
    },
]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", "-p", required=True, help="Folder containing machinelearning.pdf")
    parser.add_argument("--top_k", "-k", type=int, default=5)
    args = parser.parse_args()

    pdf_path = os.path.join(args.root_path, "machinelearning.pdf")
    if not os.path.exists(pdf_path):
        print(f"ERROR: machinelearning.pdf not found at {pdf_path}")
        print("Make sure your PDF is in the folder you specified.")
        exit(1)

    print(f"Running evaluation against: {pdf_path}")
    print(f"Test cases: {len(TEST_CASES)} | Top-K: {args.top_k}\n")

    results = run_eval_suite(
        TEST_CASES,
        root_path=args.root_path,
        top_k=args.top_k,
        specific_files=[pdf_path],
    )
    print_eval_report(results)
