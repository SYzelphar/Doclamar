"""Distractor documents placed next to the target paper during evaluation.

Several are deliberately close to the paper's vocabulary (spectrograms, PESQ,
LSTM, Adam, dropout) so that file routing has to work on meaning, not keywords.
"""
from __future__ import annotations

from pathlib import Path

TEXT_FILES = {
    "notes/cnn_training_notes.md": """# Training notes: image classifier

## Setup
We fine-tuned a ResNet-50 convolutional neural network on a 12-class product image dataset.
Images were resized to 224x224 and augmented with random crops and horizontal flips.

## Optimizer
We used SGD with momentum 0.9 and an initial learning rate of 0.01, decayed by a factor of
10 every 30 epochs. A run with the Adam optimizer at learning rate 0.001 converged faster
but generalised slightly worse on the validation split.

## Regularisation
Dropout of 0.5 before the final fully connected layer reduced overfitting. Batch
normalisation was kept in train mode for the first five epochs only.

## Results
Top-1 accuracy reached 91.4 percent on the held-out test set after 90 epochs.
""",
    "notes/speech_enhancement_survey.txt": """A short survey of deep learning for speech enhancement

Speech enhancement aims to remove background noise from recorded speech. Most modern
systems operate on a short-time Fourier transform magnitude spectrogram and predict a
time-frequency mask. Recurrent models such as LSTM networks capture temporal context,
while convolutional encoder-decoder networks capture local spectral patterns.

Evaluation commonly reports PESQ (perceptual evaluation of speech quality) and STOI
(short-time objective intelligibility). Typical gains over noisy input are 0.5 to 0.9
PESQ points on the VoiceBank-DEMAND benchmark.

Open problems include generalisation to unseen noise types, low-latency processing for
hearing aids, and avoiding musical noise artefacts introduced by aggressive masking.
""",
    "notes/transformer_reading_notes.md": """# Reading notes: transformers

The transformer replaces recurrence with self-attention. Each layer contains multi-head
attention followed by a position-wise feed-forward network, with residual connections and
layer normalisation. Positional encodings inject word order.

BERT pre-trains a bidirectional encoder with masked language modelling. GPT models
pre-train a decoder to predict the next token and are fine-tuned or prompted for tasks.

Scaling laws suggest loss falls predictably with model size, data and compute.
""",
    "business/q3_sales_report.txt": """Quarterly sales report - Q3

Revenue grew 8 percent quarter over quarter to 4.2 million, driven by the enterprise
segment. Gross margin held at 61 percent. Customer churn fell to 2.1 percent after the
onboarding redesign. The sales team added 14 new enterprise accounts, mostly in the
logistics vertical. Next quarter we will focus on renewals and the partner channel.
""",
    "personal/pancakes.md": """# Fluffy pancakes

Whisk 200 g flour, 2 tsp baking powder, a pinch of salt and 1 tbsp sugar. In another bowl
beat 1 egg with 300 ml milk and 25 g melted butter. Combine without overmixing, rest the
batter for 10 minutes, then cook ladlefuls on a hot buttered pan until bubbles form.
Flip once and serve with maple syrup.
""",
}

MEETING_NOTES = [
    ("Lab meeting - project kickoff", True),
    ("Attendees: four students and the advisor.", False),
    ("We discussed candidate topics for the semester project, including audio-visual "
     "learning and document question answering. Each student will read two papers before "
     "the next meeting and present a one-slide summary.", False),
    ("Action items", True),
]
MEETING_TABLE = [
    ("Owner", "Task", "Due"),
    ("Student A", "Collect candidate datasets", "Friday"),
    ("Student B", "Set up GPU environment", "Monday"),
]


def write_distractors(root: Path) -> None:
    for rel, text in TEXT_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    from docx import Document

    doc = Document()
    for text, heading in MEETING_NOTES:
        doc.add_heading(text, level=2) if heading else doc.add_paragraph(text)
    table = doc.add_table(rows=0, cols=3)
    for row in MEETING_TABLE:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = value
    (root / "notes").mkdir(exist_ok=True)
    doc.save(root / "notes" / "lab_meeting.docx")

    # An image-only PDF: should be reported as "no extractable text", not crash anything.
    import pypdfium2 as pdfium

    (root / "scans").mkdir(exist_ok=True)
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(612, 792)
    pdf.save(str(root / "scans" / "scanned_receipt.pdf"))
    pdf.close()
