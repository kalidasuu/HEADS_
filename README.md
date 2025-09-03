# HEADS
Extractive-Abstractive Document Summarization with Longformer

This project presents a hybrid document summarization system that integrates extractive and abstractive techniques to produce factually accurate and linguistically coherent summaries of long documents. By leveraging the Longformer model’s ability to process extended sequences and combining it with the fluent text generation capabilities of a modified BART model, our approach addresses key challenges in summarization: extractive methods often produce disjointed outputs, while abstractive methods risk factual inaccuracies and struggle with lengthy inputs. Utilizing the GovReport dataset, we demonstrate significant improvements in summary quality, achieving higher ROUGE scores, better factual consistency, and enhanced semantic alignment compared to baseline models. Our system introduces a novel pipeline that extracts key sentences, transforms them into structured factual constraints, and guides abstractive generation, offering a scalable solution for summarizing complex, long-form documents such as government reports.

We used GovReport dataset for training HEADS. [Click](https://drive.google.com/drive/folders/1dJ-f1vgMDG3R-XWCyWdYF68xXjseEOyS?usp=sharing) here to access the raw dataset. We have generated intermediate dataset to train both Longformer and BART, [click](https://drive.google.com/drive/folders/1pwtrx7N_66hJAMwOgub6IrlembLI0-Uy?usp=drive_link) here to access it.


<img width="499" height="487" alt="image" src="https://github.com/user-attachments/assets/18a22654-ab04-457d-8c77-9c3b65270a2a" />

