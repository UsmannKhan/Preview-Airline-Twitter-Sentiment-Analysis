# Airline Twitter Sentiment Analysis: Classical vs Embedding-Based Models

A comparative NLP study on three-class sentiment classification of airline customer tweets. Built models from two distinct paradigms (classical statistical, and embedding-based neural), evaluated them on identical splits with identical metrics, and produced a recommendation grounded in deployment concerns rather than just leaderboard scores.

---

## TL;DR

Built a sentiment classifier on the Twitter US Airline Sentiment corpus, a publicly available dataset of manually-labelled tweets directed at major US airlines. Trained and compared a classical statistical pipeline against an embedding-based neural model, with a probabilistic baseline below both. The neural model won on every metric, but only marginally. The more interesting finding was the trade-off rather than the winner: for most deployment contexts, the simpler model is the better practical choice.

**Stack:** Python, scikit-learn, TensorFlow / Keras, NLTK, pandas, NumPy, Matplotlib, Seaborn
**Scope:** Single Jupyter notebook covering preprocessing, baseline, statistical model, neural model, and analysis. ~2,500-word technical report.
**Context:** NLP module coursework, BSc Computer Science (University of London)

## Code access

Source code and the full technical report live in a private repository. Happy to share full access upon request.

---

## The question

Twitter sentiment analysis has been a standard NLP problem for over a decade, and two dominant approaches have emerged. Classical statistical models built on hand-engineered text features. Embedding-based neural models that learn from pre-trained word representations. Both have strong track records. The interesting practical question isn't which one "wins" on accuracy. The literature has answered that one many times, in many ways. The useful question is how the trade-offs play out on a specific real-world dataset where speed, interpretability, and deployment cost actually matter.

The aim of this project was to make that comparison rigorously, on identical splits with identical metrics, and write up a recommendation grounded in operational concerns.

## The approach

Built a complete NLP pipeline from raw tweets through model evaluation. Text cleaning appropriate for Twitter's informal conventions. Standard tokenisation and normalisation. Class-stratified train-test splitting to preserve the natural class imbalance, which is meaningful in customer-feedback data and shouldn't be flattened. Two parallel feature representations for the two model families. A baseline, a stronger statistical model, and a neural model. Identical evaluation metrics applied to all three.

## The finding

The neural model came out on top across every evaluation metric. The margin over the simpler statistical model was small. Much smaller than the gap between either model and the baseline.

That narrow margin is the actual finding. When the gap between a complex model and a simple one is this tight, model choice should stop being about test-set scores and start being about deployment context. The simpler model trains in a fraction of the time, runs faster at inference, and produces interpretable per-class coefficients that you can sanity-check when something starts misclassifying after a distribution shift. The neural model produces marginally better accuracy and a richer semantic foundation that's useful if you're going to extend into harder tasks downstream.

Same data, same evaluation, two genuinely defensible answers depending on what you need.

## What was hardest

All three models struggled on the same class: the minority class that was also the most linguistically ambiguous. The pattern was consistent across architectures, which made it a property of the task rather than the modelling choice. A better feature representation would only partially solve it. The real fixes live elsewhere: sampling-side techniques (class weights, oversampling) or a more context-aware architecture.

## Stack

Python 3.8+. scikit-learn for the statistical models and evaluation. TensorFlow / Keras for the neural model. NLTK for text preprocessing. pandas and NumPy for data handling. Matplotlib and Seaborn for visualisation. Pre-trained word embeddings from Stanford NLP.

---

*Built as coursework for the NLP module, BSc Computer Science (University of London).*
