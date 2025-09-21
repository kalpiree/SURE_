## Oracle4Rec

<h2>Overview</h2>

<p>Oracle4Rec is a sequential recommender that learns disentangled representations of past and future user behavior via dual autoencoder branches. It introduces a transition regularizer that aligns the latent embeddings from past and future interactions using an attenuation mechanism. The model is trained phase-wise and can adapt to evolving user-item distributions across temporal segments.</p>

<h2>Project Structure</h2>

<ul>
<li><code>models.py</code> – Contains the full Oracle4Rec model with dual encoders, attention layers, frequency-based filters, and transition loss logic.</li>
<li><code>main.py</code> – Orchestrates phase-wise training and evaluation. Loads data, trains the model with early stopping, and saves scores.</li>
<li><code>trainers.py</code> – Trainer class encapsulating training, validation, and inference logic. Computes HR@K, NDCG@K, and MRR.</li>
<li><code>datasets.py</code> – Implements the <code>Dataset</code> class for both training and evaluation formats, including support for negative sampling.</li>
<li><code>utils.py</code> – Includes data loading, early stopping, seed setting, metric computation, and custom dataloader logic.</li>
</ul>

<h2>How to Run</h2>

<h3>Step 1: Train Oracle4Rec Phase-by-Phase</h3>

<pre><code>python main.py \
  --data_dir ./processed_datasets \
  --output_dir ./output \
  --data_name goodreads \
  --cudaid 0 \
  --epochs 5 \
  --model_name oracle4rec
</code></pre>

<p>This will train the model across 5 phases (per user sequence) and save model checkpoints in <code>output/goodreads_model{X}_phase{Y}/model.pt</code>.</p>


