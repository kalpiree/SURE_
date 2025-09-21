## CDR

<h2>Overview</h2>

<p>CDR (Causal Disentangled Recommendation) is a sequential recommendation model designed for phase-wise training under distribution shifts. It disentangles user preferences (a) from environment-specific representations (z), enabling robust learning across temporal splits. CDR leverages variational inference with disentangled latent variables and structured masking for causal analysis.</p>

<h2>Project Structure</h2>

<ul>
<li><code>models.py</code> – Defines the CDR model architecture, including encoders, decoders, reparameterization, and environment-aware latent structure.</li>
<li><code>main_.py</code> – Trains CDR phase by phase, saving per-phase checkpoints and evaluation results.</li>
<li><code>inference.py</code> – Loads saved CDR models to perform evaluation on held-out phases using per-user candidate sets.</li>
<li><code>data_utils.py</code> – Handles input formatting, sparse matrix creation, and wrapping into PyTorch datasets.</li>
<li><code>evaluate_util.py</code> – Implements metrics: Precision@K, Recall@K, NDCG@K, and MRR.</li>
<li><code>utils.py</code> – Utility functions for model size, sparsity, parameter inspection.</li>
</ul>

<h2>How to Run</h2>

<h3>Step 1: Phase-wise Training</h3>

<p>Train the CDR model across 5 phases for each model and dataset by running:</p>

<pre><code>python main_.py \
  --data_dir processed_datasets \
  --output_dir output_cdr \
  --cuda \
  --gpu 0
</code></pre>



