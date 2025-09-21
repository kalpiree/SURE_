
## SASRec

<h2>Overview</h2>

<p>SASRec (Self-Attentive Sequential Recommendation) applies the self-attention mechanism to model user behavior sequences, effectively capturing user preferences over time. This implementation supports phased training for continual learning, with full support for evaluation from dynamic candidate sets.</p>

<h2>Project Structure</h2>

<ul>
<li><code>model.py</code> - Defines the SASRec model with self-attention, feedforward layers, and embedding mechanisms.</li>
<li><code>eval_csv.py</code> - Evaluation from CSV files containing candidate items per interaction. Supports partial candidate sets.</li>
<li><code>main.py</code> - Main script to train and evaluate SASRec across multiple datasets and phases.</li>
<li><code>utils.py</code> - Includes batch sampling logic (WarpSampler), data loading, negative sampling, and evaluation metrics.</li>
<li><code>utils_.py</code> - Variant with similar functionality, used in older or alternate script paths.</li>
</ul>

<h2>How to Run</h2>

<h3>Step 1: Train Across Phases</h3>

<p>Run the following command to start phased training:</p>

<pre><code>python main_.py
</code></pre>

<p>This will train SASRec on each phase (e.g., phase0 to phase4) per dataset/model and save evaluation results in <code>outputs/</code>.</p>



<h2>Details</h2>

<ul>
<li>Uses PyTorch’s <code>MultiheadAttention</code> module.</li>
<li>Employs masked self-attention with learnable positional embeddings.</li>
<li>Supports dropout, layer normalization, and L2 regularization.</li>
<li>Negative sampling and evaluation use 1 positive + 100 sampled negatives.</li>
</ul>

