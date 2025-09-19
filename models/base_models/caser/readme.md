# DAUR
<h1>Step 2: Training Caser Model and Generating Scores</h1>

<h2>Overview</h2>

<p>This step trains the <code>Caser</code> model across multiple user-item datasets and evaluation phases. Each model generates prediction scores which are used in later ranking and evaluation.</p>

<h2>Project Structure</h2>

<ul>
  <li><code>caser.py</code> – Caser model definition</li>
  <li><code>main.py</code> – Script to run training across datasets and bootstraps</li>
  <li><code>train_helpers.py</code> – Training loop logic</li>
  <li><code>evaluation_csv.py</code> – Evaluates model predictions with NDCG/Hit metrics</li>
  <li><code>interactions.py</code> – Handles input formatting and sequence generation</li>
  <li><code>utils.py</code> – Common utilities (minibatching, seed setting, activation)</li>
  <li><code>caser.sh</code> – Bash automation script to run Caser over phases</li>
</ul>

<h2>How to Run</h2>

<h3>Step 1: Install Requirements</h3>

<pre><code>pip install -r requirements.txt
</code></pre>

<h3>Step 2: Execute Training</h3>

<p>To run training over all datasets and bootstraps:</p>

<pre><code>python main.py
</code></pre>

<p>Or use the phase-wise bash automation:</p>

<pre><code>bash caser.sh
</code></pre>

<h3>Step 3: Output</h3>

<p>Each run saves:</p>

<ul>
  <li><code>{save_dir}/phase{}/_model.pth</code> – Checkpoints</li>
  <li><code>{save_dir}/phase{}/_eval_output.csv</code> – Prediction scores</li>
</ul>

<h2>Configuration</h2>

<p>Key arguments in <code>main.py</code> or <code>caser.sh</code>:</p>

<ul>
  <li><code>--phases</code>: Number of training phases (default 5)</li>
  <li><code>--n_iter</code>: Training epochs per phase (default 50)</li>
  <li><code>--batch_size</code>: Batch size (default 512)</li>
  <li><code>--neg_samples</code>: Negative samples per instance (default 3)</li>
  <li><code>--learning_rate</code>: Learning rate (default 1e-3)</li>
  <li><code>--drop</code>: Dropout rate in Caser model (default 0.5)</li>
</ul>

<h2>Example Output Structure</h2>

<pre><code>caser_runs/
└── gowalla_model0_caser/
    ├── phase0_model.pth
    ├── phase0_eval_output.csv
    ├── ...
</code></pre>

