
# DAUR
<h1>Step 2: Train FMLPRec Model and Generate Scores</h1>

<h2>Overview</h2>

<p>This module trains the <code>FMLPRec</code> model on phased user-item interaction datasets using filter-enhanced self-attention. It supports multi-phase training, score generation, and evaluation with ranking metrics.</p>

<h2>Project Structure</h2>

<ul>
  <li><code>main_.py</code> – Full pipeline to train on DAUR-preprocessed datasets</li>
  <li><code>models.py</code> – Defines the FMLPRec network</li>
  <li><code>modules.py</code> – Implements attention, filters, and feed-forward blocks</li>
  <li><code>trainers.py</code> – Trainer class with custom losses and metric tracking</li>
  <li><code>evaluation_csv.py</code> – Phase-based evaluation on candidate item scores</li>
  <li><code>datasets.py</code> – Dataset class with dynamic negative sampling</li>
  <li><code>utils.py</code> – Early stopping, dataloaders, and evaluation tools</li>
</ul>

<h2>How to Run</h2>


<h3>Step 1: Train on Phased Datasets</h3>

<p>Run the multi-phase training pipeline:</p>

<pre><code>python main_.py --data_name gowalla --output_dir output/
</code></pre>


<h3>Step 2: Output</h3>

<p>Each run saves:</p>

<ul>
  <li><code>{output_dir}/phase{}/_model.pth</code> – Model checkpoint</li>
  <li><code>{output_dir}/phase{}/_eval_output.csv</code> – Prediction scores</li>
</ul>

<h2>Configuration</h2>

<p>Important training arguments:</p>

<ul>
  <li><code>--hidden_size</code>: Embedding and layer size</li>
  <li><code>--max_seq_length</code>: Length of user history to encode</li>
  <li><code>--batch_size</code>, <code>--epochs</code>, <code>--lr</code>: Standard training params</li>
  <li><code>--no_filters</code>: If set, disables the Fourier-based filtering</li>
  <li><code>--do_eval</code>: Enables evaluation-only mode in <code>main_.py</code></li>
</ul>

<h2>Example Output Structure</h2>

<pre><code>fmlp_runs/
└── gowalla_model0_fmlprec/
    ├── phase0_model.pth
    ├── phase0_eval_output.csv
    ├── ...
</code></pre>

