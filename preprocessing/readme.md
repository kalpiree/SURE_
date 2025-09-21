# DAUR Preprocessing
<h1>Step 1: Data Preprocessing </h1>

<h2>Overview</h2>

<p>This module prepares raw user-item interaction data for use in recommendation models. It performs the following steps:</p>

<ul>
  <li>Densifies sparse <code>user_idx</code> and <code>item_idx</code> columns.</li>
  <li>Splits user histories into 5 temporal training/evaluation phases.</li>
  <li>Generates bootstrapped user samples for 10 model instances.</li>
  <li>Outputs structured data ready for training and evaluation.</li>
</ul>

<h2>Project Structure</h2>

<ul>
  <li><code>output/</code> - Folder with raw input CSV files</li>
  <li><code>processed_datasets/</code> - Contains phase-wise processed data</li>
  <li><code>preprocessing.py</code> - Main preprocessing script</li>
  <li><code>requirements.txt</code> - List of Python dependencies</li>
  <li><code>README.md</code> - This documentation</li>
</ul>

<h2>How to Run</h2>


<h3>Step 1: Prepare Input Data</h3>

<p>Place the following CSVs inside the <code>output/</code> folder:</p>

<ul>
  <li><code>bookcrossing.csv</code></li>
  <li><code>gowalla.csv</code></li>
  <li><code>lastfm.csv</code></li>
  <li><code>steam.csv</code></li>
  <li><code>taobao.csv</code></li>
</ul>

<p>Each file must include these columns: <code>user_idx</code>, <code>item_idx</code>, <code>timestamp</code>.</p>

<h3>Step 2: Run Preprocessing Script</h3>

<pre><code>python bootstrap_and_phase_all.py
</code></pre>

<p>This will generate:</p>
<ul>
  <li><code>train_phase{0..4}.txt</code> files</li>
  <li><code>eval_phase{0..4}.csv</code> evaluation sets</li>
  <li>Bootstrapped samples under <code>phased_data/model_{i}/</code></li>
</ul>

<h2>Configuration</h2>

<p>Edit constants inside <code>preprocessing.py</code> as needed:</p>

<ul>
  <li><code>MIN_INTERACTIONS = 100</code></li>
  <li><code>NUM_PHASES = 5</code></li>
  <li><code>EVAL_POINTS = 10</code></li>
  <li><code>NEG_SAMPLES = 50</code></li>
  <li><code>N_MODELS = 10</code></li>
</ul>

<h2>Example Output Structure</h2>

<pre><code>processed_datasets/
└── gowalla/
    ├── global_phases/
    │   ├── train_phase0.txt
    │   └── eval_phase0.csv
    └── phased_data/
        └── model_0/
            ├── train_phase0.txt
            └── eval_phase0.csv
</code></pre>


