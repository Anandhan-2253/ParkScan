<h1 align="center">🧠 Parkinson's Disease Detection using EMG & FSR Data</h1>

<p align="center">
  <i>A deep learning model for detecting Parkinson's disease using EMG and Force Sensitive Resistor sensor data</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.16.1-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/scikit--learn-1.7.1-green?logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License">
</p>

---

<h2>📋 Overview</h2>

<p>This project implements a machine learning model for detecting Parkinson's disease using EMG (Electromyography) and FSR (Force Sensitive Resistor) sensor data. The model uses a combination of CNN and LSTM layers to analyze muscle activity and pressure patterns that are characteristic of Parkinson's disease.</p>

---

<h2>✨ Features</h2>

<ul>
  <li><strong>Deep Learning Architecture</strong>: Combines Conv1D and LSTM layers for time-series analysis</li>
  <li><strong>Multi-sensor Data</strong>: Analyzes EMG signals from Tibialis Anterior and Gastrocnemius muscles</li>
  <li><strong>Pressure Analysis</strong>: Incorporates FSR data from Heel, Great Toe, First Metatarsal, and Fifth Metatarsal</li>
  <li><strong>Comprehensive Evaluation</strong>: Provides accuracy metrics, classification reports, and confusion matrices</li>
  <li><strong>Visualization</strong>: Training history plots and confusion matrix visualizations</li>
  <li><strong>Binary Classification</strong>: Healthy vs Parkinson's Disease detection</li>
</ul>

---

<h2>🏗️ Project Structure</h2>

<pre>
parkinson_ML/
├── README.md
├── requirements.txt
├── train.py                      # Main training script
├── debugging/                    # Debug files and logs
├── model/
│   └── model.h5                 # Trained model saved here
├── data/
│   └── cleaned_parkinson_data.csv # Dataset
└── venv/                        # Virtual environment
</pre>

---

<h2>🚀 Quick Start</h2>

<h3>Prerequisites</h3>
<ul>
  <li>Python 3.8 or higher</li>
  <li>pip package manager</li>
</ul>

<h3>Installation</h3>

<p><strong>1. Clone the repository</strong></p>
<pre><code>git clone https://github.com/Anandhan-2253/parkinson-disease-detection.git
cd parkinson-disease-detection</code></pre>

<p><strong>2. Create and activate virtual environment</strong></p>
<pre><code>python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate</code></pre>

<p><strong>3. Install dependencies</strong></p>
<pre><code>pip install -r requirements.txt</code></pre>

<p><strong>4. Run the training script</strong></p>
<pre><code>python train.py</code></pre>

---

<h2>📊 Data Features</h2>

<p>The model analyzes the following sensor data:</p>

<h3>🔬 EMG Sensors</h3>
<ul>
  <li><strong>Tibialis Anterior</strong>: Muscle activity from the shin area</li>
  <li><strong>Gastrocnemius</strong>: Muscle activity from the calf area</li>
</ul>

<h3>⚡ FSR Sensors (Force Sensitive Resistors)</h3>
<ul>
  <li><strong>Heel</strong>: Pressure data from heel area</li>
  <li><strong>Great Toe</strong>: Pressure data from big toe</li>
  <li><strong>First Metatarsal</strong>: Pressure from first metatarsal bone area</li>
  <li><strong>Fifth Metatarsal</strong>: Pressure from fifth metatarsal bone area</li>
</ul>

---

<h2>🤖 Model Architecture</h2>

<table align="center">
  <tr>
    <th>Layer</th>
    <th>Type</th>
    <th>Parameters</th>
  </tr>
  <tr>
    <td>Input</td>
    <td>Conv1D</td>
    <td>64 filters, kernel_size=1, ReLU</td>
  </tr>
  <tr>
    <td>Pooling</td>
    <td>MaxPooling1D</td>
    <td>pool_size=1</td>
  </tr>
  <tr>
    <td>Regularization</td>
    <td>Dropout</td>
    <td>rate=0.3</td>
  </tr>
  <tr>
    <td>Sequential</td>
    <td>LSTM</td>
    <td>64 units</td>
  </tr>
  <tr>
    <td>Regularization</td>
    <td>Dropout</td>
    <td>rate=0.3</td>
  </tr>
  <tr>
    <td>Output</td>
    <td>Dense</td>
    <td>1 unit, Sigmoid activation</td>
  </tr>
</table>

---

<h2>📈 Performance Metrics</h2>

<p>The model provides comprehensive evaluation through:</p>

<ul>
  <li><strong>🎯 Binary Classification</strong>: Healthy (0) vs Parkinson's Disease (1)</li>
  <li><strong>📊 Accuracy Score</strong>: Model performance on test data</li>
  <li><strong>📋 Classification Report</strong>: Precision, Recall, F1-score for both classes</li>
  <li><strong>🧩 Confusion Matrix</strong>: Visual representation of prediction accuracy</li>
</ul>

---

<h2>🔧 Configuration</h2>

<p>Key parameters can be modified in <code>train.py</code>:</p>

<pre><code># Model parameters
DATA_PATH = 'synthetic_parkinson_dataset_100.csv'
MODEL_SAVE_PATH = 'model/parkinson_emg_fsr_model.h5'

# Training parameters
EPOCHS = 25
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2</code></pre>

---

<h2>📚 Key Dependencies</h2>

<table align="center">
  <tr>
    <th>Category</th>
    <th>Package</th>
    <th>Version</th>
  </tr>
  <tr>
    <td rowspan="4">Core ML</td>
    <td>tensorflow</td>
    <td>2.16.1</td>
  </tr>
  <tr>
    <td>scikit-learn</td>
    <td>1.7.1</td>
  </tr>
  <tr>
    <td>pandas</td>
    <td>2.3.1</td>
  </tr>
  <tr>
    <td>numpy</td>
    <td>1.26.4</td>
  </tr>
  <tr>
    <td rowspan="2">Visualization</td>
    <td>matplotlib</td>
    <td>3.10.3</td>
  </tr>
  <tr>
    <td>seaborn</td>
    <td>0.13.2</td>
  </tr>
</table>

---

<h2>💻 Usage Example</h2>

<pre><code># Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Build model
model = build_model((X_train.shape[1], X_train.shape[2]))

# Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=8)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")</code></pre>

---

<h2>🎨 Visualizations</h2>

<p>The project generates several visualizations:</p>

<ol>
  <li><strong>📈 Training History Plot</strong>: Shows accuracy progression during training</li>
  <li><strong>🧩 Confusion Matrix</strong>: Heatmap showing prediction accuracy by class</li>
  <li><strong>📊 Sample Distribution</strong>: Bar chart of healthy vs Parkinson's samples</li>
</ol>

---

<h2>🤝 Contributing</h2>

<ol>
  <li>Fork the repository</li>
  <li>Create your feature branch (<code>git checkout -b feature/AmazingFeature</code>)</li>
  <li>Commit your changes (<code>git commit -m 'Add some AmazingFeature'</code>)</li>
  <li>Push to the branch (<code>git push origin feature/AmazingFeature</code>)</li>
  <li>Open a Pull Request</li>
</ol>

---

<h2>⚠️ Important Disclaimer</h2>

<blockquote>
  <p><strong>⚕️ Medical Disclaimer:</strong> This project is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with healthcare professionals for medical advice.</p>
</blockquote>

---

<h2>📝 License</h2>

<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

---

<h2>📧 Contact & Acknowledgments</h2>

<p align="center">
  <strong>Author:</strong> Anandhan<br>
  <strong>Project Link:</strong> <a href="https://github.com/Anandhan-2253/parkinson-disease-detection">https://github.com/Anandhan-2253/parkinson-disease-detection</a>
</p>

<h3>🙏 Acknowledgments</h3>
<ul>
  <li>Research community working on Parkinson's disease detection</li>
  <li>EMG and FSR sensor data analysis methodologies</li>
  <li>TensorFlow and scikit-learn communities</li>
</ul>

---

<p align="center">
  <strong>⭐ If you found this project helpful, please consider giving it a star!</strong>
</p>