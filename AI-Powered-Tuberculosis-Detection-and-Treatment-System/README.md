
# AI-Powered Tuberculosis Detection and Treatment System

A comprehensive AI driven project for tuberculosis detection and analysis using chest X-ray images with ConvNext-v2 backbone, and a Gradio interface.

## Features

- **Image Analysis**: Automated tuberculosis detection from chest X-ray images
- **Severity Assessment**: Determines the severity level of detected tuberculosis
- **Medication Recommendation**: AI-powered medication suggestions using LangChain and FAISS
- **Interactive Chatbot**: Medical assistant for TB-related queries
- **Comprehensive Metrics**: Detailed evaluation with reports

## Project Structure

```
TBX11K_Project/
├── data_utils.py           # Data preprocessing and augmentation
├── model.py               # ConvNext-v2 model architecture and Grad-CAM
├── train.py               # Training script with metrics tracking
├── evaluate.py            # Comprehensive evaluation script
├── inference.py           # Inference and visualization
├── med_recommendation.py  # Medication recommendation engine
├── chatbot.py            # Medical chatbot
├── visualizations.py     # Metrics and visualization utilities
├── app.py                # Gradio web interface
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Gemini API Key**:
   - Set as environment variable: `export GEMINI_API_KEY="your-key-here"`

## Dataset Setup

1. **Download the TBX11K dataset** from [Kaggle](https://www.kaggle.com/datasets/usmanshams/tbx-11)

2. **Extract the dataset** to your desired location

3. **Update the dataset path** in `train.py`:
   ```python
   DATA_DIR = 'path/to/your/TBX11K/TBX11K'
   ```

## Usage

### 1. Training the Model

```bash
python train.py
```

This will:
- Train the ConvNext-v2 model on your dataset
- Generate comprehensive training metrics
- Save the best model as `best_model.pth`
- Create detailed evaluation reports in `./results/` directory



### 2. Running Inference

```bash
python inference.py
```

This demonstrates:
- Single image prediction
- Medication recommendation

### 3. Launching the Web Interface

```bash
python app.py
```

This starts a Gradio web interface with:
- Image upload and analysis
- Medication recommendations
- Interactive medical chatbot

## Model Architecture

- **Backbone**: ConvNext-v2
- **Input**: Grayscale chest X-ray images (224x224)
- **Output**: 3-class classification (Healthy, Sick, TB)

## Data Preprocessing

- **Augmentation**: Rotation, flipping, zooming, affine transforms
- **Normalization**: Intensity scaling to [0, 1] range
- **Resizing**: Standardized to 224x224 pixels

## Evaluation Metrics

The project generates comprehensive evaluation reports including:

- **Classification Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Per-class and macro/micro averages
  
- **Visualizations**:
  - Confusion matrices
  - ROC curves with AUC scores
  - Training/validation loss and accuracy plots
  - Class distribution charts
  - Sample predictions with confidence scores

- **Reports**:
  - JSON format for programmatic access
  - CSV files for data analysis
  - PNG images for presentations

## Medication Recommendation

The system uses:
- **LangChain**: For natural language processing
- **FAISS**: For efficient similarity search
- **Gemini AI**: For intelligent recommendations
- **Medical Knowledge Base**: Curated TB medication information

## Medical Chatbot

Features:
- TB-specific medical knowledge
- Conversational interface
- Safety disclaimers
- Professional medical advice referrals

## Output Format

For each X-ray analysis, the system provides:

1. **Diagnosis**: Healthy/Sick/TB classification
2. **Severity**: Mild/Moderate/Severe assessment
3. **Confidence**: Percentage confidence score
5. **Medications**: Recommended treatment options
6. **Disclaimer**: Important safety information

## Important Notes

### Medical Disclaimer
⚠️ **This model is for research purposes only and should not be used for actual medical diagnosis or treatment. Always consult with a qualified healthcare professional.**



## License

This project is for educational and research purposes. Please ensure compliance with medical data regulations and obtain proper approvals for clinical use.

## Acknowledgments

- TBX11K dataset creators
- Gemini for language model capabilities
- Gradio for the web interface framework

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

