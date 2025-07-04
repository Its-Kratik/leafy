# Plant Disease Classification

An advanced machine learning system for classifying plant diseases using traditional ML techniques with handcrafted features.

## 🌟 Features

- **Advanced Feature Extraction**: HOG, LBP, Color, Texture, and Shape features
- **Multiple ML Models**: Random Forest, SVM, and Ensemble methods
- **High Accuracy**: Achieves 90%+ accuracy on Plant Pathology 2020 dataset
- **Interactive Web Interface**: User-friendly Streamlit application
- **Production Ready**: Optimized for deployment and real-world usage

## 🚀 Live Demo

[Your Streamlit App URL will go here]

## 🛠️ Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📊 Performance

- **Best Model**: svm
- **Accuracy**: 78.00%
- **Classes**: 4 disease types

## 🔬 Technical Details

- **Feature Engineering**: Multi-scale HOG, LBP, color histograms, Gabor filters
- **Model Architecture**: Ensemble of Random Forest and SVM
- **Dataset**: Plant Pathology 2020 (3,651 images)
- **Deployment**: Streamlit Community Cloud

## 📁 Project Structure
├── app.py # Streamlit application
├── models_production/ # Trained models
├── requirements.txt # Dependencies
└── README.md # Documentation


## 🎯 Usage

1. Upload a plant leaf image
2. Click "Analyze Image"
3. View disease classification results
4. Get treatment recommendations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
