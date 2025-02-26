# E-Commerce Recommender System

## Output Image:
<img width="812" alt="Screenshot 2025-02-26 at 1 00 05 AM" src="https://github.com/user-attachments/assets/4d416747-ce56-4b9d-bda4-5a3cc9ebc5f7" />

## Project Overview  
In today's digital landscape, e-commerce platforms thrive on **personalized recommendations** to enhance user experience and drive sales. This project implements a **Hybrid recommendation system** that combines **Content-Based Filtering** and **Collaborative Filtering** techniques to provide intelligent and highly relevant product recommendations.

### Key Features  
* **Text-Based Recommendations** – Uses **TF-IDF** & **Word2Vec** to analyze product descriptions.  
* **Image-Based Recommendations** – Employs **CNN (VGG-16)** to detect visual similarities in product images.  
* **Collaborative Filtering** – Uses **Pearson Correlation** & **Cosine Similarity** to recommend products based on user preferences.  
* **Hybrid Model** – Integrates multiple filtering methods to generate **more accurate** recommendations.  
* **Optimized Data Processing** – Handles **data acquisition, cleaning, and duplicate removal** for enhanced performance.  

---

## Repository Structure  

```
E-Commerce-Recommender-System
├── Main Code               # LaTeX files & output images
│   ├── report.tex            # Project report in LaTeX format
│   └── output_images/        # Screenshots and experiment results
├── FinalCode               # Main implementation code
│   ├── data_preprocessing.py # Data cleaning & preprocessing logic
│   ├── content_filtering.py  # TF-IDF & Word2Vec-based filtering
│   ├── image_filtering.py    # CNN-based image similarity calculations
│   ├── collaborative.py      # Collaborative filtering (Pearson & Cosine Similarity)
│   ├── hybrid_recommender.py # Final hybrid model combining all techniques
│   ├── utils.py              # Helper functions
│   ├── requirements.txt      # Dependencies for the project
├── Project Report.pdf      # Complete project documentation
├── E-Commerce Recommendation Systems.pptx  # Project presentation slides
├── Research paper.pdf      # Reference research paper
├── README.md               # Project documentation
├── .gitattributes          # Git metadata
└── Main Code.zip           # Compressed folder containing main project files
```

---

## Installation & Setup  

### Clone the Repository  
```sh
git clone https://github.com/your-username/recommender-system.git
cd recommender-system/FinalCode
```

## Methodology

### Data Collection & Preprocessing
- Extracted product data using Amazon's Product Advertising API.
- Data Cleaning: Removed null values, duplicates, and unnecessary attributes.
- Text Processing: Applied tokenization, stopword removal, and feature extraction using TF-IDF & Word2Vec.

### Recommendation Techniques

#### Content-Based Filtering:
- TF-IDF & Word2Vec – Extracts product similarities based on textual attributes.
- Image-Based Filtering – Uses CNN (VGG-16) for visual similarity detection.

#### Collaborative Filtering:
- User-User Similarity – Finds users with similar purchase behaviors.
- Item-Item Similarity – Identifies products frequently bought together.

#### Hybrid Model:
- Merges text-based, image-based, and rating-based similarities.
- Provides more accurate and personalized recommendations.

## Results & Performance

### Results from Various Techniques

#### Text-Based Recommendations (TF-IDF + Word2Vec)
- Example Query: "Sleeveless Printed Vest Women Size XL"
- Top Recommendations:
  1. "2014 Sleeveless Heart Breaker Printed Vest Tank Women Size XL"
  2. "Summer Chiffon Sleeveless Shirt Vest Tank Top - Pink XL"
  3. "Bestpriceam Sandistore Womens Summer Lace Chiffon Vest Tank Tops XL"

#### Image-Based Recommendations (CNN - VGG16)
- Example Query Image: (Women's Top - Red & Black Stripes)
- Top Image Matches:
  1. Women's Sleeveless Floral Tank
  2. Striped Sleeveless Cotton Vest
  3. Chiffon Summer Lace Blouse

#### Collaborative Filtering Results
- Example Product: "Nike Running Shoes - Black"
- Top Recommended Items Based on User Behavior:
  1. "Adidas Ultraboost Running Shoes"
  2. "Puma Sports Sneakers"
  3. "Nike Compression Socks"

### Performance Evaluation
- Hybrid filtering significantly improves recommendation accuracy and diversity.
- TF-IDF + Word2Vec enhances text-based matching for precise suggestions.
- CNN (VGG-16) effectively groups visually similar products.

## Future Scope
- Incorporate user reviews for sentiment-based recommendations using NLP.
- Implement real-time recommendation API for seamless integration with e-commerce platforms.
- Improve scalability for handling large-scale datasets efficiently.
- Explore reinforcement learning to refine recommendations based on user interactions.

## Tech Stack
- **Programming Language**: Python
- **Libraries & Frameworks**:
  - **Machine Learning**: scikit-learn, TensorFlow
  - **Data Processing**: NumPy, Pandas, NLTK, SciPy
  - **Visualization**: Matplotlib, Seaborn
  - **Deep Learning Model**: CNN (VGG-16)

## Contributors
- Gummadi Manoj Kumar
- Vamsi Krishna
- Vuchuru Purushotham
