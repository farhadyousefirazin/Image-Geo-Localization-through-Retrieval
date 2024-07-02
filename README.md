## Image-Geo-Localization-Through-Retrieval

In this study, conducted in July 2024 as part of the Masters of Data Science and Engineering in MLDL course at Politecnico di Torino, we explored various techniques to enhance the performance of a visual geo-localization model using deep learning.

Our focus was on improving recall@N scores across different datasets and evaluating the impact of key components such as loss functions, optimization algorithms, learning rate schedulers, and data augmentations.

Key findings from our experiments include:

- **Impact of Components**: The inclusion of the Generalized Mean (GeM) pooling layer significantly improved feature extraction. Combined with the Circle Loss function, this setup achieved the best recall@N scores.
  
- **Optimization Insights**: AdamW optimizer with an appropriate learning rate and weight decay provided stable convergence and improved performance compared to other algorithms.
  
- **Learning Rate Schedulers**: Cosine Annealing Scheduler effectively managed learning rates, contributing to better convergence.

- **Data Augmentations**: Nightlight transformation, horizontal flipping, and color jitter with contrast adjustment enhanced model robustness, especially for challenging datasets like Tokyo-XS.

- **Overall Performance**: Our optimized model achieved competitive recall@N scores on SF-XS and Tokyo-XS test datasets, demonstrating its effectiveness in real-world geo-localization tasks.

This research was conducted by **Farhad Yousefi Razin, Hamed Goldoust, and Shayan Bagherpour** as part of their Master's studies at Politecnico di Torino. The study highlights the importance of component selection and tuning in deep learning models for visual geo-localization. Through advanced techniques in feature extraction, optimization, and data augmentation, significant improvements in model performance were achieved, promising more accurate applications in geographic image analysis and recognition.

