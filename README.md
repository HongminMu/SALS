# Illumination Adaptation for SAM to Handle Low-Light Scenes

## Overview
This project focuses on adapting the SAM model to handle low-light scenes more effectively. By addressing illumination issues, our method enhances the performance of semantic segmentation in challenging lighting conditions.

![image](https://github.com/user-attachments/assets/c91e01dd-7493-480f-b5ee-552184cfd2f4)


## Method
Accurate segmentation in low-light scenes is challenging due to severe domain shifts when models trained on daylight data are applied to such scenes and the lack of large-scale fine-grained labels under low-light conditions. To tackle these challenges, we propose the following methods:

- **Self-Training for Source-Free Predictions**: We leverage the generalization capabilities of segmentation foundation models, like SAM, to generate predictions without relying heavily on labeled low-light data. Our self-training method enables SAM to make effective predictions in low-light scenarios.

- **Transformation Head for Feature Enhancement**: To address the domain shift between low-light target data and SAM's natural-light training data, we design a transformation head that enhances low-light features before applying SAM. This head helps in transforming low-light features into a more natural-light-like representation.

- **Domain Shift Compensation Loss**: We introduce a domain shift compensation loss to train our model in selecting an optimal illumination-enhanced feature map. This loss function helps bridge the domain gap between low-light data and SAM's training data, improving segmentation performance.

## Results
Our method significantly outperforms the state of the art on the following datasets:

### Dark Zurich
![Dark_Zurich_ICRA2025](https://github.com/user-attachments/assets/bee5d8ac-5690-44bf-a472-e5f17faa4d44)

### Nighttime Driving
![Nighttime_Driving_ICRA2025](https://github.com/user-attachments/assets/9b150ddc-7538-4169-bfed-812c91ae5df1)

## Code Release
Our code will be released publicly soon! 🚀🚀🚀
Stay tuned for updates and detailed documentation on how to use our method!

## Contact
For any inquiries or feedback, please contact [hongmin_mu@163.com](mailto:hongmin_mu@163.com).
