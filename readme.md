# Personalized Clustering via Targeted Representation Learning

PyTorch implementation of ["Personalized Clustering via Targeted Representation Learning"] (AAAI-25).

The conference paper:https://test.com.

For Appendix of the paper, you can get [here](). 

This repository contains the index.html webpage, which serves as a concise introduction to the methods, experimental results, and data sharing aspects of our research paper. The webpage is designed to offer a clear and accessible summary of our work, enabling readers to quickly grasp the core concepts and findings of the study.

For the convenience of readers who prefer a document format, we have included index.pdf, which is the PDF version of the HTML webpage. This PDF provides the same information in a format that is easy to download and view offline.

The ./assets folder includes all the necessary materials and resources used by the webpage, such as images, stylesheets, and other media files. These assets ensure the webpage is visually informative and engaging, enhancing the overall presentation of the research content.

## 1. Requirements

### Environment Setup

To ensure a consistent and reproducible environment for running this project, we provide an `environment.yml` file that specifies all the necessary dependencies. Follow the steps below to set up the Python environment using Conda:

1. **Install Conda**

2. **Create the Environment:**

   Open a terminal or command prompt and navigate to the directory containing the `environment.yml` file. Run the following command to create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Environment:**

   Once the environment is created, activate it using the following command:

   ```bash
   conda activate <environment_name>
   ```

   Replace `<environment_name>` with the name specified in the `environment.yml` file.

### Datasets

For CIFAR10 and CIFAR100, we provide a function to automatically download and preprocess the data, you can also download the datasets from the link, and please download it to `../dataset` or you can change the path of dataset in file `~/config/config.yaml`.

* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
* [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

For ImageNet-10, we provided its description in the `./dataset` folder.



## 2. Usage

### Configuration

There is a configuration file `./config/config.yaml`, where one can edit both the training and test options. This configuration file allows you to choose between loading a pre-trained model or retraining the model from scratch. Adjust the settings according to your requirements before starting the training.

### Training

After setting the configuration, to start training, simply run

```bash
python train.py
```

Upon completion of the training process, the trained model parameters will be automatically saved in the `./save` directory. Ensure this directory exists to store your model checkpoints.

### Output

At the end of each training epoch and upon program completion, the clustering performance metrics will be outputted. This information provides insights into the effectiveness and progress of the clustering process.



By following these instructions, you can efficiently train and evaluate the model to meet your specific requirements.
