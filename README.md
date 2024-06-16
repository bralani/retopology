# Direction Field Extraction through DiffusionNet for Quad Mesh Retopology

## Members:
- Balice Matteo (10978268)
- Doronzo Antonio Giuseppe (11016435)


## Project Structure
The project structure is organized as follows:

- **crest_lines**: Contains the script in C++ for extracting the principal direction and curvatures for each vertice.
- **neural_network**: Contains the diffusionNet model.
- **dataset**: Includes the dataset used for training and evaluation. (Download it from the link below)

## Setup and Dependencies
To set up the project environment, follow these steps:
1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`:
  ```pip install -r requirements.txt```
4. Download our preprocessed dataset at the following [link](https://polimi365-my.sharepoint.com/:u:/g/personal/10978268_polimi_it/EdHy8Ij3NSpPmQh7nrogHWYB7OizNwVeL_f_Vt6rfnmYbA?e=LtZy1c) and put it in the root folder `/dataset` or use your own dataset.
5. Skip this part if you use the preprocessed dataset, otherwise build the C++ script for extracting the principal direction and curvatures by running the following command (be sure to have installed a c++/g++ compiler):
  ```python setup.py build```. We ran this command using MacOS operating system, so we suggest to use the same OS.

## Usage
Follow the workflow in the notebook `retopology.ipynb` to train the model and extract the direction field.

## License
This project is licensed under the [MIT License](LICENSE).
