# Direction Field Extraction through DiffusionNet for Quad Mesh Retopology

## Project Structure
The project structure is organized as follows:

- **crest_lines**: Contains the script in C++ for extracting the principal direction and curvatures for each vertice.
- **neural_network**: Contains the diffusionNet model.
- **dataset**: Includes the dataset used for training and evaluation.

## Setup and Dependencies
To set up the project environment, follow these steps:
1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Build the C++ script for extracting the principal direction and curvatures by running the following command
  ```python setup.py build```

## Usage
Follow the workflow in the notebook `retopology.ipynb` to train the model and extract the direction field.

## License
This project is licensed under the [MIT License](LICENSE).