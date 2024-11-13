#FLAKE: Secure Multi-Party Computation (SMPC) with Randomized Encoding protocol

`FLAKE` provides a secure SMPC framework for multiple parties (distributed learning) using randomized encoding for kernel-based machine learning models. It enables multiple parties to collaboratively train kernel-based machine learning models without compromising data privacy in a semi-honest adversary setting. 
THis is the code for the paper: [A Privacy-Preserving Framework for Collaborative Machine Learning with Kernel Methods](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10431639)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-repo/kernel_lib.git
cd kernel_lib
pip install -r requirements.txt
```
Make sure sklearn, numpy, and scipy are installed.

## Functionality

- Random Matrix Generation: Create random matrices for encoding.
- Data Transformation: Transform data with random matrices for secure encoding.
- Custom Kernel Computation: Compute Gram matrices for various kernel methods.

## Client-Server Architecture

The client-server setup is managed with a Bash script (`server_client_classification.sh`), allowing multiple runs with configurable parameters.

### Usage

To start the server-client architecture, use:
```bash
bash server_client_classification.sh <no_of_runs> <base_port>
```

**Arguments**:
- `<no_of_runs>`: Specifies the number of independent runs for model training.
- `<base_port>`: Defines the base port number for starting the server. Each subsequent run will use an incremented port number based on this base.
One can experiment with number of input parties and number of features.
