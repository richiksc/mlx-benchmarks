# MLX and PyTorch, Benchmarked
**Author: Richik Sinha Choudhury <sinhachoudhu (at) wisc (dot) edu>**

**Date: 11th December 2023**

Apple recently released the [MLX framework](https://github.com/ml-explore/mlx),
a Python array and machine learning framework designed for Apple Silicon Macs (M1, M2, etc).

Apple touts that MLX takes advantage of Apple Silicon's [unified memory architecture](https://en.wikipedia.org/wiki/Graphics_processing_unit#Integrated_graphics),
enabling training and inference on CPU and GPU without incurring the cost of copying data.
This may make a larger difference with smaller models, where the constant-time overhead of
data copying may negate any parameter count-dependent gains in computation.

To verify this, I benchmarked MLX performance for small models with popular architectures (MLP and Transformer) on CPU and GPU against [PyTorch](https://pytorch.org/), the most popular framework in the machine learning community.

These benchmarks are based on sample code released by Apple at [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples).

## Running benchmarks

The benchmarks were developed and run with Python 3.10.3.

Ensure you have the required packages installed:
```sh
python -m pip install -r requirements.txt
```

To benchmark the [MNIST](mnist) example:

```sh
cd mnist

# For MLX
python benchmark_mlx.py -n <number of training runs> -e <number of epochs> [--gpu]

# For PyTorch
python benchmark_torch.py -n <number of training runs> -e <number of epochs> [--gpu]
```
Where `<number of training runs>` indicates the number of times to train a model from scratch,
and `<number of epochs>` indicates the number of training epochs per training run.
Therefore, the total number of training epochs is `n * e`.

Similar benchmark scripts are available for the [Transformer language model](transformer_lm) example. Please note they must be ran with `--eval` to benchmark inference.


This will produce a CSV file, e.g. `mlx-train-cpu-n100-e10.csv`, with `n` rows, one per training run.

The first `e` columns are the elapsed training time for epochs 0 through `e`.
The second to last column indicates the mean epoch time. The final column is
the total training time for that training run, including model loading.

## Results

Benchmarks were run on a 13" MacBook Pro (M1, 2020) with 16 GB of RAM. The system load was controlled, with only the benchmark script, Visual Studio Code, Activity Monitor, Google Chrome (<3 tabs), and standard background processes running.

MLX benchmarks were evaluated on the `gpu` and `cpu` devices, and PyTorch benchmarks were evaluated on the `cpu` and `mps` (Metal Performance Shaders, GPU) backends.

In addition to the CSV files included under `results/` directories in `mnist` and `transformer_lm`, a [Google Sheet](https://docs.google.com/spreadsheets/d/17Rid-DTGz_0k8TLOxUJ2Y3_7QKAOVB2MlagqA5BjNro/edit?usp=sharing) is available with all the data and relevant summaries and charts.

### MNIST
Code: [mnist](mnist)

Evaluated the training and inference performance of a simple MLP model trained on the MNIST dataset with 2 hidden layers of 32 neurons and the ReLU activation function. Each training epoch processed all 60,000 training images with a batch size of 256.

In addition to measuring throughput and performance, I've also included CPU and GPU usage statistics to compare the compute efficiency of the frameworks.

**ℹ️ Note:** the PyTorch sample code has been modified to move each batch to the target device for every batch iteration,
to simulate using a `DataLoader`. The provided code moved the entire train and test dataset to the target device at the beginning, which is not possible using `Dataset` and `DataLoader` in a typical PyTorch scenario.


- Number of train/test iterations (`n`): 100
- Number of epochs per iteration  (`e`): 20

#### MLX CPU
- Sustained CPU usage: 101%
- CPU Time: 3:27.31
- GPU usage: 0%
- GPU Time: 0.0

#### MLX GPU
- Sustained CPU usage: 88% (50-95%)
- CPU time: 9:59.66
- Sustained GPU Usage: 98%
- GPU Time: 8:09.73

#### PyTorch CPU
- Sustained CPU usage: 100%
- CPU Time: 3:23.37
- GPU usage: 0%
- GPU Time: 0.0

#### PyTorch GPU

#### Charts

#### Conclusions

Training the model on the M1's GPU with MLX severely limits throughput,
even when


### Transformer Language Model
Code: [transformer_lm](transformer_lm)


