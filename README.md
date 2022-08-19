# Deep Euler Method Tests
Codes for testing the Deep Euler Method (DEM). The Deep Euler Method is a numerical integration scheme which takes advantage of a neural network to approximate the local truncation error of the Euler Method. By adding the approximation to the Euler-step a higher order solution is achieved.

Read about the Deep Euler Method in the paper:
[Deep Euler method: solving ODEs by approximating the local truncation error of the Euler method](https://arxiv.org/abs/2003.09573).

The same scheme was proposed in the paper [Hypersolvers: Toward Fast Continuous-Depth Models](https://papers.nips.cc/paper/2020/hash/f1686b4badcf28d33ed632036c7ab0b8-Abstract.html)

In this repository the Deep Euler Method is tested on two different equations: the Lotka-Volterra equation and the Van der Pol equation. The aim of the tests is to compare the solutions got by using the Euler, the Deep Euler and the Dormand--Prince method. No attempt is made to compare the computational costs.

The whole process of training data generation, neural network training and ODE solution are all implemented in Python. 
Additionally, the Deep Euler Method is also implemented in C++. This demonstrates the viability of a C++ DEM implementation, but the code is far from optimized.

## Running the Codes

The codes are written in Python (some in Jupyter notebooks, some as standalone scripts). No matter which equation you want to test, the process is the following:
1. Install the [necessary packages](#necessary-packages).
1. Go to the test's root folder: `Lotka` for Lotka-Volterra, `VanDerPol` for Van der Pol. :slightly_smiling_face:
1. Create the empty folders: `build`, `data`, `training`, `simulations` (within Lotka or VanDerPol).
1. Generate learning data using the `data_gen_*` jupyter notebooks. After all the cells ran, a `.hdf5` file will be available in the `data` folder.
1. Train a neural network model with `dem_train.py`. The trained model will be available in the `training` folder with several files about the training. The file `traced_model_*.pt` is the trained neural network. The `scaler*.psca` is the data scaling. These are necessary for DEM. In case of Lotka, there is no scaling, no scaler file. More about training [here](#training).
1. Integrate the ODE with Euler, Dormand--Prince and DEM using an `*_figures` jupyter notebook. Now you need your traced model and its scaler to use DEM. 
1. Ready, view the plots :relaxed:

Alternatively, you can test the trained Deep Euler Method in C++ code. Even in this case the data generation and training should still be carried out in Python (1-5). Using the C++ code simulate the ODE with DEM. This outputs the simulation data into the `simulations` folder, which can be read and plotted with an `*_figures` notebook. On how to set up the C++ environment consult [this section](#running-the-c-codes).

### Training
First you need to have data to train a neural network, so use a `data_gen_*` jupyter notebook to generate learning data. 

Training is done by `dem_train.py`. It accepts command line arguments, in order to view the options, enter
```
python dem_train.py --help
```
The most important options are `epoch` and `batch`. It is advised to always explicitly provide them. A useful combination of arguments is the following.
```
python dem_train.py --name MyModel --epoch 5000 --early_stop --batch 100 --save_plots --print_epoch 50 --print_losses 50 --data data/vdp_data.hdf5
```
This tells the script to name the model *MyModel*, run for 5000 epochs at most, but use early stop. The batch size should be 100, and the plots of the learning and validation losses should be saved. Moreover after every 50 epochs the number of the current epoch and the current training and validation losses are printed to the console. The training data is loaded from the given file. Altough the default data path might work, it is recommended to always set the data path with the option `--data`. If you do not want the script to use all available CPU cores set the number of cores with `--num_threads`.

The script outputs several files. The following is a list of the output files. In the following naming patterns `[yymmdd]` indicates the date in year-month-day format. The expression e[0] means the number of epochs the model was trained. Examples: `e400` for 400 epochs, `e23` for 23 epochs. `[name]` is the optional name of the model.
* `[name]_[yymmdd].log`: Log file. It includes the most important informations about the training and all the training and validation losses.
* `model_[name]_e[0]_[yymmdd].pt`: Trained model in Pytorch's own format. It also includes information about the optimizer and the number of epochs, so it can be used to continue a training.
* `traced_model_[name]_e[0]_[yymmdd].pt`: Trained model in Pytorch's jit format (aka traced model). This file can be loaded into Python and also into C++.
* `scaler_[name]_[yymmdd].psca`: File containing the input and output data scalers of the model. This is a normal text file, named `.psca` by me (**P**ytorch Model **Sca**ler). It contains the important attributes of the scikit-learn scaler objects in plain text. It can be loaded into Python and C++.
* `learning_curve_[name]_[yymmdd].png`: The plot of the training and validation losses.

The first three files are always saved, the plots only when `--save_plots` is set. You can create plots later by training for 0 epochs and setting `--load_model` to your model file (not trace model file).

### Integration and comparison

The `*_figures.ipynb` notebooks can be used to integrate the ODEs with the Dormand--Prince (`scipy.integrate.solve_ivp` default method), the Euler (defined in `model/Euler.py`) and the Deep Euler methods (defined in `model/DEM.py`). In order to use them, the trained network and its scaler (if applicable) need to be referenced. See the notebook for more info. Note, that the same notebooks can be used to read and plot the results of the C++ simulations.


### Running the C++ Codes
*CMake* and *Libtorch* (Pytorch C++) are necessary for this section. Libtorch has two distributions, *release* and *debug*, both should work. 
The following steps apply to Windows users. On Mac or Linux please consult the CMake documentation.

Install *Microsoft Visual Studio C++*, *CMake* and *Libtorch*, then go to the `build` folder. (Create it, if you haven't already done so.) 
Run the command 
```
cmake -DCMAKE_PREFIX_PATH=your/path/to/libtorch ..
```
The trailing .. is important, since the `CMakeLists.txt` file searched by CMake is located there. 

Open the created project in Visual Studio. 

Set the project 'VdP' (or 'DEM') to be the start project in the right-click menu. 

Change the run configuration to Debug or Release depending on the used *Libtorch* version. 

Rewrite the variables `model_file` and `scaler_file` (only for VdP) to the respective files you got from training. You should be able to run the code now. The result of the Deep Euler integration will be available in the `simulations` folder. (Unless you changed the `file_name` variable, of course.) To view the results use a `*_figures` jupyter notebook.

### Hyperparameter Tuning
This section only applies to the Van der Pol equation. The file `hyperopt.py` is a hyperparameter optimization script. By running it, a tuned neural network topology can be found. The optimization is done with the help of the Optuna package, check it out to understand its workings: https://optuna.readthedocs.io . The optimization script is written to be deterministic, so the tuning should be reproducible.

The optimized topology I got is implemented in `models/MLPs.py` as `OptimizedMLP`. It is the default model in case of the Van der Pol equation. For the Lotka-Volterra equation `SimpleMLP` is used, which is an 8 layer neural network with 80 neurons in every layer. This was recommended in the Deep Euler [paper](https://arxiv.org/abs/2003.09573).

The file `hyperopt_rect.py` is also an optimization script. It seeks the best neural network among 'rectangular' networks, which have the same number of neurons in every layer.

### Necessary Packages
The necessary Python packages are the following:
* numpy
* matplotlib
* h5py
* scipy
* scikit-learn
* Pytorch: pytorch, torchaudio, torchvision. (More on installation [here](https://pytorch.org/get-started/locally/))
* cudatoolkit (to use gpu, otherwise unnecessary)
* optuna
* plotly
* jupyter

The C++ codes are trickier, you need *CMake* and the Pytorch C++ distribution aka *Libtorch*. *Libtorch* has a *debug* and a *release* distribution, both are supposed to work. On Windows you also need to install *Microsoft Visual Studio C++*. Read more in [this section](#running-the-c-codes).
