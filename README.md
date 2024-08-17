# LeNet5-Implementation

"Gradient Based Learning Applied to Document Recognition" by Yann LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner.

Paper: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf.

## Architecture

![LeNet-5 Architecture](./images/architecture.png)

Implementation of the OG network as shown in figure above is here: [model.py](./model/model.py). There is change in output activation function from the original paper: Euclidean Radial Basis Function to Softmax.

Also, a more modern version of LeNet-5 is here: [modern.py](./model/modern.py).

## Usage

- For OG LeNet5

```bash
python train.py --epochs 10
```

- For Modern LeNet5

```bash
python train.py --epochs 10 --modern
```

## Plots

- After 10 epochs of training, the models showed following results:

    - OG LeNet5

    ![OG LeNet-5 Loss and Accuracy Plots](./results/LeNet5.png)

    - Modern LeNet5

    ![Modern LeNet-5 Loss and Accuracy Plots](./results/LeNet5Modern.png)

## Citation

```
@ARTICLE{726791,
  author={Lecun, Y. and Bottou, L. and Bengio, Y. and Haffner, P.},
  journal={Proceedings of the IEEE}, 
  title={Gradient-based learning applied to document recognition}, 
  year={1998},
  volume={86},
  number={11},
  pages={2278-2324},
  keywords={Neural networks;Pattern recognition;Machine learning;Optical character recognition software;Character recognition;Feature extraction;Multi-layer neural network;Optical computing;Hidden Markov models;Principal component analysis},
  doi={10.1109/5.726791}}
```
