---
author: Pietro Alovisi
title: Assignment 3 - Report
header-includes: \usepackage{subcaption} \usepackage{graphicx}
---


# Gradient Check

To check the gradient I computed for each layer the maximum of the absolute difference of the gradients, as in the equations below:

$$
max(|W_i-W_{i\ num}|)
$$
$$
max(|b_i-b_{i\ num}|)
$$

Where $\cdot_{num}$ represents the numerical computed value. The

| Inner nodes $\rightarrow$ | $[50]$   | $[20,10]$  | $[10,10,10]$ |
|---------------------------|----------|------------|--------------|
| $W_1$                     | 7.01e-10 |  8.33e-10  |   6.71e-10   |
| $b_1$                     | 3.49e-10 |  2.33e-10  |   3.18e-10   |
| $W_2$                     | 5.51e-10 |  5.48e-10  |   4.97e-10   |
| $b_2$                     | 3.12e-10 |  2.86e-10  |   3.18e-10   |
| $W_3$                     | -        |  4.58e-10  |   5.77e-10   |
| $b_3$                     | -        |  3.05e-10  |   1.80e-10   |
| $W_4$                     | -        | -          |   4.90e-10   |
| $b_4$                     | -        | -          |   3.36e-10   |

Table: Maximum absolute difference between the computed gradients for the parameter of each layer (first column) for different network configuration. The lists like $[10,10,10]$ mean that there are 3 hidden nodes with 10 nodes each. For all the networks $\lambda$ was set to 1.0.

# Test Multi-Layer

As suggested on the instructions I created a network with 2 hidden layers of 50 neurons each. Then I trained it on 45000 samples and validate it on 5000. The parameters I used were the one stated in the instructions : $eta_min=1e-5 \ ,\ eta_max1e-1 \ ,\ n_s = 5\cdot 450 \ \lambda=0.0001$ (but I guess thre was a typo, $n_s$ was supposed to be $n_s = 2\cdot 450$).

The loss and the accuracy throughout the 2 cycles of training are shown in Figure \ref{fig:MLN}. The accuracy I got on the test data is $52.6\%$.

\begin{figure}[h]
\centering
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{./pictures/MLN_cost.pdf}
\caption{Cost Fuction.}
\label{fig:MLN_cost}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{./pictures/MLN_loss.pdf}
\caption{Loss Fuction.}
\label{fig:MLN_loss}
\end{subfigure}
~
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{./pictures/MLN_acc.pdf}
\caption{Accuracy of the network (\%).}
\label{fig:MLN_loss}
\end{subfigure}
\caption{Training evolution of cost loss and accuracy on test and validation sets for the 2 layer network, with parameters $eta_min=1e-5 \ ,\ eta_max1e-1 \ ,\ n_s = 5\cdot 450 \ \lambda=0.0001$, run for 2 cycles.}
\label{fig:MLN}
\end{figure}

Then I tested with the same hyperparameters the 9 layer network described by the sequence of hidden states $[50, 30, 20, 20, 10, 10, 10, 10]$. The performance I get is $43.27\%$ (even though various rounds show a lot of variability in the result ranging from 40\% to 45\%). For completeness I also plot in Figure \ref{fig:9layer} the cost, loss and accuracy.

\begin{figure}[h]
\centering
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{./pictures/cost_9l.pdf}
\caption{Cost Fuction.}
\label{fig:MLN_cost}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{./pictures/loss_9l.pdf}
\caption{Loss Fuction.}
\label{fig:MLN_loss}
\end{subfigure}
~
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{./pictures/accuracy_9l.pdf}
\caption{Accuracy of the network (\%).}
\label{fig:MLN_loss}
\end{subfigure}
\caption{Training evolution of cost loss and accuracy for the 9-layer network.}
\label{fig:9layer}
\end{figure}



# Batch Normalization

## Gradient Computation

I repeated the process described above for comparing the numerical and the computed gradients. What I found was that both the absolute value and the relative error was in the order of $1e-2$ for every parameter. To check the correctness I just tried to train the network and see the evolution of the loss function. What I found was that the loss was decreasing and that the performance of the network was fine.

I still double checked the code but I could not find any error. The only concern I have is that the sample mean and variance is not constant in practice (as assumed in the formulas), so this is what could cause such variations in the gradient.

## Testing the Network

Using the same parameters as for the multi-layer network tested before, I got the following results:


|  Network    |  Test Accuracy  |
|:-----------:|:---------------:|
|  3 layer    |   $53.37\%$     |
|  9 layer    |   $46.98\%$     |


For completeness I also reported in Figure \ref{fig:BN3l} and Figure \ref{fig:BN9l} the loss, cost, and accuracy for both the networks. As we can see batch normalization helps mantaining performance even for deep networks.


\begin{figure}[h]
\centering
\begin{subfigure}{.3\textwidth}
\includegraphics[width=1\linewidth]{./pictures/BN3l_cost.pdf}
\caption{Cost Fuction.}
\label{fig:BN3l_cost}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.3\textwidth}
\includegraphics[width=1\linewidth]{./pictures/BN3l_loss.pdf}
\caption{Loss Fuction.}
\label{fig:BN3l_loss}an accuracy of $51.81\%$
\end{subfigure}
\begin{subfigure}{.3\textwidth}
\includegraphics[width=1\linewidth]{./pictures/BN3l_acc.pdf}
\caption{Accuracy of the network (\%).}
\label{fig:BN3l_loss}
\end{subfigure}
\caption{Training evolution of cost loss and accuracy on test and validation sets for the 2 layer network with batch normalization.}
\label{fig:BN3l}
\end{figure}


\begin{figure}[h]
\centering
\begin{subfigure}{.3\textwidth}
\includegraphics[width=1\linewidth]{./pictures/BN9l_cost.pdf}
\caption{Cost Fuction.}
\label{fig:BN9l_cost}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.3\textwidth}
\includegraphics[width=1\linewidth]{./pictures/BN9l_loss.pdf}
\caption{Loss Fuction.}
\label{fig:BN9l_loss}
\end{subfigure}
\begin{subfigure}{.3\textwidth}
\includegraphics[width=1\linewidth]{./pictures/BN9l_acc.pdf}
\caption{Accuracy of the network (\%).}
\label{fig:BN9l_loss}
\end{subfigure}
\caption{Training evolution of cost loss and accuracy on test and validation sets for the 9 layer network with batch normalization.}
\label{fig:BN9l}
\end{figure}


## Search for $\lambda$

I searched for the best value of $\lambda$ to optimize the 3 layer network. I tested around 40 values between $1e-5$ and $1e-1$, and the search is depicted in Figure \ref{fig:lamd}.

\begin{figure}[h]
\centering
\includegraphics[width=0.8\linewidth]{./pictures/lambdasearch.pdf}
\caption{All the performances on the validation test for all the trials I made on the different values of $\lambda$.}
\label{fig:lamd}
\end{figure}



The best value for $\lambda$ is $0.004718$ which corresponds to an accuracy of $51.81\%$ on the test set, and an accuracy of $51.26\%$ on the validation set. The value is very close to the one given by the assignment instructions.
Of course this search does not account for variability in the results, which can lead to higher performance scores.


## Sensitivity to initialization

I thested the 3 layer networks with different standard deviation for the initializaiton parameter. The result for the different values of $\sigma$ are reported in the table below, and the loss plots are shown in Figure \ref{fig:sig}.


|  $\sigma$    |  Multi-Layer  |  Batch Normalized  |
|:-----------:|:---------------:|:---------------:|
|  $0.1$    |   $52.3\%$     |   $51.1\%$     |
|  $0.001$    |   $10.0\%$     |   $50.45\%$     |
|  $0.0001$    |   $10.0\%$     |   $10.0\%$     |


It is obvious how a very small value of $\sigma$ nullifies the network, by having a random performance on the test set. Also we can see the effect of the initializaion in the losses which, as $\sigma$ goes down, they get more spiky and more caotic, and enhancing the peakes in the plot caused by the cyclic learning rate.

\begin{figure}[h]
\centering
\begin{subfigure}{.4\textwidth}
\includegraphics[width=1\linewidth]{./pictures/s1ML.pdf}
\caption{Multi-Layer Network, $sig=1e-1$.}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.4\textwidth}
\includegraphics[width=1\linewidth]{./pictures/s1BN.pdf}
\caption{Batch Normalization, $sig = 1e-1$.}
\end{subfigure}
~
\begin{subfigure}{.4\textwidth}
\includegraphics[width=1\linewidth]{./pictures/s2ML.pdf}
\caption{Multi-Layer Network, $sig=1e-3$.}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.4\textwidth}
\includegraphics[width=1\linewidth]{./pictures/s2BN.pdf}
\caption{Batch Normalization, $sig = 1e-3$.}
\end{subfigure}
~
\begin{subfigure}{.4\textwidth}
\includegraphics[width=1\linewidth]{./pictures/s3ML.pdf}
\caption{Multi-Layer Network, $sig=1e-4$.}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.4\textwidth}
\includegraphics[width=1\linewidth]{./pictures/s3BN.pdf}
\caption{Batch Normalization, $sig = 1e-4$.}
\end{subfigure}

\caption{Loss plots for different values of $sig$ and for the two types of network.}
\label{fig:sig}
\end{figure}
