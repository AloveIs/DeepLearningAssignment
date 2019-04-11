---
author: Pietro Alovisi
title: Assignment 2 - Report
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

The loss and the accuracy throughout the 2 cycles of training are shown in Figure \ref{fig:MLN}. The accuracy I got on the test data is $50.17\%$.

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

Then I tested with the same hyperparameters the 9 layer network described by the sequence of hidden states $[50, 30, 20, 20, 10, 10, 10, 10]$. The performance I get is $48.75\%$. For completeness I also polot again in Figure \ref{fig:9layer} the cost, loss and accuracy.

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
