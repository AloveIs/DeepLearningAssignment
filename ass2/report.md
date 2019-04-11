---
author: Pietro Alovisi
title: Assignment 2 - Report
header-includes: \usepackage{subcaption} \usepackage{graphicx}
---


# Gradient Check

To check the correctness of the algorithm I computed the maximum of the absolute value of the difference of the weight matrices and the bias vector:

$$
max(|W_1-W_{1\ num}|)
$$
$$
max(|W_2-W_{2\ num}|)
$$
$$
max(|b_1-b_{1\ num}|)
$$
$$
max(|b_2-b_{2\ num}|)
$$

Where $\cdot_{num}$ represents the numerical computed value. All these values were smaller than $1e-6$ for different initialization and for different choiches of $\lambda$. The numerical step used was $h=1e-5$.


# Network Performance

First I tested the network using the parameters in the Assignment instruction. My results to replicate Figure 3 and 4 in the instructions are respectively represented in Figure \ref{fig:3} and \ref{fig:4}


\begin{figure}[h]
\centering
\begin{subfigure}{.4\textwidth}
  \includegraphics[width=1\linewidth]{fig3_2s.pdf}
  \caption{Loss function.}
\end{subfigure}%
\begin{subfigure}{.4\textwidth}
  \includegraphics[width=1\linewidth]{fig3_1s.pdf}
  \caption{Cost function.}
\end{subfigure}
~
\begin{subfigure}{.4\textwidth}
  \includegraphics[width=1\linewidth]{fig3_3s.pdf}
  \caption{Performance function.}
\end{subfigure}
\caption{Representation of the cost, loss and performance of the network during training usign as parameters $n_s=500$, $\lambda=0.01$ for 1 cycle.}
\label{fig:3}
\end{figure}

\begin{figure}[h]
  \centering
\begin{subfigure}{.4\textwidth}
  \includegraphics[width=1\linewidth]{fig4_2s.pdf}
  \caption{Loss function.}
\end{subfigure}%
\begin{subfigure}{.4\textwidth}
  \includegraphics[width=1\linewidth]{fig4_1s.pdf}

  \caption{Cost function.}
\end{subfigure}
~
\begin{subfigure}{.4\textwidth}
  \includegraphics[width=1\linewidth]{fig4_3s.pdf}
  \caption{Performance function.}
\end{subfigure}
\caption{Representation of the cost, loss and performance of the network during training usign as parameters $n_s=800$, $\lambda=0.01$ for 1 cycle.}
\label{fig:4}
\end{figure}



# $\lambda$ search

To find the best fitting value for the hyperparameter $\lambda$ I first fixed the other hyper parametrs as follows: $eta_min=1e-5 \ ,\ eta_max1e-1 \ ,\ n_s = 900$ and run for each lambda for 2 cycles.

I have done a preliminary search for $\lambda$ in the range $(10^{-8},10^{-1})$ for a total of 100 samples. The result is shown in Figure \ref{fig:lambda1}.

Then with a second, more focused, search in the interval $(10^{-4},10^{-2})$ etching 60 samples. I obtained the results shown in Figure \ref{fig:lambda2}.

\begin{figure}[h]
\centering
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{lambda_coarse.pdf}
\caption{Performance on the vaidation set for the coarse search of $\lambda$.}
\label{fig:lambda1}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{lambda_fine.pdf}
\caption{Performance on the validation set for the finer search of $\lambda$.}
\label{fig:lambda2}
\end{subfigure}
\caption{Result of search for the best hyperparameter $\lambda$.}
\end{figure}

In the table below are shown the results on the validation set for the best 3 values of $\lambda$.

| $\lambda$ | Performance on Validation    |
|:---------:|:----------------------------:|
| 0.000305227 |51.78\%                     |
| 0.000141724 |51.70\%                     |
| 0.000928504 |51.70\%                     |


# Full data with best $\lambda$

Now I can apply the best found $\lambda$ to the whole dataset but 1000 samples for validation, and test the final result on the test data.
The result is shown in the following table, and the loss and accuracy are shown in Figure \ref{fig:best}.

| $\lambda$ | Performance on Test data    |
|:---------:|:----------------------------:|
| 0.000305227 |51.76\%                     |



\begin{figure}[h]
\centering
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{best_l_cost.pdf}
\caption{Cost function.}
\end{subfigure}\hspace{5mm}%
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{best_l_loss.pdf}
\caption{Loss function.}
\end{subfigure}\hspace{5mm}
~
\begin{subfigure}{.45\textwidth}
\includegraphics[width=1\linewidth]{best_l_accuracy.pdf}
\caption{Accuracy function.}
\end{subfigure}
\caption{Tracked quantities during training with the best value of $\lambda$.}
\label{fig:best}
\end{figure}
