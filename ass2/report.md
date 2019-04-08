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
\begin{subfigure}{.4\textwidth}
  \centering
  \includegraphics[width=1\linewidth]{fig3_2s.pdf}
  \caption{Loss function.}
\end{subfigure}%
\begin{subfigure}{.4\textwidth}
  \centering
  \includegraphics[width=1\linewidth]{fig3_1s.pdf}
  \caption{Cost function.}
\end{subfigure}
\begin{subfigure}{.4\textwidth}
  \centering
  \includegraphics[width=1\linewidth]{fig3_3s.pdf}
  \caption{Performance function.}
\end{subfigure}
\caption{Representation of the cost, loss and performance of the network during training usign as parameters $n_s=500$, $\lambda=0.01$ for 1 cycle.}
\label{fig:3}
\end{figure}

\begin{figure}[h]
\begin{subfigure}{.4\textwidth}
  \centering
  \includegraphics[width=1\linewidth]{fig4_2s.pdf}
  \caption{Loss function.}
\end{subfigure}%
\begin{subfigure}{.4\textwidth}
  \centering
  \includegraphics[width=1\linewidth]{fig4_1s.pdf}
  \caption{Cost function.}
\end{subfigure}
\begin{subfigure}{.4\textwidth}
  \centering
  \includegraphics[width=1\linewidth]{fig4_3s.pdf}
  \caption{Performance function.}
\end{subfigure}
\caption{Representation of the cost, loss and performance of the network during training usign as parameters $n_s=800$, $\lambda=0.01$ for 1 cycle.}
\label{fig:4}
\end{figure}



# $\lambda$ search

I have done a preliminary search for $\lambda$ in the range $(10^{-5},10^{1})$ for a total of 30 samples. The result is shown in Figure \ref{fig:lambda1}.

Then with a second, more focused, search in the interval $(10^{-5},10^{1})$, I obtained the results shown in Figure \ref{fig:lambda2}.


\begin{figure}[h]
  \centering
  \includegraphics[width=1\linewidth]{lambda_coarse.pdf}
  \caption{Results for different lambdas.}
  \label{fig:lambda1}
\end{figure}

\begin{figure}[h]
  \centering
  \includegraphics[width=1\linewidth]{fig4_3s.pdf}
  \caption{Results for different lambdas.}
  \label{fig:lambda2}
\end{figure}


# Full data with best $\lambda$
