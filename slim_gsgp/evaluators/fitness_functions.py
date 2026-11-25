# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module provides various error metrics functions for evaluating machine learning models.
"""

import torch


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        RMSE value.
    """
    return torch.sqrt(torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1))


def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MSE value.
    """
    return torch.mean(torch.square(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MAE value.
    """
    return torch.mean(torch.abs(torch.sub(y_true, y_pred)), dim=len(y_pred.shape) - 1)


def mae_int(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Absolute Error (MAE) for integer values.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        MAE value for integer predictions.
    """
    return torch.mean(torch.abs(torch.sub(y_true, torch.round(y_pred))), dim=len(y_pred.shape) - 1)


def signed_errors(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute signed errors between true and predicted values.

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        Signed error values.
    """
    return torch.sub(y_true, y_pred)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute R-squared (R²) score.

    If using this fitness function, please ensure that you are maximizing the fitness value when

    Parameters
    ----------
    y_true : torch.Tensor
        True values.
    y_pred : torch.Tensor
        Predicted values.

    Returns
    -------
    torch.Tensor
        R² score value.
    """
    ss_res = torch.sum(torch.square(y_true - y_pred))
    ss_tot = torch.sum(torch.square(y_true - torch.mean(y_true)))
    r2 = 1 - (ss_res / ss_tot)
    return r2

"""
---
Classification fitness functions (multiclass SLIM-GSGP)
y_true: integer labels (n,) in 0..K-1 ; y_pred: logit matrix (n, K). All are minimised.
---
"""

def cross_entropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean multiclass cross-entropy (softmax + negative log-likelihood) computed from logits.

    Parameters
    ----------
    y_true : torch.Tensor
        Integer class labels, shape (n,).
    y_pred : torch.Tensor
        Logits, shape (n, K).

    Returns
    -------
    torch.Tensor
        Cross-entropy value (to minimise).
    """
    return torch.nn.functional.cross_entropy(y_pred, y_true.long())


def macro_f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    1 - macro-F1 of the argmax predictions. Classes are those present in y_true or in the
    predictions (same convention as sklearn's f1_score(average="macro", zero_division=0)).

    Parameters
    ----------
    y_true : torch.Tensor
        Integer class labels, shape (n,).
    y_pred : torch.Tensor
        Logits, shape (n, K).

    Returns
    -------
    torch.Tensor
        1 - macro-F1 (to minimise).
    """
    y_true = y_true.long()
    pred = torch.argmax(y_pred, dim=1)
    f1s = []
    for k in torch.unique(torch.cat([y_true, pred])):
        tp = torch.sum((pred == k) & (y_true == k)).float()
        fp = torch.sum((pred == k) & (y_true != k)).float()
        fn = torch.sum((pred != k) & (y_true == k)).float()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else torch.tensor(0.0))
    return 1 - torch.stack(f1s).mean()


def error_rate(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    1 - accuracy of the argmax predictions.

    Parameters
    ----------
    y_true : torch.Tensor
        Integer class labels, shape (n,).
    y_pred : torch.Tensor
        Logits, shape (n, K).

    Returns
    -------
    torch.Tensor
        Error rate (to minimise).
    """
    return 1 - torch.mean((torch.argmax(y_pred, dim=1) == y_true.long()).float())
