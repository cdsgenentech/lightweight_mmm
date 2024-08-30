# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plotting functions pre and post model fitting."""

import functools
import logging

# Using these types from typing instead of their generic types in the type hints
# in order to be compatible with Python 3.7 and 3.8.
from typing import Any, List, Optional, Sequence, Tuple

import arviz
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import seaborn as sns
from sklearn import metrics

from lightweight_mmm import lightweight_mmm
from lightweight_mmm import models
from lightweight_mmm import preprocessing
from lightweight_mmm import utils

plt.style.use("default")

_PALETTE = sns.color_palette(n_colors=100)


def _calculate_media_resp_curves_for_UI(
    media_mix_model: lightweight_mmm.LightweightMMM, 
    multiplyer,
    extra_features,
    roi_period=12) -> jnp.ndarray:
  """Computes contribution for each sample, time, channel.

  Serves as a helper function for making predictions for each channel, time
  and estimate sample. It is meant to be used in creating media baseline
  contribution dataframe and visualize media attribution over spend proportion
  plot.

  Args:
    media_mix_model: Media mix model.

  Returns:
    Estimation of contribution for each sample, time, channel.

  Raises:
    NotFittedModelError: if the model is not fitted before computation
  """
  if not hasattr(media_mix_model, "trace"):
    raise lightweight_mmm.NotFittedModelError(
        "Model needs to be fit first before attempting to plot its fit.")

  if media_mix_model.trace["media_transformed"].ndim > 3:
    # s for samples, t for time, c for media channels, g for geo
    einsum_str = "stcg, scg->stcg"
  elif media_mix_model.trace["media_transformed"].ndim == 3:
    # s for samples, t for time, c for media channels
    einsum_str = "stc, sc->stc"
  media = media_mix_model.media
  data = media.at[-roi_period:].set(media[-roi_period:] * multiplyer)
  predict_media = media_mix_model._predict(rng_key=jax.random.PRNGKey(seed=101), media_data=data, 
                         extra_features=extra_features,
                        media_prior=jnp.array(media_mix_model._media_prior),
                        degrees_seasonality=media_mix_model._degrees_seasonality,
                        frequency=media_mix_model._seasonality_frequency,
                        weekday_seasonality=media_mix_model._weekday_seasonality,
                        transform_function=media_mix_model._model_transform_function,
                        model=media_mix_model._model_function,
                        custom_priors=media_mix_model.custom_priors,
                        posterior_samples=media_mix_model.trace)

  media_contribution = jnp.einsum(einsum_str,
                                  predict_media["media_transformed"],
                                  media_mix_model.trace["coef_media"])
  if predict_media["media_transformed"].ndim > 3:
    # Aggregate media channel contribution across geos.
    media_contribution = media_contribution.sum(axis=-1)
  return data, media_contribution, predict_media



def create_response_contribution_df_for_UI(
    media_mix_model: lightweight_mmm.LightweightMMM,
    prices,
    multiplyer,
    media_scaler:Optional[preprocessing.CustomScaler] = None,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    channel_names: Optional[Sequence[str]] = None,
    roi_period=12) -> pd.DataFrame:
  """Creates a dataframe for weekly media channels & basline contribution.

  The output dataframe will be used to create a stacked area plot to visualize
  the contribution of each media channels & baseline.

  Args:
    media_mix_model: Media mix model.
    target_scaler: Scaler used for scaling the target.
    channel_names: Names of media channels.

  Returns:
    contribution_df: DataFrame of weekly channels & baseline contribution
    percentage & volume.
  """
  # Create media contribution matrix.
  data, scaled_media_contribution, predicted_media = _calculate_media_resp_curves_for_UI(media_mix_model, multiplyer, None)

  # Aggregate media channel contribution across samples.
  sum_scaled_media_contribution_across_samples = scaled_media_contribution.sum(
      axis=0)
  # Aggregate media channel contribution across channels.
  sum_scaled_media_contribution_across_channels = scaled_media_contribution.sum(
      axis=2)

  # Calculate the baseline contribution.
  # Scaled prediction - sum of scaled contribution across channels.
  scaled_prediction = predicted_media["mu"]
  if media_mix_model.trace["media_transformed"].ndim > 3:
    # Sum up the scaled prediction across all the geos.
    scaled_prediction = scaled_prediction.sum(axis=-1)
  baseline_contribution = scaled_prediction - sum_scaled_media_contribution_across_channels

  # Sum up the scaled media, baseline contribution and predictio across samples.
  sum_scaled_media_contribution_across_channels_samples = sum_scaled_media_contribution_across_channels.sum(
      axis=0)
  sum_scaled_baseline_contribution_across_samples = baseline_contribution.sum(
      axis=0)

  # Adjust baseline contribution and prediction when there's any negative value.
  adjusted_sum_scaled_baseline_contribution_across_samples = np.where(
      sum_scaled_baseline_contribution_across_samples < 0, 0,
      sum_scaled_baseline_contribution_across_samples)
  adjusted_sum_scaled_prediction_across_samples = adjusted_sum_scaled_baseline_contribution_across_samples + sum_scaled_media_contribution_across_channels_samples

  # Calculate the media and baseline pct.
  # Media/baseline contribution across samples/total prediction across samples.
  media_contribution_pct_by_channel = (
      sum_scaled_media_contribution_across_samples /
      adjusted_sum_scaled_prediction_across_samples.reshape(-1, 1))
  # Adjust media pct contribution if the value is nan
  media_contribution_pct_by_channel = np.nan_to_num(
      media_contribution_pct_by_channel)

  baseline_contribution_pct = adjusted_sum_scaled_baseline_contribution_across_samples / adjusted_sum_scaled_prediction_across_samples
  # Adjust baseline pct contribution if the value is nan
  baseline_contribution_pct = np.nan_to_num(
      baseline_contribution_pct)

  # If the channel_names is none, then create naming covention for the channels.
  if channel_names is None:
    channel_names = media_mix_model.media_names

  # Create media/baseline contribution pct as dataframes.
  media_contribution_pct_by_channel_df = pd.DataFrame(
      media_contribution_pct_by_channel, columns=channel_names)
  baseline_contribution_pct_df = pd.DataFrame(
      baseline_contribution_pct, columns=["baseline"])
  contribution_pct_df = pd.merge(
      media_contribution_pct_by_channel_df,
      baseline_contribution_pct_df,
      left_index=True,
      right_index=True)

  # If there's target scaler then inverse transform the posterior prediction.
  posterior_pred = predicted_media["mu"]
  if target_scaler:
    posterior_pred = target_scaler.inverse_transform(posterior_pred)

  # Take the sum of posterior predictions across geos.
  if media_mix_model.trace["media_transformed"].ndim > 3:
    posterior_pred = posterior_pred.sum(axis=-1)

  # Take the average of the inverse transformed prediction across samples.
  posterior_pred_df = pd.DataFrame(
      posterior_pred.mean(axis=0), columns=["avg_prediction"])

  # Adjust prediction value when prediction is less than 0.
  posterior_pred_df["avg_prediction"] = np.where(
      posterior_pred_df["avg_prediction"] < 0, 0,
      posterior_pred_df["avg_prediction"])

  contribution_pct_df.columns = [
      "{}_percentage".format(col) for col in contribution_pct_df.columns
  ]
  contribution_df = pd.merge(
      contribution_pct_df, posterior_pred_df, left_index=True, right_index=True)

  # Create contribution by multiplying average prediction by media/baseline pct.
  for channel in channel_names:
    channel_contribution_col_name = "{} contribution".format(channel)
    channel_pct_col = "{}_percentage".format(channel)
    contribution_df.loc[:, channel_contribution_col_name] = contribution_df[
        channel_pct_col] * contribution_df["avg_prediction"]
    contribution_df.loc[:, channel_contribution_col_name] = contribution_df[
        channel_contribution_col_name].astype("float")
  contribution_df.loc[:, "baseline contribution"] = contribution_df[
      "baseline_percentage"] * contribution_df["avg_prediction"]
  
  contrib_cols = [x for x in list(contribution_df) if 'contribution' in x and 'baseline' not in x]
  contribution_df = contribution_df.loc[:, contrib_cols]

  media = data
  media = media_scaler.inverse_transform(media)
  media = media.sum(axis=(2))
  if prices is not None:
    media = media*prices
  media_array = np.array(media)
  spend_df = pd.DataFrame(media_array)
  spend_df.columns =  [x + ' spend' for x in channel_names]
  combined_df = pd.concat([spend_df, contribution_df], axis=1)
  combined_df = combined_df.tail(roi_period)
  combined_df = pd.DataFrame(combined_df.sum()).T
  combined_df.loc[:, "multiplyer"] = multiplyer
  return combined_df

def generate_response_curves_for_UI(media_mix_model, multiplyer, prices, media_scaler, target_scaler, channel_names, roi_period=12):
  resp_df = pd.DataFrame()
  num_points = int(1//(multiplyer)+3)
  for i in range(0,num_points):
    multi = i*multiplyer
    combined_df = create_response_contribution_df_for_UI(media_mix_model, jnp.array(prices), multi, media_scaler, target_scaler, channel_names) 
    resp_df = pd.concat([resp_df, combined_df], ignore_index=True)
  return resp_df


def compute_mroi_for_UI(media_mix_model, eps_, prices, media_scaler, target_scaler, channel_names, roi_period=12):
  resp_df = pd.DataFrame()
  for i in [-1, 0, 1]:
    multi = 1 + i*multiplyer
    combined_df = create_response_contribution_df_for_UI(media_mix_model, jnp.array(prices), multi, media_scaler, target_scaler, channel_names) 
    resp_df = pd.concat([resp_df, combined_df], ignore_index=True)
  df_lower = resp_df[resp_df.multiplyer== 1-eps_]
  df_actual = resp_df[resp_df.multiplyer== 1]
  df_upper = resp_df[resp_df.multiplyer== 1+eps_]
  channel_list = []
  mroi_list = []
  roi_list = []
  for channel in channel_names:
    mroi = (df_upper[f'{channel} contribution'].values[0] - df_lower[f'{channel} contribution'].values[0])/(df_upper[f'{channel} spend'].values[0] - df_lower[f'{channel} spend'].values[0])
    roi = df_actual[f'{channel} contribution'].values[0] / df_actual[f'{channel} spend'].values[0]
    channel_list.append(channel)
    mroi_list.append(round(mroi,2))
    roi_list.append(roi)

  mroi_df = pd.DataFrame({'channel':channel_list, 'mroi':mroi_list})
  return mroi_df

