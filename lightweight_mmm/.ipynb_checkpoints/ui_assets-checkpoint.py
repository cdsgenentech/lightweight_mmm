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



def compute_impactable(
    original_data: pd.DataFrame,
    cost_cols: List[str],
    media_contribution_df: pd.DataFrame,
    start_date: int,
    end_date: int
) -> Tuple[pd.DataFrame, float, float]:
    
  """
  Computes impactable metrics for a specified time period across all channels.

    This function aggregates sales and cost data within a specified date range,
    merges it with media contribution data, and calculates return on investment (ROI)
    for each channel. Additionally, it computes the total contribution of all channels
    relative to sales and the overall ROI.

    Args:
        original_data : original feature table
        
        cost_cols : list of cost columns
        
        media_contribution_df : media contribution dataframe obtained from
        plot.create_media_baseline_contribution_df
        
        start_date : int or str
        The start date of the period for which to compute the impact, in YYYYMM format.
        
        end_date : int or str
        The end date of the period for which to compute the impact, in YYYYMM format.

   Returns:
        contrib_df : pd.DataFrame
        A DataFrame containing the total contributions, cost and ROI for each channel
        within the specified date range.
        
        total_contribution : float
        The total contribution of all channels relative to the sales within the date range.
        
        total_roi : float
        The overall return on investment (ROI) across all channels within the date range.

  """
  original_data = original_data.groupby('date', as_index=False).sum()
  channel_list = [x.replace('_cost', '') for x in cost_cols]
  start_date = pd.to_datetime(str(start_date), format='%Y%m')
  end_date = pd.to_datetime(str(end_date), format='%Y%m')
  original_data = original_data[['date', 'sale']+cost_cols]
  if pd.api.types.is_integer_dtype(original_data['date']) or pd.api.types.is_string_dtype(original_data['date']):
    original_data['date'] = pd.to_datetime(original_data['date'].astype(str), format='%Y%m')
  media_contribution_df = pd.merge(original_data, media_contribution_df, left_index=True, right_index=True)
  impactable_data = media_contribution_df[(media_contribution_df['date'] >= start_date)&(media_contribution_df['date'] <= end_date)]
  impactable_data = impactable_data.drop('date', axis=1)
  contrib_df = impactable_data.sum()
  contrib_df = pd.DataFrame(contrib_df).T
  for channel in channel_list:
    contrib_df[f'{channel} roi'] = contrib_df[f'{channel} contribution']/contrib_df[f'{channel}_cost']
  contrib_total = contrib_df[[x for x in list(impactable_data) if x.find('contribution') != -1 and x.find('baseline') == -1]]
  contrib_total = contrib_total.sum().sum()
  total_contribution = contrib_total/impactable_data['sale'].sum()
  total_roi = contrib_total/impactable_data[cost_cols].sum().sum()
  return contrib_df, total_contribution, total_roi


def _calculate_media_resp_curves(
    media_mix_model: lightweight_mmm.LightweightMMM, 
    multiplyer: float,
    extra_features: jnp.ndarray,
    roi_period: int = 12
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    
  """
    Computes the media contribution curves for each sample, time, and channel.

    This helper function estimates the contribution of media channels over a specified 
    ROI period by adjusting the media data with a given multiplier. It performs predictions 
    using the media mix model, taking into account additional features, and calculates 
    the contribution of each media channel across different samples and time points.

    The function is intended to be used within the `create_response_contribution_df` function.

    Args:
        media_mix_model : lightweight_mmm.LightweightMMM
    
        multiplyer : float
        A multiplier applied to the media data for the specified ROI period. This 
        array adjusts the media data to simulate different spending scenarios.
    
        extra_features : jnp.ndarray
        Additional features used by the media mix model during prediction. These 
        features may include variables like seasonality, holidays, or other external factors.
    
        roi_period : int, optional
        The number of time periods (e.g., months, weeks) for which to compute the 
        media contributions. The default is 12.

    Returns:
        data : jnp.ndarray
        The adjusted media data after applying the multiplier for the specified ROI period.
    
        media_contribution : jnp.ndarray
        The estimated contribution of each media channel for each sample, time, and 
        channel combination. If the data includes multiple geographic regions (geos), 
        contributions are aggregated across geos.
    
        predict_media : Dict[str, jnp.ndarray]
        A dictionary containing the predicted media transformations and other outputs 
        from the model's `_predict` method. This includes the transformed media data 
        and other prediction-related metrics.

    Raises:
        NotFittedModelError
        If the media mix model has not been fitted with data before attempting to compute 
        the media response curves.
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



def create_response_contribution_df(
    media_mix_model: lightweight_mmm.LightweightMMM,
    prices: jnp.ndarray,
    multiplyer: float,
    media_scaler: Optional[preprocessing.CustomScaler] = None,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    channel_names: Optional[Sequence[str]] = None,
    roi_period: int = 12) -> pd.DataFrame:
    
  """
    Creates a DataFrame summarizing the contribution of media channels and baseline over a specified ROI period
    for different level of spend for each channel.

    Args:
        media_mix_model : lightweight_mmm.LightweightMMM

        prices : jnp.ndarray
        A JAX array containing the prices for each media channel. This is used to calculate the spend.

        multiplyer : float
        A multiplier applied to the media data to simulate different scenarios, such as increased or decreased media spend.

        media_scaler : Optional[preprocessing.CustomScaler], optional
        An optional scaler used to inverse transform the media data. If not provided, no scaling is applied.

        target_scaler : Optional[preprocessing.CustomScaler], optional
        An optional scaler used to inverse transform the target predictions. If not provided, no scaling is applied.

        channel_names : Optional[Sequence[str]], optional
        A list of names corresponding to the media channels. If not provided, the names will be taken from the media mix model.

        roi_period : int, optional
        The number of periods (months) for which to calculate the media contributions. The default is 12.

    Returns:
        pd.DataFrame
        A DataFrame containing the contribution percentages, volumes, and spends for each media channel and baseline. 
        The DataFrame is summarized over the specified ROI period, with each row representing the total contributions 
        and spends for that period.

    Raises:
        NotFittedModelError
        If the media mix model has not been fitted with data before calling this function.

  """
  # Create media contribution matrix.
  data, scaled_media_contribution, predicted_media = _calculate_media_resp_curves(media_mix_model, multiplyer, None)

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

def generate_response_curves(
    media_mix_model: lightweight_mmm.LightweightMMM,
    multiplyer: float,
    prices: jnp.ndarray,
    media_scaler: Optional[preprocessing.CustomScaler] = None,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    channel_names: Optional[Sequence[str]] = None,
    roi_period: int = 12
) -> pd.DataFrame:
    
  """
    Generates response curves by varying media spend and calculating contributions for each multiplier.

    This function creates a DataFrame summarizing the contributions of media channels and baseline across different
    levels of media spend. It repeatedly calls `create_response_contribution_df` with increasing multipliers to simulate
    different spend scenarios. The resulting DataFrame can be used to analyze how changes in media spend affect 
    channel contributions and ROI.

    Args:
        media_mix_model : lightweight_mmm.LightweightMMM

        multiplyer : float
        The increment by which media spend is varied to generate different scenarios. Determines the step size in the response curves.

        prices : jnp.ndarray
        A JAX array containing the prices for each media channel. This is used to calculate the spend.

        media_scaler : Optional[preprocessing.CustomScaler], optional
        An optional scaler used to inverse transform the media data. If not provided, no scaling is applied.

        target_scaler : Optional[preprocessing.CustomScaler], optional
        An optional scaler used to inverse transform the target predictions. If not provided, no scaling is applied.

        channel_names : Optional[Sequence[str]], optional
        A list of names corresponding to the media channels. If not provided, the names will be taken from the media mix model.

        roi_period : int, optional
        The number of periods (months) for which to calculate the media contributions. The default is 12.

    Returns:
        pd.DataFrame
        A DataFrame containing the contribution percentages, volumes, and spends for each media channel and baseline 
        across different levels of media spend. Each row represents the total contributions and spends for a specific multiplier value.
  """

  resp_df = pd.DataFrame()
  num_points = int(1//(multiplyer)+3)
  for i in range(0,num_points):
    multi = i*multiplyer
    combined_df = create_response_contribution_df(media_mix_model, jnp.array(prices), multi, media_scaler, target_scaler, channel_names) 
    resp_df = pd.concat([resp_df, combined_df], ignore_index=True)
  return resp_df


def compute_mroi(
    media_mix_model: lightweight_mmm.LightweightMMM,
    eps_: float,
    prices: jnp.ndarray,
    media_scaler: Optional[preprocessing.CustomScaler] = None,
    target_scaler: Optional[preprocessing.CustomScaler] = None,
    channel_names: Optional[Sequence[str]] = None,
    roi_period: int = 12
) -> pd.DataFrame:
    
  """
    Computes the marginal return on investment (MROI) for each media channel.

    This function calculates the MROI by comparing the media contributions and spend at two different levels of 
    media investment: slightly lower, and slightly higher. The MROI is computed as the change in contribution 
    divided by the change in spend for each channel. The function returns a DataFrame containing the MROI for each 
    channel, which is useful for understanding the efficiency of additional investments in each media channel.

    Args:
        media_mix_model : lightweight_mmm.LightweightMMM
        A fitted media mix model used to estimate media contributions over the ROI period.

        eps_ : float
        A small percentage change in media spend used to compute the MROI. It determines the sensitivity of the calculation.

        prices : jnp.ndarray
        A JAX array containing the prices for each media channel. This is used to calculate the spend.

        media_scaler : Optional[preprocessing.CustomScaler], optional
        An optional scaler used to inverse transform the media data. If not provided, no scaling is applied.

        target_scaler : Optional[preprocessing.CustomScaler], optional
        An optional scaler used to inverse transform the target predictions. If not provided, no scaling is applied.

        channel_names : Optional[Sequence[str]], optional
        A list of names corresponding to the media channels. If not provided, the names will be taken from the media mix model.

        roi_period : int, optional
        The number of periods (e.g., weeks, months) for which to calculate the media contributions. The default is 12.

    Returns:
        pd.DataFrame
        A DataFrame containing the MROI for each media channel. The DataFrame includes the channel name and its 
        corresponding MROI value.
  """
    
  resp_df = pd.DataFrame()
  for i in [-1, 0, 1]:
    multi = 1 + i * eps_
    combined_df = create_response_contribution_df(media_mix_model, jnp.array(prices), multi, media_scaler, target_scaler, channel_names) 
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

