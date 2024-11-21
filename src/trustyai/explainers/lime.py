"""Explainers.lime module"""
# pylint: disable = import-error, too-few-public-methods, wrong-import-order, line-too-long,
# pylint: disable = unused-argument, duplicate-code, consider-using-f-string, invalid-name
from typing import Dict, Union

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from trustyai import _default_initializer  # pylint: disable=unused-import
from trustyai.utils._visualisation import DEFAULT_STYLE as ds
from trustyai.utils.data_conversions import (
    OneInputUnionType,
    data_conversion_docstring,
    OneOutputUnionType,
)

from .explanation_results import SaliencyResults
from trustyai.model import simple_prediction, Model

from org.kie.trustyai.explainability.local.lime import (
    LimeConfig as _LimeConfig,
    LimeExplainer as _LimeExplainer,
)
from org.kie.trustyai.explainability.model import (
    EncodingParams,
    PredictionProvider,
    Saliency,
    PerturbationContext,
    PredictionInputsDataDistribution,
)

from java.util import Random


LimeConfig = _LimeConfig


class LimeResults(SaliencyResults):
    """Wraps LIME results. This object is returned by the :class:`~LimeExplainer`,
    and provides a variety of methods to visualize and interact with the explanation.
    """

    def __init__(self, saliencyResults: SaliencyResults):
        """Constructor method. This is called internally, and shouldn't ever need to be used
        manually."""
        self._java_saliency_results = saliencyResults

    def saliency_map(self) -> Dict[str, Saliency]:
        """
        Return a dictionary of found saliencies.

        Returns
        -------
        Dict[str, Saliency]
             A dictionary of :class:`~trustyai.model.Saliency` objects, keyed by output name.
        """
        return {
            entry.getKey(): entry.getValue()
            for entry in self._java_saliency_results.saliencies.entrySet()
        }

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return the LIME result as a dataframe.

        Returns
        -------
        pandas.DataFrame
            Dictionary of DataFrames, keyed by output name, containing the results of the LIME
            explanation. For each model output, the table will contain the following columns:

            * ``Feature``: The name of the feature
            * ``Value``: The value of the feature for this particular input.
            * ``Saliency``: The importance of this feature to the output.
            * ``Confidence``: The confidence of this explanation as returned by the explainer.

        """
        outputs = self.saliency_map().keys()

        data = {}
        for output in outputs:
            output_rows = []
            for pfi in self.saliency_map().get(output).getPerFeatureImportance():
                output_rows.append(
                    {
                        "Feature": str(pfi.getFeature().getName().toString()),
                        "Value": pfi.getFeature().getValue().getUnderlyingObject(),
                        "Saliency": pfi.getScore(),
                        "Confidence": pfi.getConfidence(),
                    }
                )
            data[output] = pd.DataFrame(output_rows)
        return data

    def as_html(self) -> pd.io.formats.style.Styler:
        """
        Return the LIME results as Pandas Styler objects.

        Returns
        -------
        Dict[str, pandas.Styler]
            Dictionary of stylers keyed by output name. Each styler containing the results of the
            LIME explanation for that particular output, in the same
            schema as in :func:`as_dataframe`. This will:

            * Color each ``Saliency`` based on how their magnitude.
        """

        htmls = {}
        for k, df in self.as_dataframe().items():
            htmls[k] = df.style.background_gradient(
                LinearSegmentedColormap.from_list(
                    name="rwg",
                    colors=[
                        ds["negative_primary_colour"],
                        ds["neutral_primary_colour"],
                        ds["positive_primary_colour"],
                    ],
                ),
                subset="Saliency",
                vmin=-1 * max(np.abs(df["Saliency"])),
                vmax=max(np.abs(df["Saliency"])),
            )
        return htmls


class LimeExplainer:
    """*"Which features were most important to the results?"*

    LIME (`Local Interpretable Model-agnostic Explanations <https://arxiv.org/abs/1602.04938>`_)
    seeks to answer this question via providing *saliencies*, weights associated with each input
    feature that describe how strongly said feature contributed to the model's output.
    """

    def __init__(self, **kwargs):
        r"""Initialize the :class:`LimeExplainer`.

        Parameters
        ----------
        Keyword Arguments:
            * penalise_sparse_balance : bool
                (default= ``True``) Whether to penalise features that are likely to produce linearly
                inseparable outputs. This can improve the efficacy and interpretability of the
                outputted saliencies.
            * normalise_weights : bool
                (default= ``False``) Whether to normalise the saliencies generated by LIME. If selected,
                saliencies will be normalized between 0 and 1.
            * use_wlr_model : bool
                (default= ``True``) Whether to use a weighted linear regression as the LIME explanatory
                model. If `false`, a multilayer perceptron is used, which generally has a slower
                runtime,
            * seed: int
                (default= ``0``) The random seed to be used.
            * perturbations: int
                (default= ``1``) The starting number of feature perturbations within the explanation
                process.
            * trackCounterfactuals : bool
                (default= ``False``) Keep track of produced byproduct counterfactuals during LIME run.
            * samples: int
                (default= ``300``) Number of samples to be generated for the local linear model training.
            * encoding_params: Union[list, tuple]
                (default= ``(0.07, 0.3)``) Lime encoding parameters, as a tuple/list of two float numbers:
                - encoding_params[0] is the width of the Gaussian filter for clustering number features.
                - encoding_params[1] is the threshold for clustering number features.
            * data_distribution: PredictionInputsDataDistribution
                (default= ``PredictionInputsDataDistribution([])``) Data distribution used to find better feature perturbations
            * features: int
                (default= ``6``) Number of feature to select from the original set of input features
            * retries: int
                (default= ``3``) Number of retries performed by LIME to find a separable dataset
            * dataset_minimum: int
                (default= ``10``) Minimum number of samples retained by the proximity filter to be acceptable
            * separable_dataset_ratio: float
                (default= ``0.1``) Minimum portion of the encoded dataset that needs to have a different label
            *  kernel_width: float
                (default= ``0.5``) Width of the proximity kernel
            * proximity_threshold: float
                (default= ``0.83``) Proximity threshold used to retain close samples
            * adapt_dataset_variance: bool
                (default= ``True``) Whether LIME should try to increase the perturbation variance in subsequent retries
            * feature_selection: bool
                (default= ``True``) Whether LIME should generate saliency for to the most important features only
            * filter_interpretable: bool
                (default= ``False``) Whether the proximity filter should happen in the interpretable space

        """
        self._jrandom = Random()
        self._jrandom.setSeed(kwargs.get("seed", 0))
        ep = kwargs.get("encoding_params", (0.07, 0.3))
        self._lime_config = (
            LimeConfig()
            .withNormalizeWeights(kwargs.get("normalise_weights", False))
            .withPerturbationContext(
                PerturbationContext(self._jrandom, kwargs.get("perturbations", 1), kwargs.get("std_dev", 0.01))
            )
            .withSamples(kwargs.get("samples", 300))
            .withDataDistribution(
                kwargs.get("data_distribution", PredictionInputsDataDistribution([]))
            )
            .withNoOfFeatures(kwargs.get("features", 6))
            .withRetries(kwargs.get("retries", 3))
            .withProximityFilteredDatasetMinimum(kwargs.get("dataset_minimum", 10))
            .withSeparableDatasetRatio(kwargs.get("separable_dataset_ratio", 0.1))
            .withProximityKernelWidth(kwargs.get("kernel_width", 0.5))
            .withProximityThreshold(kwargs.get("proximity_threshold", 0.83))
            .withEncodingParams(EncodingParams(ep[0], ep[1]))
            .withAdaptiveVariance(kwargs.get("adapt_dataset_variance", True))
            .withFeatureSelection(kwargs.get("feature_selection", True))
            .withPenalizeBalanceSparse(kwargs.get("penalise_sparse_balance", True))
            .withFilterInterpretable(kwargs.get("filter_interpretable", False))
            .withUseWLRLinearModel(kwargs.get("use_wlr_model", True))
            .withTrackCounterfactuals(kwargs.get("track_counterfactuals", False))
        )

        self._explainer = _LimeExplainer(self._lime_config)

    @data_conversion_docstring("one_input", "one_output")
    def explain(
        self,
        inputs: OneInputUnionType,
        outputs: OneOutputUnionType,
        model: Union[PredictionProvider, Model],
    ) -> LimeResults:
        """Produce a LIME explanation.

        Parameters
        ----------
        inputs : {}
            The input features to the model, as a: {}
        outputs : {}
            The corresponding model outputs for the provided features, that is,
            ``outputs = model(input_features)``. These can take the form of a: {}
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model`
            or :class:`~trustyai.model.ArrowModel`.

        Returns
        -------
        :class:`~LimeResults`
            Object containing the results of the LIME explanation.
        """
        feature_names = model.feature_names if isinstance(model, Model) else None
        output_names = model.output_names if isinstance(model, Model) else None
        _prediction = simple_prediction(inputs, outputs, feature_names, output_names)

        with Model.ArrowTransmission(model, inputs):
            return LimeResults(self._explainer.explainAsync(_prediction, model).get())
