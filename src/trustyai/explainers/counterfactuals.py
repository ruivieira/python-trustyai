"""Explainers.countefactual module"""
# pylint: disable = import-error, too-few-public-methods, wrong-import-order, line-too-long,
# pylint: disable = unused-argument
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import uuid as _uuid

from trustyai import _default_initializer  # pylint: disable=unused-import
from .explanation_results import ExplanationResults
from trustyai.utils._visualisation import (
    DEFAULT_STYLE as ds,
    DEFAULT_RC_PARAMS as drcp,
)

from trustyai.model import (
    counterfactual_prediction,
    PredictionInput,
)

from trustyai.utils.data_conversions import (
    prediction_object_to_numpy,
    prediction_object_to_pandas,
    OneInputUnionType,
    OneOutputUnionType,
    data_conversion_docstring,
)

from org.kie.trustyai.explainability.local.counterfactual import (
    CounterfactualExplainer as _CounterfactualExplainer,
    CounterfactualResult as _CounterfactualResult,
    SolverConfigBuilder as _SolverConfigBuilder,
    CounterfactualConfig as _CounterfactualConfig,
)
from org.kie.trustyai.explainability.model import (
    DataDistribution,
    PredictionProvider,
)
from org.optaplanner.core.config.solver.termination import TerminationConfig
from java.lang import Long

SolverConfigBuilder = _SolverConfigBuilder
CounterfactualConfig = _CounterfactualConfig


class CounterfactualResult(ExplanationResults):
    """Wraps Counterfactual results. This object is returned by the
    :class:`~CounterfactualExplainer`, and provides a variety of methods to visualize and interact
    with the results of the counterfactual explanation.
    """

    def __init__(self, result: _CounterfactualResult) -> None:
        """Constructor method. This is called internally, and shouldn't ever need to be
        used manually."""
        self._result = result

    @property
    def proposed_features_array(self):
        """Return the proposed feature values found from the counterfactual explanation
        as a Numpy array.
        """
        return prediction_object_to_numpy(
            [PredictionInput([entity.as_feature() for entity in self._result.entities])]
        )

    @property
    def proposed_features_dataframe(self):
        """Return the proposed feature values found from the counterfactual explanation
        as a Pandas DataFrame.
        """
        return prediction_object_to_pandas(
            [PredictionInput([entity.as_feature() for entity in self._result.entities])]
        )

    def as_dataframe(self) -> pd.DataFrame:
        """
        Return the counterfactual result as a dataframe

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the results of the counterfactual explanation, containing the
            following columns:

            * ``Features``: The names of each input feature.
            * ``Proposed``: The found values of the features.
            * ``Original``: The original feature values.
            * ``Constrained``: Whether this feature was constrained (held fixed) during the search.
            * ``Difference``: The difference between the proposed and original values.
        """
        entities = self._result.entities
        features = self._result.getFeatures()

        data = {}
        data["features"] = [f"{entity.as_feature().getName()}" for entity in entities]
        data["proposed"] = [entity.as_feature().value.as_obj() for entity in entities]
        data["original"] = [
            feature.getValue().getUnderlyingObject() for feature in features
        ]
        data["constrained"] = [feature.is_constrained for feature in features]

        dfr = pd.DataFrame.from_dict(data)
        dfr["difference"] = dfr.proposed - dfr.original
        return dfr

    def as_html(self) -> pd.io.formats.style.Styler:
        """
        Return the counterfactual result as a Pandas Styler object.

        Returns
        -------
        pandas.Styler
            Styler containing the results of the counterfactual explanation, in the same
            schema as in :func:`as_dataframe`. Currently, no default styles are applied
            in this particular function, making it equivalent to :code:`self.as_dataframe().style`.
        """
        return self.as_dataframe().style

    def plot(self) -> None:
        """
        Plot the counterfactual result.
        """
        _df = self.as_dataframe().copy()
        _df = _df[_df["difference"] != 0.0]

        def change_colour(value):
            if value == 0.0:
                colour = ds["neutral_primary_colour"]
            elif value > 0:
                colour = ds["positive_primary_colour"]
            else:
                colour = ds["negative_primary_colour"]
            return colour

        with mpl.rc_context(drcp):
            colour = _df["difference"].transform(change_colour)
            plot = _df[["features", "proposed", "original"]].plot.barh(
                x="features", color={"proposed": colour, "original": "black"}
            )
            plot.set_title("Counterfactual")
            plt.show()


class CounterfactualExplainer:
    """*"How do I get the result I want?"*

    The CounterfactualExplainer class seeks to answer this question by exploring "what-if"
    scenarios. Given some initial input and desired outcome, the counterfactual explainer tries to
    find a set of nearby inputs that produces the desired outcome. Mathematically, if we have a
    model :math:`f`, some input :math:`x` and a desired model output :math:`y'`, the counterfactual
    explainer finds some nearby input :math:`x'` such that :math:`f(x') = y'`.
    """

    def __init__(self, steps=10_000):
        """
        Build a new counterfactual explainer.

        Parameters
        ----------
        steps: int
            The number of search steps to perform during the counterfactual search.
        """
        self._termination_config = TerminationConfig().withScoreCalculationCountLimit(
            Long.valueOf(steps)
        )
        self._solver_config = (
            SolverConfigBuilder.builder()
            .withTerminationConfig(self._termination_config)
            .build()
        )
        self._cf_config = CounterfactualConfig().withSolverConfig(self._solver_config)

        self._explainer = _CounterfactualExplainer(self._cf_config)

    # pylint: disable=too-many-arguments
    @data_conversion_docstring("one_input", "one_output")
    def explain(
        self,
        inputs: OneInputUnionType,
        goal: OneOutputUnionType,
        model: PredictionProvider,
        data_distribution: Optional[DataDistribution] = None,
        uuid: Optional[_uuid.UUID] = None,
        timeout: Optional[float] = None,
    ) -> CounterfactualResult:
        """Request for a counterfactual explanation given a list of features, goals and a
        :class:`~PredictionProvider`

        Parameters
        ----------
        inputs : {}
            List of input features, as a: {}
        goal : {}
            The desired model outputs to be searched for in the counterfactual explanation.
            These can take the form of a: {}
        model : :obj:`~trustyai.model.PredictionProvider`
            The TrustyAI PredictionProvider, as generated by :class:`~trustyai.model.Model` or
             :class:`~trustyai.model.ArrowModel`.
        data_distribution : Optional[:class:`DataDistribution`]
            The :class:`DataDistribution` to use when sampling the inputs.
        uuid : Optional[:class:`_uuid.UUID`]
            The UUID to use during search.
        timeout : Optional[float]
                The timeout time in seconds of the counterfactual explanation.
        Returns
        -------
        :class:`~CounterfactualResult`
            Object containing the results of the counterfactual explanation.
        """
        _prediction = counterfactual_prediction(
            input_features=inputs,
            outputs=goal,
            data_distribution=data_distribution,
            uuid=uuid,
            timeout=timeout,
        )
        return CounterfactualResult(
            self._explainer.explainAsync(_prediction, model).get()
        )
