"""
Code directly taken from https://github.com/allenai/allennlp/blob/master/allennlp/predictors/bidaf.py
Minor changes made to `dump_line` to output `id: answer`
"""

from copy import deepcopy
from typing import Dict, List

from overrides import overrides
import numpy
import json

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import (
    IndexField,
    ListField,
    LabelField,
    SpanField,
    SequenceLabelField,
    SequenceField,
)


@Predictor.register("qanet_predictor")
class QaNetPredictor(Predictor):
    def predict(self, question: str, passage: str) -> JsonDict:
        """
        Make a machine comprehension prediction on the supplied input.
        See https://rajpurkar.github.io/SQuAD-explorer/ for more information about the machine comprehension task.

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        passage : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"passage": passage, "question": question})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        passage_text = json_dict["passage"]
        return self._dataset_reader.text_to_instance(question_text, passage_text)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = deepcopy(instance)
        # For BiDAF
        if "best_span" in outputs:
            span_start_label = outputs["best_span"][0]
            span_end_label = outputs["best_span"][1]
            passage_field: SequenceField = new_instance["passage"]  # type: ignore
            new_instance.add_field(
                "span_start", IndexField(int(span_start_label), passage_field)
            )
            new_instance.add_field(
                "span_end", IndexField(int(span_end_label), passage_field)
            )
        return [new_instance]

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps({outputs["id"]: outputs["best_span_str"]}) + "\n"
