import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive

_DESCRIPTION = """ Test version of the Finnish Natural Questions dataset."""


class SquadV2Config(datasets.BuilderConfig):
    """BuilderConfig for NQ."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUADV2.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquadV2Config, self).__init__(**kwargs)


class SquadV2(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        SquadV2Config(name="nq_fi", version=datasets.Version(
            "1.0.0"), description="NQ in Finnish"),
    ]

    def _info(self):
        # TODO(squad_v2): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://turkunlp.org/",
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question",
                    context_column="context",
                    answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": "nq_fi/nq-train.json"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": "nq_fi/nq-dev.json"}
            ),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for example in squad["data"]:
                title = example.get("title", "")
                for paragraph in example["paragraphs"]:
                    # do not strip leading blank spaces GH-2585
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        question = qa["question"]
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"]
                                         for answer in qa["answers"]]
                        answers = [answer["text"].strip(
                            " .,-:") for answer in qa["answers"]]

                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
