import warnings
from typing import Dict, List, Any, Union
import math

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode

from allennlp_models.structured_prediction.metrics.srl_eval_scorer import (
    DEFAULT_SRL_EVAL_PATH,
    SrlEvalScorer,
)


@Model.register("srl_self_training_baseline")
class SrlSelfTrainingBaseline(Model):
    """

    A BERT based model [Simple BERT Models for Relation Extraction and Semantic Role Labeling (Shi et al, 2019)]
    (https://arxiv.org/abs/1904.05255) with some modifications (no additional parameters apart from a linear
    classification layer), which is currently the state-of-the-art single model for English PropBank SRL
    (Newswire sentences).

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.

    bert_model : `Union[str, Dict[str, Any], BertModel]`, required.
        A string describing the BERT model to load, a BERT config in the form of a dictionary,
        or an already constructed BertModel.

        !!! Note
            If you pass a config `bert_model` (a dictionary), pretrained weights will
            not be cached and loaded! This is ideal if you're loading this model from an
            AllenNLP archive since the weights you need will already be included in the
            archive, but not what you want if you're training.

    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.

    label_smoothing : `float`, optional (default = `0.0`)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.

    ignore_span_metric : `bool`, optional (default = `False`)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.

    srl_eval_path : `str`, optional (default=`DEFAULT_SRL_EVAL_PATH`)
        The path to the srl-eval.pl script. By default, will use the srl-eval.pl included with allennlp,
        which is located at allennlp/tools/srl-eval.pl . If `None`, srl-eval.pl is not used.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, Dict[str, Any], BertModel],
        embedding_dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        label_smoothing: float = None,
        ignore_span_metric: bool = False,
        srl_eval_path: str = DEFAULT_SRL_EVAL_PATH,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        elif isinstance(bert_model, dict):
            warnings.warn(
                "Initializing BertModel without pretrained weights. This is fine if you're loading "
                "from an AllenNLP archive, but not if you're training.",
                UserWarning,
            )
            bert_config = BertConfig.from_dict(bert_model)
            self.bert_model = BertModel(bert_config)
        else:
            self.bert_model = bert_model

        self.num_classes = self.vocab.get_vocab_size("labels")
        if srl_eval_path is not None:
            # For the span based evaluation, we don't want to consider labels
            # for verb, because the verb index is provided to the model.
            self.span_metric = SrlEvalScorer(srl_eval_path, ignore_classes=["V"])
        else:
            self.span_metric = None
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)

        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        initializer(self)

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        verb_indicator: torch.Tensor,
        metadata: List[Any],
        tags: torch.LongTensor = None,
        self_training: bool = False,
        weighted_self_training: bool = False
    ):

        """
        # Parameters

        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        verb_indicator: `torch.LongTensor`, required.
            An integer `SequenceFeatureField` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels
            of shape `(batch_size, num_tokens)`
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containing the original words in the sentence, the verb to compute the
            frame for, and start offsets for converting wordpieces back to a sequence of words,
            under 'words', 'verb' and 'offsets' keys, respectively.

        # Returns

        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            a distribution of the tag classes per word.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        """
        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.bert_model(
            input_ids=util.get_token_ids_from_text_field_tensors(tokens),
            token_type_ids=verb_indicator,
            attention_mask=mask,
            return_dict=False,
        )

        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()
        logits = self.tag_projection_layer(embedded_text_input)

        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
            [batch_size, sequence_length, self.num_classes]
        )
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.make_output_human_readable.
        output_dict["mask"] = mask
        # We add in the offsets here so we can compute the un-wordpieced tags.
        words, verbs, offsets = zip(*[(x["words"], x["verb"], x["offsets"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["verb"] = list(verbs)
        output_dict["wordpiece_offsets"] = list(offsets)

        if tags is not None:
            if not self_training:
                loss = sequence_cross_entropy_with_logits(
                    logits, tags, mask, label_smoothing=self._label_smoothing
                )
                if not self.ignore_span_metric and self.span_metric is not None and not self.training:
                    #------------- get disagreement scores --------------#
                    # if "parse_spans" in metadata[0]:
                    #     batch_parse_spans = [example_metadata["parse_spans"] for example_metadata in metadata]
                    #     batch_bio_predicted_tags, pred_tags = self.get_pred_tags(output_dict)
                    #     batch_pred_srl_spans = self.bio2spans(batch_bio_predicted_tags,metadata)
                    #     scores, num_instance = self.measure_disagreement(batch_parse_spans,batch_pred_srl_spans)
                    #     output_dict["disagr_score"] = torch.mean(scores)
                    output_dict["disagr_score"] = torch.zeros(1)
                    #------------------------------------------------#
                    output_dict["parse_spans"] = [example_metadata["parse_spans"] for example_metadata in metadata]
                    output_dict["gold_tags"] = [example_metadata["gold_tags"] for example_metadata in metadata]
                    batch_verb_indices = [
                        example_metadata["verb_index"] for example_metadata in metadata
                    ]
                    batch_sentences = [example_metadata["words"] for example_metadata in metadata]
                    # Get the BIO tags from make_output_human_readable()
                    # TODO (nfliu): This is kind of a hack, consider splitting out part
                    # of make_output_human_readable() to a separate function.
                    batch_bio_predicted_tags = self.make_output_human_readable(output_dict).pop("tags")
                    from allennlp_models.structured_prediction.models.srl import (
                        convert_bio_tags_to_conll_format,
                    )

                    batch_conll_predicted_tags = [
                        convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
                    ]
                    batch_bio_gold_tags = [
                        example_metadata["gold_tags"] for example_metadata in metadata
                    ]
                    batch_conll_gold_tags = [
                        convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
                    ]
                    self.span_metric(
                        batch_verb_indices,
                        batch_sentences,
                        batch_conll_predicted_tags,
                        batch_conll_gold_tags,
                    )
            else:
                # print('tags: ',tags)
                # print('mask length : ',mask.sum(-1))
                # print('tokens: ', util.get_token_ids_from_text_field_tensors(tokens))
                batch_parse_spans = [example_metadata["parse_spans"] for example_metadata in metadata]
                batch_bio_predicted_tags, pred_tags = self.get_pred_tags(output_dict)
                if weighted_self_training:
                    batch_pred_srl_spans = self.bio2spans(batch_bio_predicted_tags,metadata)
                    scores, num_instance = self.measure_disagreement(batch_parse_spans,batch_pred_srl_spans)
                    scores = scores.to(device=output_dict['logits'].device)
                    batch_loss = sequence_cross_entropy_with_logits(
                        logits, pred_tags, mask, average=None, label_smoothing=self._label_smoothing
                    )
                    scaled_loss = scores * batch_loss * -1
                    if num_instance >0:
                        loss = torch.sum(scaled_loss) / num_instance
                    else:
                        loss = torch.mean(scaled_loss)
                    # pass batch disagreement scores
                    output_dict["disagr_score"] = torch.mean(scores)
                    # print('weighted self-training loss',loss)
                else:
                    loss = sequence_cross_entropy_with_logits(
                        logits, pred_tags, mask, label_smoothing=self._label_smoothing
                    )
                    # print('self-training loss',loss)

            output_dict["loss"] = loss
        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        `"tags"` key to the dictionary with the result.

        NOTE: First, we decode a BIO sequence on top of the wordpieces. This is important; viterbi
        decoding produces low quality output if you decode on top of word representations directly,
        because the model gets confused by the 'missing' positions (which is sensible as it is trained
        to perform tagging on wordpieces, not words).

        Secondly, it's important that the indices we use to recover words from the wordpieces are the
        start_offsets (i.e offsets which correspond to using the first wordpiece of words which are
        tokenized into multiple wordpieces) as otherwise, we might get an ill-formed BIO sequence
        when we select out the word tags from the wordpiece tags. This happens in the case that a word
        is split into multiple word pieces, and then we take the last tag of the word, which might
        correspond to, e.g, I-V, which would not be allowed as it is not preceeded by a B tag.
        """
        all_predictions = output_dict["class_probabilities"]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [
                all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))
            ]
        else:
            predictions_list = [all_predictions]
        wordpiece_tags = []
        word_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        for predictions, length, offsets in zip(
            predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]
        ):
            max_likelihood_sequence, _ = viterbi_decode(
                predictions[:length], transition_matrix, allowed_start_transitions=start_transitions
            )
            tags = [
                self.vocab.get_token_from_index(x, namespace="labels")
                for x in max_likelihood_sequence
            ]

            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])
        output_dict["wordpiece_tags"] = wordpiece_tags
        output_dict["tags"] = word_tags
        return output_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            return {}

        else:
            metric_dict = self.span_metric.get_metric(reset=reset)

            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "overall" in x}

    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        # Returns

        transition_matrix : `torch.Tensor`
            A `(num_labels, num_labels)` matrix of pairwise potentials.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == "I" and not previous_label == "B" + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix

    def get_start_transitions(self):
        """
        In the BIO sequence, we cannot start the sequence with an I-XXX tag.
        This transition sequence is passed to viterbi_decode to specify this constraint.

        # Returns

        start_transitions : `torch.Tensor`
            The pairwise potentials between a START token and
            the first token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)

        start_transitions = torch.zeros(num_labels)

        for i, label in all_labels.items():
            if label[0] == "I":
                start_transitions[i] = float("-inf")

        return start_transitions

    def get_pred_tags(self, output_dict: Dict[str, torch.Tensor]):
        all_predictions = output_dict["class_probabilities"] # batch, sequence, num_tags
        pred_tags = torch.zeros_like(output_dict["logits"][:,:,0], device=output_dict["logits"].device, dtype=torch.long)

        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [
                all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))
            ]
        else:
            predictions_list = [all_predictions]
        wordpiece_tags = []
        word_tags = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        start_transitions = self.get_start_transitions()
        # **************** Different ********************
        # We add in the offsets here so we can compute the un-wordpieced tags.
        count = 0
        for predictions, length, offsets in zip(
            predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]
        ):
            max_likelihood_sequence, _ = viterbi_decode(
                predictions[:length], transition_matrix, allowed_start_transitions=start_transitions
            )
            pred_tags[count,:length] = torch.tensor(max_likelihood_sequence)
            count+=1
            tags = [
                self.vocab.get_token_from_index(x, namespace="labels")
                for x in max_likelihood_sequence
            ]

            wordpiece_tags.append(tags)
            # print('tags length',len(tags))
            word_tags.append([tags[i] for i in offsets])
        return word_tags, pred_tags
    
    def bio2spans(self, batch_bio_predicted_tags,metadata):
        def tags2spans(tags):
            # print(tags)
            spans = []
            start = 0
            # print(tags)
            if tags[0] == "O":
                prev_tag = tags[0]
            else:
                if not tags[0][:2] == "B-":
                    print("tags sequence is not correct")
                    return None
                prev_tag = tags[0][2:]
            for pos in range(1,len(tags)):
                curr_tag = tags[pos]
                if curr_tag == "O":
                    if not prev_tag == "O":
                        spans.append([start,pos-start,prev_tag])
                        prev_tag = "O"
                        # start = pos
                elif curr_tag[:2] == "I-":
                    if not prev_tag == curr_tag[2:]:
                        # print(tags)
                        # print(prev_tag, curr_tag, pos)
                        print("tags sequence is not correct")
                        return None
                elif curr_tag[:2] == "B-":
                    if prev_tag == "O":
                        prev_tag = curr_tag[2:]
                        start = pos
                    else:
                        spans.append([start,pos-start,prev_tag])
                        prev_tag = curr_tag[2:]
                        start = pos
            curr_tag = tags[-1]
            if not curr_tag == "O":
                spans.append([start,len(tags)-start,prev_tag])
            # print(spans)
            return spans

        batch_spans = []
        for bio_predicted_tags, sample_metadata in zip(batch_bio_predicted_tags,metadata):
            spans = tags2spans(bio_predicted_tags)
            # print(bio_predicted_tags)
            # print(sample_metadata['gold_tags'])
            # print(spans)
            batch_spans.append(spans)
        
        return batch_spans

    def measure_disagreement(self, batch_parse_spans, batch_pred_srl_spans):
        batch_size = len(batch_parse_spans)
        scores = torch.zeros(batch_size)
        num_instance = 0
        for i in range(batch_size):
            parse_spans = batch_parse_spans[i]
            pred_spans = batch_pred_srl_spans[i]
            # print('parse spans: ',parse_spans)
            # print('pred_spans: ',pred_spans)
            if pred_spans is None:
                scores[i] = 1.0
                num_instance += 1
                continue
            if parse_spans is not None:
                parse_spans = set([(span[0],span[1]) for span in parse_spans])
                # print("gold: ",parse_spans)
            else:
                scores[i] = 0.0
                continue
            # pred_spans = set([(span[0],span[1]) for span in pred_spans if not span[2]=='V'])
            # pred_spans = set([(span[0],span[1]) for span in pred_spans if span[1]>1])
            # print('pred: ',pred_spans)
            # print('parse_spans: ',parse_spans)
            # print('pred_spans: ',pred_spans)
            # if len(pred_spans) > 0:
            #     _score = 2 * len(pred_spans - pred_spans.intersection(parse_spans))/len(pred_spans) - 1.0
            #     num_instance += 1
            # else:
            #     _score = 0.0
            # print('intersection: ',pred_spans.intersection(parse_spans))
            # print(_score)
            scores[i], valid_inst = self.score_function(pred_spans, parse_spans)
            num_instance += valid_inst
            # print(scores[i])
        
        return scores, num_instance

    def score_function(self, pred_spans, parse_spans):
        total_num_span = len(pred_spans)
        pred_spans = set([(span[0],span[1]) for span in pred_spans if span[1]>1])
        # exp score function 
        # if len(pred_spans) < 1:
        #     return 0.0, False
        # else:
        #     num_violate_span = len(pred_spans - pred_spans.intersection(parse_spans))
        #     if num_violate_span > 0:
        #         violate_rate = num_violate_span/len(pred_spans)
        #         score = (math.exp(violate_rate) - 1)/(math.exp(1) - 1)
        #     else:
        #         score = -len(pred_spans)/total_num_span
        #     # print('current score',score)
        #     return score, True

        # square root score function
        # if len(pred_spans) < 1:
        #     return 0.0, False
        # else:
        #     num_violate_span = len(pred_spans - pred_spans.intersection(parse_spans))
        #     if num_violate_span > 0:
        #         violate_rate = num_violate_span/len(pred_spans)
        #         score = math.sqrt(violate_rate)
        #     else:
        #         score = -len(pred_spans)/total_num_span
        #     # print('current score',score)
        #     return score, True
        
        # step function
        # if len(pred_spans) < 1:
        #     return 0.0, False
        # else:
        #     num_violate_span = len(pred_spans - pred_spans.intersection(parse_spans))
        #     if num_violate_span > 0:
        #         # violate_rate = num_violate_span/len(pred_spans)
        #         score = 1
        #     else:
        #         consistant_rate = len(pred_spans)/total_num_span
        #         score = - (math.exp(consistant_rate) - 1)/(math.exp(1) - 1)
        #     # print('current score',score)
        #     return score, True

        # step harsh function
        # if len(pred_spans) < 1:
        #     return 0.0, False
        # else:
        #     num_violate_span = len(pred_spans - pred_spans.intersection(parse_spans))
        #     if num_violate_span > 0:
        #         # violate_rate = num_violate_span/len(pred_spans)
        #         score = 1
        #     else:
        #         consistant_rate = len(pred_spans)/total_num_span
        #         score = - consistant_rate**5
        #     # print('current score',score)
        #     return score, True

        # step function + length scale
        if len(pred_spans) < 1:
            return 0.0, False
        else:
            num_violate_span = len(pred_spans - pred_spans.intersection(parse_spans))
            if num_violate_span > 0:
                # violate_rate = num_violate_span/len(pred_spans)
                score = 1
            else:
                # consistant_rate = len(pred_spans)/total_num_span
                score = - math.exp(-1/3*(total_num_span-1))
            # print('current score',score)
            return score, True


            



    default_predictor = "semantic_role_labeling"
