from typing import List
import math
import torch
from nltk.tokenize import sent_tokenize
from transformers import PreTrainedModel, PreTrainedTokenizer


class MetaChunker:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        threshold: float = 1.5,
        batch_size: int = 4096,
        max_context_size: int = 9000,
        dynamic_merge: bool = False,
        target_chunk_size: int = 200,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.batch_size = batch_size
        self.max_context_size = max_context_size
        self.dynamic_merge = dynamic_merge
        self.target_chunk_size = target_chunk_size
        self.device = model.device

    def split_text(self, text: str) -> List[str]:
        """Main entry point for semantic text chunking"""
        if not text.strip():
            return []

        # Preprocess and segment text
        sentences = self._segment_text(text)

        # Handle short texts directly
        if len(sentences) <= 3:
            return [" ".join(sentences)]

        # Process through perplexity analysis
        perplexities = self._calculate_perplexities(sentences)
        split_points = self._find_semantic_boundaries(perplexities)

        # Create chunks based on boundaries
        chunks = self._form_chunks(sentences, split_points)

        # Post-process chunks if needed
        if self.dynamic_merge:
            chunks = self._merge_small_chunks(chunks)

        return chunks

    def _segment_text(self, text: str) -> List[str]:
        """Split text into meaningful sentences/segments"""
        return [s.strip() for s in sent_tokenize(text) if s.strip()]

    def _calculate_perplexities(self, sentences: List[str]) -> List[float]:
        """Calculate perplexity scores for sentence boundaries"""
        input_ids, attention_mask, sentence_lengths = self._prepare_inputs(sentences)
        losses = self._process_in_batches(input_ids, attention_mask)
        return self._compute_sentence_perplexities(losses, sentence_lengths)

    def _prepare_inputs(self, sentences: List[str]):
        """Tokenize sentences and prepare model inputs"""
        input_ids = torch.tensor([[]], device=self.device, dtype=torch.long)
        attention_mask = torch.tensor([[]], device=self.device, dtype=torch.long)
        sentence_lengths = []

        for sent in sentences:
            tokenized = self.tokenizer(
                sent, return_tensors="pt", add_special_tokens=False
            )
            input_ids = torch.cat(
                [input_ids, tokenized["input_ids"].to(self.device)], dim=-1
            )
            attention_mask = torch.cat(
                [attention_mask, tokenized["attention_mask"].to(self.device)], dim=-1
            )
            sentence_lengths.append(tokenized["input_ids"].shape[1])

        return input_ids, attention_mask, sentence_lengths

    def _process_in_batches(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Process text in manageable batches"""
        total_batches = math.ceil(input_ids.shape[1] / self.batch_size)
        losses = torch.tensor([], device=self.device)
        past_key_values = None

        for batch_idx in range(total_batches):
            batch_inputs, batch_mask, past_key_values = self._prepare_batch(
                input_ids, attention_mask, batch_idx, past_key_values
            )

            batch_loss, past_key_values = self._compute_batch_perplexity(
                batch_inputs, batch_mask, past_key_values
            )

            losses = torch.cat([losses, batch_loss], dim=-1)

        return losses

    def _prepare_batch(self, input_ids, attention_mask, batch_idx, past_key_values):
        """Prepare a processing batch with context management"""
        start = batch_idx * self.batch_size
        end = start + self.batch_size

        batch_inputs = input_ids[:, start:end]
        batch_mask = attention_mask[:, :end]

        # Add context token
        batch_inputs = torch.cat(
            [
                self.tokenizer(" ", return_tensors="pt", add_special_tokens=False)[
                    "input_ids"
                ].to(self.device),
                batch_inputs,
            ],
            dim=-1,
        )

        # Manage attention mask growth
        batch_mask = torch.cat(
            [
                batch_mask,
                torch.ones((1, batch_idx + 1), device=self.device, dtype=torch.long),
            ],
            dim=-1,
        )

        # Prune past key values if needed
        if batch_mask.shape[1] > self.max_context_size and past_key_values is not None:
            past_key_values = [
                [
                    k[:, :, -(self.max_context_size // 2) :],
                    v[:, :, -(self.max_context_size // 2) :],
                ]
                for k, v in past_key_values
            ]

        return batch_inputs, batch_mask, past_key_values

    def _compute_batch_perplexity(self, inputs, mask, past_key_values):
        """Calculate perplexity for a single batch"""
        with torch.no_grad():
            outputs = self.model(
                inputs,
                attention_mask=mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits[..., :-1, :].contiguous()
        labels = inputs[..., 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss(reduction="none")(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        return loss, outputs.past_key_values

    def _compute_sentence_perplexities(
        self, losses: torch.Tensor, lengths: List[int]
    ) -> List[float]:
        """Convert raw losses to sentence-level perplexity scores"""
        perplexities = []
        pointer = 0

        for i, length in enumerate(lengths):
            if i == 0:
                segment = losses[1:length]
            else:
                segment = losses[pointer : pointer + length]

            perplexities.append(segment.mean().item())
            pointer += length

        return perplexities

    def _find_semantic_boundaries(self, perplexities: List[float]) -> List[int]:
        """Identify chunk boundaries based on perplexity minima"""
        minima = []
        for i in range(1, len(perplexities) - 1):
            if self._is_significant_minima(perplexities, i):
                minima.append(i)
        return minima

    def _is_significant_minima(self, values: List[float], index: int) -> bool:
        """Determine if a point is a meaningful minima"""
        left_diff = values[index - 1] - values[index]
        right_diff = values[index + 1] - values[index]
        return (left_diff >= self.threshold or right_diff >= self.threshold) or (
            values[index] < values[index - 1] and values[index] == values[index + 1]
        )

    def _form_chunks(self, sentences: List[str], split_points: List[int]) -> List[str]:
        """Create text chunks from detected boundaries"""
        boundaries = [0] + split_points + [len(sentences)]
        return [
            " ".join(sentences[start:end])
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Combine small chunks while preserving semantic boundaries"""
        merged = []
        current = []

        for chunk in chunks:
            chunk_tokens = chunk.split()
            current_tokens = len(current) if current else 0

            if current_tokens + len(chunk_tokens) <= self.target_chunk_size:
                current.append(chunk)
            else:
                merged.append(" ".join(current))
                current = [chunk]

        if current:
            merged.append(" ".join(current))

        return merged
