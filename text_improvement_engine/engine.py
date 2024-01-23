import logging
import math
import time
from dataclasses import dataclass

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('main')


@dataclass
class ReplacementCandidate:
    """
    Data structure to store a potential replacement.

    Attributes:
        original_phrase (str): The original phrase from the text.
        suggested_phrase (str): The phrase suggested as a replacement.
        similarity_score (float): The cosine similarity score between
                                  the original and suggested phrase.
    """

    original_phrase: str
    suggested_phrase: str
    similarity_score: float


MIN_NGRAM_LENGTH = 2
MAX_NGRAM_LENGTH = 4


class TextImprovementEngine:
    """
    A text analysis engine that suggests improvements to a given text by comparing
    its phrases to a set of standard phrases using semantic similarity.

    Attributes:
        model (BertModel): The BERT model used for generating phrase embeddings.
        tokenizer (BertTokenizer): The tokenizer corresponding to the BERT model.
        standard_phrases (list[str]): A list of standard phrases for comparison.
        standard_embeddings (numpy.ndarray): Embeddings of the standard phrases.
    """

    def __init__(self, standard_phrases_path: str = "standard_phrases.txt"):
        """
        Initialise the text improvement engine.

        Args:
            standard_phrases_path (str): Path to the file containing standard phrases.
        """
        start_time = time.time()
        self._download_nltk_resource("punkt")

        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.standard_phrases = self.load_standard_phrases(standard_phrases_path)
        self.standard_embeddings = self.get_embeddings(self.standard_phrases)

        init_time = time.time() - start_time
        logger.info(f"Initialising engine took {init_time:.2f}s")

    def analyse_text_from_file(self, file_path: str):
        """
        Analyze text from a file, suggesting phrase replacements based on semantic similarity.

        Args:
            file_path (str): The path to the text file to be analyzed.

        Returns:
            dict[str, ReplacementCandidate]: A dictionary of suggestions with
                                             the suggested phrase as key.
        """
        text = self.load_text(file_path)
        return self.analyse_text(text)

    def analyse_text(
            self,
            text: str,
            min_similarity_score: float = 0.75
    ) -> dict[str, ReplacementCandidate]:
        """
        Analyze text, suggesting phrase replacements based on semantic similarity.

        Args:
            text (str): The text to be analyzed.
            min_similarity_score (float): The minimum similarity score for a suggestion to be considered.

        Returns:
            dict[str, ReplacementCandidate]: A dictionary of suggestions with the suggested phrase as key.
        """
        start_time = time.time()

        text_phrases = self.get_text_phrases(text)
        text_embeddings = self.get_embeddings(text_phrases)
        final_candidates = {}

        for i, phrase_embedding in enumerate(text_embeddings):
            similarities = cosine_similarity([phrase_embedding], self.standard_embeddings)
            max_index = np.argmax(similarities)
            original_phrase = text_phrases[i]
            suggested_phrase = self.standard_phrases[max_index]
            similarity_score = similarities[0][max_index]
            candidate = ReplacementCandidate(
                original_phrase=original_phrase,
                suggested_phrase=suggested_phrase,
                similarity_score=similarity_score,
            )

            # We don't want exact matches,
            # it wouldn't make sense to suggest to replace X by X
            if math.isclose(similarity_score, 1.0, rel_tol=1e-3):
                continue

            # We don't really want partial matches either
            # This particular check could definitely be improved
            if suggested_phrase in original_phrase:
                continue

            # Only provide candidates with high enough similarity score (default 0.75)
            if similarity_score < min_similarity_score:
                continue

            if suggested_phrase in final_candidates:
                old_similarity_score = final_candidates[suggested_phrase].similarity_score
                if similarity_score > old_similarity_score:
                    final_candidates[suggested_phrase] = candidate
            else:
                final_candidates[suggested_phrase] = candidate

        analysing_time = time.time() - start_time
        logger.info(f"Analysing took {analysing_time:.2f}s")

        return final_candidates

    @staticmethod
    def get_text_phrases(text: str) -> list[str]:
        """
        Extract n-grams of varying lengths from the given text.
        """
        result = []
        sentences = sent_tokenize(text)
        for sentence in sentences:
            words = sentence.split()

            for n in range(MIN_NGRAM_LENGTH, MAX_NGRAM_LENGTH + 1):
                result.extend([" ".join(ngram) for ngram in ngrams(words, n)])

        return result

    @staticmethod
    def load_standard_phrases(standard_phrases_path: str) -> list[str]:
        """
        Load standard phrases from a file, one phrase per line.
        """
        with open(standard_phrases_path, 'r') as file:
            phrases = [line.strip().lower() for line in file.readlines()]

        return phrases

    @staticmethod
    def load_text(text_path: str) -> str:
        """
        Load text from a file, as one string.
        """
        with open(text_path, 'r') as file:
            text = file.readline()

        return text

    def get_embeddings(self, phrases: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of phrases using the BERT model.

        Args:
            phrases (list[str]): A list of phrases to generate embeddings for.

        Returns:
            numpy.ndarray: An array of embeddings.
        """
        embeddings = []
        for phrase in phrases:
            inputs = self.tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

        return np.vstack(embeddings)

    @staticmethod
    def _download_nltk_resource(resource: str):
        """
        Download an NLTK resource if it's missing.
        """
        try:
            # check if the resource is already downloaded
            nltk.data.find("tokenizers/" + resource)
        except LookupError:
            # if not, download it
            logger.info("Downloading resource...")
            nltk.download(resource)
