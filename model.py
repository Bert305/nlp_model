"""
LLM NLP Pipeline (OpenAI)
- Sentiment analysis
- Named Entity Recognition (NER)
- Summarization
- Zero-shot classification

Setup:
  pip install openai
  export OPENAI_API_KEY="YOUR_KEY"   # mac/linux
  setx OPENAI_API_KEY "YOUR_KEY"     # windows (new terminal)

Docs:
- Responses API: https://platform.openai.com/docs/api-reference/responses
- Structured outputs: https://platform.openai.com/docs/guides/structured-outputs
"""

# Enables postponed evaluation of type hints (helps avoid circular imports)
from __future__ import annotations

# Used to access environment variables like OPENAI_API_KEY
import os

# Used for type hints (helps readability + IDE support)
from typing import List, Dict, Any, Optional

# Loads environment variables from a .env file
from dotenv import load_dotenv

# Reads your .env file and injects variables into os.environ
load_dotenv()

# OpenAI Python SDK
from openai import OpenAI

# Hugging Face tokenizer + model loaders
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# PyTorch (used for running the HF model)
import torch


class LLMNlpPipeline:
    """
    A HuggingFace-like pipeline interface, but powered by an OpenAI LLM.

    Notes:
    - Uses Structured Outputs (JSON Schema) to ensure consistent JSON.
    - Replace `model` with whichever model you have access to.
    """

    def __init__(self, model: str = "gpt-4-mini"):
        # Create OpenAI client using API key from environment
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Safety check: crash if API key missing
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

        # Store OpenAI model name
        self.model_name = model  

        # HuggingFace model checkpoint name
        self.hf_model_name = "bert-base-uncased"

        # Load tokenizer for HF model (converts text → tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)

        # Load HF classification model (neural network weights)
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_name)

    def _call_json(self, instructions: str, user_input: str, schema_name: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls the Chat Completions API and enforces JSON Schema output.
        """

        # Call OpenAI with structured JSON output
        resp = self.client.beta.chat.completions.parse(
            model=self.model_name,  # Which OpenAI model to use
            messages=[
                {"role": "system", "content": instructions},  # System instructions
                {"role": "user", "content": user_input},      # User input
            ],
            response_format={
                "type": "json_schema",   # Enforce JSON
                "json_schema": {
                    "name": schema_name,
                    "strict": True,      # No extra fields allowed
                    "schema": schema,   # Expected response structure
                },
            },
        )

        # Import JSON library
        import json

        # Convert model response string → Python dict
        return json.loads(resp.choices[0].message.content)

    # -------------------------
    # 1) Sentiment Analysis
    # -------------------------
    def sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:

        # JSON schema describing sentiment output format
        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "label": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["text", "label", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        }

        # Instructions sent to LLM
        instructions = (
            "You are an NLP classifier. Classify sentiment for each input text. "
            "Return POSITIVE, NEGATIVE, NEUTRAL, or MIXED with a confidence 0-1."
        )

        # Package input texts
        user_input = {"texts": texts}

        # Call OpenAI + extract results
        return self._call_json(instructions, str(user_input), "sentiment_output", schema)["results"]

    # -------------------------
    # 2) Named Entity Recognition
    # -------------------------
    def ner(self, text: str) -> List[Dict[str, Any]]:

        # JSON schema defining NER output
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "label": {"type": "string", "enum": ["PERSON", "ORG", "LOC", "DATE", "GPE", "PRODUCT", "EVENT", "OTHER"]},
                            "start": {"type": "integer", "minimum": 0},
                            "end": {"type": "integer", "minimum": 0},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["text", "label", "start", "end", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["entities"],
            "additionalProperties": False,
        }

        # LLM instructions
        instructions = (
            "You are an NER system. Extract named entities from the text. "
            "Return character offsets (start, end) for each entity span, plus a label and confidence."
        )

        # Wrap text for OpenAI
        user_input = {"text": text}

        # Call OpenAI + return entities
        return self._call_json(instructions, str(user_input), "ner_output", schema)["entities"]

    # -------------------------
    # 3) Summarization
    # -------------------------
    def summarize(self, text: str, max_bullets: int = 5) -> Dict[str, Any]:

        # Schema for summary output
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": max_bullets
                },
            },
            "required": ["summary", "key_points"],
            "additionalProperties": False,
        }

        # Summarization instructions
        instructions = (
            "You are a helpful summarizer. Produce a short summary and key bullet points. "
            f"Return at most {max_bullets} bullet points."
        )

        # Wrap text
        user_input = {"text": text}

        # Call OpenAI
        return self._call_json(instructions, str(user_input), "summary_output", schema)

    # -------------------------
    # 4) Zero-shot Classification
    # -------------------------
    def zero_shot(self, text: str, candidate_labels: List[str]) -> Dict[str, Any]:

        # Schema for zero-shot output
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "labels_ranked": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "scores": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "required": ["text", "labels_ranked", "scores"],
            "additionalProperties": False,
        }

        # Zero-shot instructions
        instructions = (
            "You are a zero-shot classifier. Rank the candidate labels from most to least relevant "
            "for the given text. Return scores 0-1 aligned with labels_ranked."
        )

        # Package inputs
        user_input = {"text": text, "candidate_labels": candidate_labels}

        # Call OpenAI
        return self._call_json(instructions, str(user_input), "zero_shot_output", schema)

    def run_transformers_model(self, input_text: str):

        # Convert text into tensors for HF model
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # Disable gradients (faster inference)
        with torch.no_grad():
            outputs = self.hf_model(**inputs)

        # Extract raw scores
        logits = outputs.logits

        # Choose most likely class
        predictions = torch.argmax(logits, dim=-1)

        # Return predicted class index
        return predictions


# Only runs when file is executed directly
if __name__ == "__main__":

    # Initialize pipeline
    nlp = LLMNlpPipeline(model="gpt-4.1-mini")

    print("\n--- Sentiment ---")
    print(nlp.sentiment([
        "I've been waiting for a Hugging Face course my whole life.",
        "I hate this so much!"
    ]))

    print("\n--- NER ---")
    print(nlp.ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

    print("\n--- Summarize ---")
    print(nlp.summarize("Transformers are neural networks built around attention mechanisms...", max_bullets=3))

    print("\n--- Zero-shot ---")
    print(nlp.zero_shot(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business", "technology"]
    ))

    print("\n--- Transformers Model ---")
    print(nlp.run_transformers_model("Your input text here"))

