test_questions = ["How retired people earn income?",
                  "I've have an headache, what should I do?",
                  "What is the capital of France?",
                  "How does compound interest work?",
                  "Is pasta italian?",
                  "What is renormalization in QED?"]

not_relevant = [
"What is the capital city of Mars?",
"How do I bake a cake using a microwave?"
]

vague = [
"What is that thing called?",
"How do I do it?"
]

not_factual = [
"What is your favorite color?",
"Do you think aliens exist?"
]

multiple_sub_questions = [
"What is the population of New York City, and how does it compare to the population of Los Angeles? Also, what are the major industries in each city?",
"Who won the World Cup in 2018, and what was the final score? Additionally, which country hosted the tournament that year?"
]

multi_hop = [
"Who was the director of the movie that won the Best Picture Oscar in the same year that the actor who played the lead role in Titanic won the Best Actor award?",
"What is the capital city of the country that is the largest producer of coffee in the world?"
]

non_semantic = [
"What is the meaning of life, the universe, and everything? [42]",
"How can I learn to play the guitar? [Insert random emoji here]"
]

conflicting_information = [
"Is the Great Wall of China visible from space or not?",
"Are tomatoes a fruit or a vegetable?"
]
"""
import spacy
nlp = spacy.load("en_core_web_sm")
from spacy_experimental.coref.coref_component import DEFAULT_COREF_MODEL
from spacy_experimental.coref.coref_util import DEFAULT_CLUSTER_PREFIX

config={
    "model": DEFAULT_COREF_MODEL,
    "span_cluster_prefix": DEFAULT_CLUSTER_PREFIX,
}
nlp.add_pipe("experimental_coref", config=config)
"""
from decouple import config
SECRET_KEY = config('DEEPINFRA_API_TOKEN')
print(SECRET_KEY)