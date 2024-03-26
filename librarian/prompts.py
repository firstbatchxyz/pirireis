SYS_PROMPT = (
    "You are a network graph maker who extracts terms and their relations from a given context. "
    "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
    "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
    "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
        "\tTerms may include object, entity, location, organization, person, \n"
        "\tcondition, acronym, documents, service, concept, etc.\n"
        "\tTerms should be as atomistic as possible\n\n"
    "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
        "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
        "\tTerms can be related to many other terms\n\n"
    "Thought 3: Find out the relation between each such related pair of terms. \n\n"
    "Format your output as a list of json. Each element of the list contains a pair of terms"
    "and the relation between them, like the following: \n"
    "[\n"
    "   {\n"
    '       "node_1": "A concept from extracted ontology",\n'
    '       "node_2": "A related concept from extracted ontology",\n'
    '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
    "   }, {...}\n"
    "]"
)

SYS_PROMPT_H = (
    "You are a network graph maker who extracts terms and their conceptual distance from a given context. "
    "Conceptual distance refers to the number of nodes separating two concepts in a hypothetical graph that connects every concept in existence. "
    "You are provided with a context chunk (delimited by ```). Your task is to extract the ontology of terms mentioned in the given context. "
    "These terms should represent the key concepts as per the context. \n"
    "Thought 1: While traversing through each sentence, think about the key terms mentioned in it.\n"
    "\tTerms may include objects, entities, locations, organizations, persons, conditions, acronyms, documents, services, concepts, etc.\n"
    "\tTerms should be as atomistic as possible.\n\n"
    "Thought 2: Think about how these terms can have one-on-one relationships with other terms.\n"
    "\tTerms mentioned in the same sentence or paragraph are typically related to each other.\n"
    "\tTerms can be related to many other terms.\n\n"
    "Thought 3: Imagine a giant graph connecting every single concept in existence. "
    "Determine the number of nodes separating node_1 and node_2 in this hypothetical graph.\n"
    "If there are multiple possible paths, choose the most reasonable one based on the given context and domain knowledge.\n\n"
    "Format your output as a list of JSON objects. Each object should contain a pair of terms, their conceptual distance, "
    "a description of the path between them, and a confidence score, like the following: \n"
    "[\n"
    "  {\n"
    '    "node_1": "A concept from extracted ontology",\n'
    '    "node_2": "A related concept from extracted ontology",\n'
    '    "edge": "Number of nodes separating node_1 and node_2 in the hypothetical graph",\n'
    '    "desc": "A comma-separated list of representative nodes in the path from node_1 to node_2",\n'
    '    "confidence": 0.85\n'
    "  },\n"
    "  {...}\n"
    "]"
)

SYS_PROMPT_H2 = (
    "You are a network graph maker who extracts terms and their conceptual distance from a given context. "
    "Conceptual distance refers to the number of nodes separating two concepts in a hypothetical graph that connects related concepts. "
    "You are provided with a context chunk (delimited by ```). Your task is to extract the ontology of terms mentioned in the given context. "
    "These terms should represent the key concepts as per the context. \n"
    "Thought 1: While traversing through each sentence, think about the key terms mentioned in it.\n"
    "\tTerms may include objects, entities, locations, organizations, persons, conditions, acronyms, documents, services, concepts, etc.\n"
    "\tTerms should be as atomistic as possible.\n\n"
    "Thought 2: Think about how these terms can have one-on-one relationships with other terms.\n"
    "\tTerms mentioned in the same sentence or paragraph are typically related to each other.\n"
    "\tTerms can be related to many other terms.\n\n"
    "Thought 3: Imagine a hypothetical graph where related concepts are connected by edges. "
    "Determine the conceptual distance between node_1 and node_2 by counting the minimum number of edges separating them in this graph.\n"
    "If the concepts are directly related, the conceptual distance is 1. If they are not directly related, estimate the number of intermediate nodes connecting them.\n\n"
    "Format your output as a list of JSON objects. Each object should contain a pair of terms, their conceptual distance, "
    "a list of intermediate nodes connecting them (if applicable), and a confidence score, like the following: \n"
    "[\n"
    "  {\n"
    '    "node_1": "A concept from extracted ontology",\n'
    '    "node_2": "A related concept from extracted ontology",\n'
    '    "distance": 2,\n'
    '    "path": ["intermediate_node_1", "intermediate_node_2"],\n'
    '    "confidence": 0.8\n'
    "  },\n"
    "  {...}\n"
    "]"
)

SYS_PROMPT_CORRECTIVE_H2 = (
    "I apologize, but the output you provided is not a valid JSON format. "
    "Please make sure to follow the specified format and provide a list of JSON objects. "
    "Each JSON object should contain the following fields: "
    '"node_1" (string): A concept from the extracted ontology.\n'
    '"node_2" (string): A related concept from the extracted ontology.\n'
    '"distance" (integer): The conceptual distance between node_1 and node_2.\n'
    '"path" (list of strings): The list of intermediate nodes connecting node_1 and node_2 (if applicable).\n'
    '"confidence" (float): A confidence score between 0 and 1 indicating the certainty of the relationship.\n'
    "Please ensure that the JSON objects are properly formatted, with correct syntax and data types. "
    "Here's an example of the expected format:\n"
    "[\n"
    "  {\n"
    '    "node_1": "concept1",\n'
    '    "node_2": "concept2",\n'
    '    "distance": 2,\n'
    '    "path": ["intermediate_concept1", "intermediate_concept2"],\n'
    '    "confidence": 0.85\n'
    "  },\n"
    "  {\n"
    '    "node_1": "concept3",\n'
    '    "node_2": "concept4",\n'
    '    "distance": 1,\n'
    '    "path": [],\n'
    '    "confidence": 0.95\n'
    "  }\n"
    "]\n"
    "Please revise your output to adhere to this format and provide a valid JSON response."
)

SYS_PROMPT_CORRECTIVE_ = (
"It seems like the JSON output in your previous response had some errors or missing elements. "
"Let's fix the given text and make it valid JSON format. "
)

SYS_PROMPT_CORRECTIVE = (
"It seems like the JSON output in your previous response had some errors or missing elements. "
"Let's try again with a focus on generating a valid JSON format. "
"Please make sure to follow these guidelines:\n"
"1. The output should be a valid JSON list, enclosed in square brackets [].\n"
"2. Each element in the list should be a JSON object, enclosed in curly braces {}.\n"
"3. Each JSON object should contain three key-value pairs: 'node_1', 'node_2', and 'edge'.\n"
"4. The values for 'node_1' and 'node_2' should be strings representing concepts from the extracted ontology.\n"
"5. The value for 'edge' should be a string describing the relationship between 'node_1' and 'node_2' in one or two sentences.\n"
"6. Make sure to enclose all string values in double quotes "".\n"
"7. Separate each key-value pair with a comma ,\n"
"8. Ensure that there are no trailing commas after the last key-value pair in each JSON object.\n\n"
"Here's an example of the expected JSON format:\n"
"[\n"
"  {\n"
'    "node_1": "A concept from extracted ontology",\n'
'    "node_2": "A related concept from extracted ontology",\n'
'    "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
"  },\n"
"  {\n"
'    "node_1": "Another concept from extracted ontology",\n'
'    "node_2": "Another related concept from extracted ontology",\n'
'    "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
"  }\n"
"]\n\n"
"Please regenerate the JSON output for the given context, ensuring that it adheres to the specified format and guidelines."
)

def create_prompt(input):
    return f"context: ```{input}``` \n\n output: "