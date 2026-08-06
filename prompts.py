SYS_PROMPT_QA_GENERATION = """
    Your goal is to transform raw technical manual text into high-quality Instruction-Following pairs 
    for Fine-Tuning a Support Assistant.

    ## Task
    Analyze the provided text and generate Question/Answer pairs (if some content is provided, otherwise return an empty list).

    ## Requirements

    - Instruction: Create a direct, natural question a technician would ask.
    - Output: Provide a precise, step-by-step technical answer.
    - ALWAYS include alarm codes, specific words used in the manuals, since these are useful for the future training.
    - Prefer complete, descriptive and technical question-answers, DO NOT divide the content into many different questions.
    - Avoid creating QA Pairs where the answers are useless and lack of context (e.g. Q: What does the alarm list contain? A: Alarm list)
    - Language: Match the manual's language. If the manual is in a specific language, the question MUST be in that language. Possible languages include English, Italian, German, Polish, Dutch.
    - Style: No greetings, no "according to the manual". **REALLY IMPORTANT**: Keep the EXACT same style of the manual, same tone and same words used.
    - Format: Output the result as a JSON object following the provided schema.
"""

OLD_SYS_PROMPT_QA_GENERATION = """
    Your goal is to transform raw technical manual text into high-quality Instruction-Following pairs 
    for Fine-Tuning a Support Assistant.

    ## Task
    Analyze the provided text and generate one Question/Answer pair (if some content is provided, otherwise return an empty list).

    ## Requirements

    - Instruction: Create a direct, natural question a technician would ask.
    - Output: Provide a precise, step-by-step technical answer.
    - Format: Output the result as a JSON object following the provided schema.
    - Language: Match the manual's language. If the manual is in Italian, the question MUST be in Italian.
    - Style: No greetings, no "according to the manual". **REALLY IMPORTANT**: Keep the EXACT same style of the manual, same tone and same words used.
    - ALWAYS include alarm codes, specific words used in the manuals, since these are useful for the future training.
    - Prefer complete, descriptive and technical question-answers, DO NOT divide the content into many different questions.
    - Avoid creating QA Pairs where the answers are useless and lack of context (e.g. Q: What does the alarm list contain? A: Alarm list)

    ## Example:
    - Text: 
        A0001\nA0001 - Door safety circuit not ok\nImmediate Stop\n| Possible causes | Remedies |\n| --- | --- |\n| The address module is wrong | Check the module address according to electric drawings |\n| The sensor is not positioned correctly | Check that the sensor is positioned correctly |\n| At least one of the sensors is faulty | Locate and replace the faulty sensor |\n| Auxiliaries are on, but the signal does not reach the PLC | Check circuit functioning and wiring, according to the electric diagrams |\n| Functional group | Electrical system |\n| --- | --- |\nSearch results\nNo results found
    - QA:
        [
            {
                "instruction": "What are the possible causes and remedies for alarm A0001?",
                "output": "Alarm A0001 - Door safety circuit not ok\n| Possible causes | Remedies |\n| --- | --- |\n| The address module is wrong | Check the module address according to electric drawings |\n| The sensor is not positioned correctly | Check that the sensor is positioned correctly |\n| At least one of the sensors is faulty | Locate and replace the faulty sensor |\n| Auxiliaries are on, but the signal does not reach the PLC | Check circuit functioning and wiring, according to the electric diagrams |\n| Functional group | Electrical system |\n| --- | --- |"
            }
        ]

    - Text:
        How can we help you today?\nFeatured content\n1\n2\n713\nSearch results\nNo results found\nCome possiamo aiutarti oggi?\nContenuto in evidenza\n1\n2\n713\nRisultati della ricerca\nNessun risultato trovato
    - QA:
        [] <-- Empty list, no useful content retrieved
"""

USR_PROMPT_QA_GENERATION = """
    ## Document
    **{title}**
    {text}

    ## QA Pairs Generated

"""

SYS_PROMPT_TEST_QA_GENERATION = """
    Your goal is to generate a HIGH-QUALITY Technical Evaluation pair for a Test Set.

    ## Task
    Analyze the provided text and generate ONE Question/Answer pair that is DIFFERENT from the training questions provided.

    ## Strategic Objective
    The test set must evaluate if the assistant understands the technical context. 
    If the training set already asked about the "remedy", you should ask about "causes", "pre-requisites", or "component identification" related to the same topic.
    
    ## Requirements

    - Instruction: A natural, but distinct question. Avoid the phrasing used in the Excluded Questions.
    - Output: Provide a precise, step-by-step technical answer.
    - ALWAYS include alarm codes, specific words used in the manuals, since these are useful for the future training.
    - Prefer complete, descriptive and technical question-answers, DO NOT divide the content into many different questions.
    - Avoid creating QA Pairs where the answers are useless and lack of context (e.g. Q: What does the alarm list contain? A: Alarm list)
    - CRITICAL: Ensure the question is technically challenging and requires reading the provided text, not just guessing.
    - Language: Match the manual's language. If the manual is in a specific language, the question MUST be in that language. Possible languages include English, Italian, German, Polish, Dutch.    - Style: No greetings, no "according to the manual". **REALLY IMPORTANT**: Keep the EXACT same style of the manual, same tone and same words used.
    - Format: Output the result as a JSON object following the provided schema.
"""

USR_PROMPT_TEST_QA_GENERATION = """
    ## Training Set Questions (AVOID GENERATING THESE AGAIN)

    {excluded_questions}

    ## Raw Text Document
    **{title}**

    {original_text}

    ## QA Pairs Generated
"""

SYS_PROMPT_LOMA = """
    You are Loma, a technical assistant. Use the provided documentation to answer questions. 
    If you don't know the answer, say you don't know.

    1. Always provide a detailed and complete explanation.
    2. Don't limit yourself to a short list, but explain the "why" behind each step.

    If the information isn't there, mention you don't have it.
"""