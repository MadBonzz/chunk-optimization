from openai import OpenAI
from Evaluation.src.prompts import context_evaluation as ce
from Evaluation.src.prompts import qa_generation as qa
from Evaluation.src.prompts import qa_eval as qae
import re

client = OpenAI(base_url="http://192.168.84.106:1234/v1", api_key="lm-studio")

qac_eval = qae.question_evaluation_prompt
chunk = {'question': 'What considerations does the Central Government have in mind when appointing a Controller of Insurance, as per Section 2B of the Securities and Exchange Board of India Act, 1992?',
        'answer': 'In appointing a Controller of Insurance under Section 2B of the Securities and Exchange Board of India Act, 1992, the Central Government must have due regard to two considerations: whether the person to be appointed has had experience in industrial, commercial or insurance matters, and whether such person has actuarial qualifications.',
        'context': 'under section 15K of the Securities and Exchange Board of India Act, 1992 (15 of 1992);\n 2A. Interpretation of certain words and expressions. — Words and expressions used and not\ndefined in this Act but defined in the Life Insurance Corporation Act, 1956 (31 of 1956), the General\nInsurance Business (Nationalisation) Act, 1972 (57 of 1972) and the Insurance Regulatory and\nDevelopment Authority Act, 1999 shall have the meanings respectively assigned to them in those Acts.\n 2B. Appointment of Controller of Insurance. — 6 If at any time, the Authority is superseded\nunder sub-section of section 19 of the Insurance Regulatory and Development Authority Act, 1999, the\nCentral Government may, by notification in the Official Gazette, appoint a person to be the Controller of\nInsurance till such time the Authority is reconstituted under sub-section of section 19 of that Act.\n In making any appointment under this section, the Central Government shall have due regard\nto the following considerations, namely, whether the person to be appointed has had experience in\nindustrial, commercial or insurance matters and whether such person has actuarial qualifications.\nPART II'}

generation = client.chat.completions.create(
                model='meta-llama-3.1-8b-instruct',
                messages=[
                    {
                        "role": "system",
                        "content": """You are an AI question evaluator. Given the context and the corresponding question,
                        evaluate the question on relevance, groundness and standalone levels of the question."""
                    },
                    {
                        "role": "user",
                        "content": qac_eval.format(
                            context = chunk['context'],
                            question = chunk['question']
                        )
                    }
                ],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "QuestionEvaluator",
                            "parameters": {
                                "type": "object",
                                "title": "QuestionEvaluator",
                                "properties": {
                                    "evaluation": {
                                        "title": "Evaluation",
                                        "type": "string",
                                        "description": "A detailed evaluation explaining why the question is appropriate or not for the given context based on relevance, groundness, and its ability to stand alone."
                                    },
                                    "relevance_rating": {
                                        "title": "Relevance Rating",
                                        "type": "number",
                                        "description": "A rating from 1 to 5 indicating how relevant the question is to the provided context."
                                    },
                                    "groundness_rating": {
                                        "title": "Groundness Rating",
                                        "type": "number",
                                        "description": "A rating from 1 to 5 indicating how well the context provides adequate information to answer the question."
                                    },
                                    "standalone_rating": {
                                        "title": "Standalone Rating",
                                        "type": "number",
                                        "description": "A rating from 1 to 5 indicating how well the question stands on its own without needing additional context."
                                    }
                                },
                                "required": [
                                    "evaluation",
                                    "relevance_rating",
                                    "groundness_rating",
                                    "standalone_rating"
                                ]
                            }
                        }
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "QuestionEvaluator"}},
                temperature=0.5,
            )

response = generation.choices[0].message.content
response = str(response)
pattern = re.compile(
    r'"evaluation"\s*:\s*"(?P<evaluation>(?:\\.|[^"\\])*)"[^}]*'
    r'"relevance_rating"\s*:\s*(?P<relevance_rating>\d+),\s*'
    r'"groundness_rating"\s*:\s*(?P<groundness_rating>\d+),\s*'
    r'"standalone_rating"\s*:\s*(?P<standalone_rating>\d+)',
    re.DOTALL
)

match = pattern.search(response)
result = None
if match:
    result = match.groupdict()
    result['relevance_rating'] = int(result['relevance_rating'])
    result['groundness_rating'] = int(result['groundness_rating'])
    result['standalone_rating'] = int(result['standalone_rating'])
    for key in chunk.keys():
        result[key] = chunk[key]
    print(result)
else:
    print(response)
